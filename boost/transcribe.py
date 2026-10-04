"""
Speech-to-text with word-level timestamps (faster-whisper).

Used by the repeated-line detector: we need an accurate transcript *and* precise
timings so flagged redundant takes map back onto the timeline. faster-whisper
(CTranslate2) runs locally on CPU, no torch required.
"""
from __future__ import annotations

import subprocess
import tempfile
import os

from . import config as C

_model = None
_model_name = None


def _load(name: str):
    global _model, _model_name
    if _model is None or _model_name != name:
        from faster_whisper import WhisperModel
        # cpu_threads=0 -> CTranslate2 uses all cores.
        _model = WhisperModel(name, device="cpu", compute_type="int8",
                              cpu_threads=0)
        _model_name = name
    return _model


def transcribe(src: str, model_name: str = None,
               language: "str | None" = None) -> list:
    """Return a list of segments: {id, start, end, text, words:[{start,end,word}]}.

    `src` may be audio or video; the audio is extracted to 16 kHz mono first.
    """
    model_name = model_name or C.WHISPER_MODEL
    wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src,
                    "-ac", "1", "-ar", "16000", wav], check=True)
    try:
        # condition_on_previous_text=False stops Whisper's "repeat loop"
        # hallucination. NOTE: no_repeat_ngram_size was set here as a second
        # guard and that was an own-goal — it forbids the model from writing a
        # repeated phrase, i.e. it ERASED exactly the stumbles the repeat
        # detector exists to find. Never re-add it.
        # language=None -> auto-detect per video: these lessons include
        # French/Irish classes that were being force-transcribed as English,
        # which produced garbled text no detector could work with.
        segs, _ = _load(model_name).transcribe(
            wav, language=language, word_timestamps=True, vad_filter=True,
            condition_on_previous_text=False)
        out = []
        for i, s in enumerate(segs):
            out.append({
                "id": i,
                "start": float(s.start),
                "end": float(s.end),
                "text": s.text.strip(),
                "words": [{"start": float(w.start), "end": float(w.end),
                           "word": w.word} for w in (s.words or [])],
            })
        return out
    finally:
        os.unlink(wav)

def detect_language_spans(src: str, spans: list, model_name: str = None) -> list:
    """Detect the spoken language of each (start, end) span, from the AUDIO.

    Needed because Whisper writes French speech as phonetically-similar English
    ("Je suis simpa, sociable" -> "I am simple, sociable"), so a French line and
    its English translation look like a verbatim repeat in the transcript. The
    audio does not lie: if the two takes are different languages it is the
    lesson (say it in French, then in English), never a mistake.

    Returns [(lang, probability), ...] aligned with `spans`.
    """
    model = _load(model_name or C.WHISPER_MODEL)
    wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src,
                    "-ac", "1", "-ar", "16000", wav], check=True)
    out = []
    try:
        import numpy as np
        import wave
        with wave.open(wav, "rb") as wf:
            sr = wf.getframerate()
            raw = wf.readframes(wf.getnframes())
        pcm = np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0
        for (a, b) in spans:
            i0, i1 = int(max(0.0, a) * sr), int(max(0.0, b) * sr)
            chunk = pcm[i0:i1]
            if len(chunk) < sr // 2:                 # too short to judge
                out.append((None, 0.0))
                continue
            try:
                lang, prob, _all = model.detect_language(audio=chunk)
                out.append((lang, float(prob)))
            except Exception:  # noqa: BLE001
                out.append((None, 0.0))
    finally:
        os.unlink(wav)
    return out
