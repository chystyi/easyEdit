"""
Speech detection (VAD) for the content-cleanup pass.

Replaces the old amplitude-threshold silence detector. A waveform threshold can't
separate quiet speech (~-40 dB) from dead-zone noise / mouse clicks (also
~-40..-28 dB) — their levels overlap, so any threshold either keeps junk or clips
quiet words. Silero VAD classifies *speech vs non-speech* by acoustic content, so
it cuts the start fumbling and dead air while preserving quiet real speech.

Returns speech time-spans; `speech_keep_ranges` turns them into the keep-list the
auto-cut renders (cutting long gaps + leading/trailing dead air, keeping natural
short pauses).
"""
from __future__ import annotations

import subprocess

import numpy as np

SR = 16000
_model = None


def _load():
    global _model
    if _model is None:
        from silero_vad import load_silero_vad
        _model = load_silero_vad(onnx=True)
    return _model


def _decode_mono16k(src: str) -> np.ndarray:
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", src, "-ac", "1", "-ar", str(SR),
         "-f", "f32le", "-"],
        capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


def detect_speech(src: str, threshold: float = 0.5) -> list:
    """Return speech spans as a list of (start_s, end_s)."""
    from silero_vad import get_speech_timestamps
    audio = _decode_mono16k(src)
    if audio.size == 0:
        return []
    ts = get_speech_timestamps(
        audio, _load(), sampling_rate=SR, threshold=threshold,
        min_speech_duration_ms=200, min_silence_duration_ms=200,
        return_seconds=True,
    )
    return [(float(t["start"]), float(t["end"])) for t in ts]


def speech_keep_ranges(segments: list, dur: float,
                       min_silence: float, pad: float) -> list:
    """Convert speech spans into keep-ranges.

    - gaps between speech shorter than `min_silence` are KEPT (natural pauses);
    - gaps >= `min_silence` (incl. leading/trailing dead air) are CUT, leaving
      `pad` seconds of breathing room on each side of speech.
    """
    if not segments:
        return []
    merged = []
    for s, e in segments:
        if merged and s - merged[-1][1] < min_silence:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append([s, e])

    keep = []
    for s, e in merged:
        a, b = max(0.0, s - pad), min(dur, e + pad)
        if keep and a <= keep[-1][1]:
            keep[-1][1] = max(keep[-1][1], b)
        else:
            keep.append([a, b])
    return [(a, b) for a, b in keep if b - a > 0.1]
