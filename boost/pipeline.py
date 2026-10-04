"""End-to-end orchestration of the Boost editing pipeline."""
from __future__ import annotations

import os
import shutil
import tempfile
import threading
from dataclasses import dataclass
from typing import Callable, Optional

from . import config as C
from . import edit
from .detect import detect_webcam_bubble

Progress = Callable[[str, float], None]   # (message, fraction 0..1)


@dataclass
class Result:
    output: str
    bubble_detected: bool
    repeats: Optional[list] = None          # repeated-line candidates (if requested)
    repeats_error: Optional[str] = None
    transcript: Optional[list] = None       # slim transcript (for re-search)
    warnings: Optional[list] = None         # non-fatal issues (e.g. Cleanvoice skipped)


def run_pipeline(
    src: str,
    title: str,
    out_dir: str,
    *,
    work_dir: Optional[str] = None,
    progress: Optional[Progress] = None,
    make_intro: Optional[bool] = None,
    make_outro: Optional[bool] = None,
    detect_repeats: bool = False,
    precleaned: Optional[str] = None,
    save_cleaned: Optional[str] = None,
) -> Result:
    if make_intro is None:
        make_intro = C.ADD_INTRO
    if make_outro is None:
        make_outro = C.ADD_OUTRO

    def report(msg: str, frac: float) -> None:
        if progress:
            progress(msg, frac)

    os.makedirs(out_dir, exist_ok=True)
    work = work_dir or tempfile.mkdtemp(prefix="boost_work_")
    os.makedirs(work, exist_ok=True)

    clean = os.path.join(work, "clean.mp4")
    body = os.path.join(work, "body.mp4")
    cut = os.path.join(work, "cut.mp4")
    intro = os.path.join(work, "intro.mp4")
    safe_name = "".join(c for c in title if c.isalnum() or c in " -_").strip() or "boost"
    final = os.path.join(out_dir, f"{safe_name}.mp4")

    # Pre-step: content cleanup — local VAD dead-time removal + (optional)
    # Cleanvoice fillers/coughs/stutters, in a single re-encode. Shortens the
    # source so the rest of the pipeline also processes less.
    #
    # `precleaned`: a cached cleanup result from a previous render — reuse it and
    # skip content_cleanup entirely (no Cleanvoice call, no credits, faster). Used
    # by "visual-only re-render" (e.g. to apply a background fix like taskbar
    # removal without redoing the audio work).
    src_proc = src
    warnings: list = []
    if precleaned and os.path.exists(precleaned):
        report("Reusing cleaned audio/timeline (no Cleanvoice)…", 0.05)
        src_proc = precleaned
    elif C.AUTOCUT:
        report("Cleaning dead time / fillers…", 0.03)
        try:
            if edit.content_cleanup(src, cut, warn=warnings.append):
                src_proc = cut
                if save_cleaned:
                    try:
                        shutil.copyfile(cut, save_cleaned)
                    except Exception:  # noqa: BLE001 — caching is best-effort
                        pass
        except Exception:  # noqa: BLE001 — never fail the whole job on cleanup
            src_proc = src

    # Repeat detection (transcript + LLM) runs in parallel with the heavy video
    # work below. It transcribes the cleaned source `src_proc`, whose timeline
    # equals the final output's (clean_grey/composite don't change timing), so
    # the candidate timestamps line up with the rendered video.
    rt_box: dict = {}
    rt = None
    if detect_repeats:
        from . import repeats as _repeats

        def _run_detect() -> None:
            try:
                cands, transcript = _repeats.analyze(src_proc)
                rt_box["cands"] = cands
                rt_box["transcript"] = transcript
            except Exception as exc:  # noqa: BLE001 — never fail render on this
                rt_box["error"] = str(exc)

        report("Transcribing for repeats (in parallel)…", 0.12)
        rt = threading.Thread(target=_run_detect, daemon=True)
        rt.start()

    report("Detecting webcam bubble…", 0.10)
    bubble = detect_webcam_bubble(src_proc)

    report("Removing grey background + masking page breaks…", 0.18)
    edit.clean_grey(src_proc, clean, paint_bubble=bubble)

    report("Compositing layers…", 0.50)
    edit.composite(clean, src_proc, bubble, body)

    # NOTE: the old webcam-surround "artifact repair" stage is gone for good —
    # it erased the teacher's annotations near the bubble. The webcam area is
    # sacred: it is overlaid LAST from the untouched source and nothing in the
    # pipeline may repaint pixels around it.

    if make_intro or make_outro:
        report("Building intro / outro…", 0.80)
        parts_body = body
        if make_intro:
            edit.make_intro(title, intro)
        if make_intro and make_outro:
            edit.assemble(intro, parts_body, C.ASSETS.outro, final)
        elif make_intro:
            edit.assemble(intro, parts_body, parts_body, final)  # rare path
        else:
            # outro only
            tmp_intro = os.path.join(work, "blank_intro.mp4")
            edit.make_intro(title, tmp_intro, duration=0.1)
            edit.assemble(tmp_intro, parts_body, C.ASSETS.outro, final)
    else:
        os.replace(body, final)

    if rt is not None:
        report("Finishing transcript…", 0.96)
        rt.join()

    report("Done.", 1.0)
    return Result(output=final, bubble_detected=bubble.detected,
                  repeats=rt_box.get("cands"), repeats_error=rt_box.get("error"),
                  transcript=rt_box.get("transcript"), warnings=warnings or None)
