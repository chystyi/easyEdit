"""
Concatenate several finished renders into one video.

Some lessons are recorded in two takes and land in the SAME Drive folder as
"part 1 / part 2" (client, 2026-09-21: "House: Part Two has 2 videos in it
that need to be merged into one"). Each take goes through the pipeline on its
own — different bubble geometry, different slide extents — so they can only
be joined AFTER rendering, when both are plain 1920×1080 outputs.

Two paths:
  * every part shares codec / size / fps / audio layout  ->  concat demuxer
    with stream copy: instant and bit-exact, no second encode;
  * anything differs (e.g. one part came out of the editor with different
    encoder settings)                                     ->  concat filter
    with a full re-encode at the pipeline's own quality.
Either way the result is checked against the summed input length so a
silently truncated join can never ship.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from typing import Callable, List, Optional

from . import config as C
from .edit import probe

Progress = Callable[[str, float], None]


def _stream_signature(path: str) -> tuple:
    """Everything that must match for a copy-concat to be valid."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "stream=codec_type,codec_name,width,height,r_frame_rate,pix_fmt,"
         "sample_rate,channels,profile",
         "-of", "csv=p=0", path],
        capture_output=True, text=True).stdout
    return tuple(sorted(line.strip() for line in out.splitlines() if line))


def _duration(path: str) -> float:
    return float(probe(path)["duration"])


def merge_videos(parts: List[str], dst: str,
                 progress: Optional[Progress] = None) -> dict:
    """Join `parts` in order into `dst`. Returns {"duration", "copied"}."""
    if len(parts) < 2:
        raise ValueError("need at least two parts to merge")
    for p in parts:
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    def report(m: str, f: float) -> None:
        if progress:
            progress(m, f)

    expected = sum(_duration(p) for p in parts)
    same = len({_stream_signature(p) for p in parts}) == 1
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)

    if same:
        report("Joining parts (stream copy)…", 0.2)
        lst = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False)
        for p in parts:
            # concat demuxer quoting: single quotes, with embedded ' escaped
            lst.write("file '" + p.replace("'", "'\\''") + "'\n")
        lst.close()
        r = subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0",
             "-i", lst.name, "-c", "copy", "-movflags", "+faststart", dst],
            capture_output=True, text=True)
        os.unlink(lst.name)
        if r.returncode != 0:
            same = False                      # fall through to the re-encode
            report("Stream copy refused — re-encoding…", 0.3)

    if not same:
        report("Joining parts (re-encode)…", 0.3)
        cmd = ["ffmpeg", "-v", "error", "-y"]
        for p in parts:
            cmd += ["-i", p]
        n = len(parts)
        # normalise every part to the output format first so concat never
        # sees a size / rate mismatch
        fc = "".join(
            f"[{i}:v]scale=1920:1080:force_original_aspect_ratio=decrease,"
            f"pad=1920:1080:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={C.OUT_FPS},"
            f"format=yuv420p[v{i}];"
            f"[{i}:a]aresample=44100,aformat=channel_layouts=stereo[a{i}];"
            for i in range(n))
        fc += "".join(f"[v{i}][a{i}]" for i in range(n))
        fc += f"concat=n={n}:v=1:a=1[outv][outa]"
        cmd += ["-filter_complex", fc, "-map", "[outv]", "-map", "[outa]",
                "-r", str(C.OUT_FPS), "-c:v", "libx264", "-preset", "medium",
                "-crf", "17", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", dst]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"merge failed: {r.stderr[-800:]}")

    got = _duration(dst)
    if abs(got - expected) > 0.5 + 0.002 * expected:
        raise RuntimeError(
            f"merged length {got:.1f}s does not match the parts "
            f"({expected:.1f}s) — refusing to ship a truncated join")
    report("Done.", 1.0)
    return {"duration": got, "copied": same}
