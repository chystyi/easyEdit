"""Command-line entry point for the Boost pipeline.

Usage:
    python -m boost.cli "input.mp4" --title "Lesson Title" --out ./output
    python -m boost.cli "input.mp4" --title "Test" --out ./output --clip 20
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile

from .pipeline import run_pipeline


def main() -> int:
    ap = argparse.ArgumentParser(description="Boost lesson video editor")
    ap.add_argument("input", help="raw Loom mp4")
    ap.add_argument("--title", required=True, help="lesson title (used in intro + filename)")
    ap.add_argument("--out", default="./output", help="output directory")
    ap.add_argument("--clip", type=float, default=0,
                    help="process only the first N seconds (quick test)")
    # intro/outro default off (config); use --intro/--outro to force on
    ap.add_argument("--intro", dest="intro", action="store_true", default=None)
    ap.add_argument("--no-intro", dest="intro", action="store_false")
    ap.add_argument("--outro", dest="outro", action="store_true", default=None)
    ap.add_argument("--no-outro", dest="outro", action="store_false")
    args = ap.parse_args()

    src = args.input
    if args.clip > 0:
        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-i", src, "-t", str(args.clip),
             "-c", "copy", tmp], check=True)
        src = tmp

    def progress(msg: str, frac: float) -> None:
        print(f"[{frac*100:5.1f}%] {msg}", flush=True)

    res = run_pipeline(
        src, args.title, args.out,
        progress=progress,
        make_intro=args.intro,   # None -> config default (off)
        make_outro=args.outro,
    )
    print(f"\nOutput: {res.output}")
    print(f"Webcam auto-detected: {res.bubble_detected}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
