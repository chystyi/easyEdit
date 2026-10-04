"""Batch-render every .mp4 in a folder through the Boost pipeline (dev/QA tool).

Usage:
    python scripts/run_batch.py [INPUT_DIR] [OUTPUT_DIR]

Defaults: INPUT_DIR=./samples  OUTPUT_DIR=./output/batch
The lesson title is derived from each file name. SCI_FINAL.mp4 (the reference)
is skipped.
"""
import os
import sys
import time

# allow running as `python scripts/run_batch.py` from the project root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from boost.pipeline import run_pipeline  # noqa: E402

in_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "samples")
out_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "output", "batch")

files = sorted(f for f in os.listdir(in_dir)
               if f.lower().endswith(".mp4") and f != "SCI_FINAL.mp4")
if not files:
    print(f"No .mp4 files in {in_dir}")
    sys.exit(0)

for fname in files:
    src = os.path.join(in_dir, fname)
    title = os.path.splitext(fname)[0]
    t0 = time.time()
    print(f"\n=== {title} ===", flush=True)
    try:
        res = run_pipeline(src, title, out_dir,
                           progress=lambda m, f: print(f"  [{f*100:4.0f}%] {m}", flush=True))
        print(f"  OK -> {res.output} (bubble={res.bubble_detected}) "
              f"in {time.time()-t0:.0f}s", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"  FAILED: {exc}", flush=True)
print("\nBATCH DONE", flush=True)
