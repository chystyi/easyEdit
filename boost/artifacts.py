"""
Auto-remove stray artifacts (accidental pen flicks / marks that appear next to
the webcam or in the blank margins & sides) from the composited output — WITHOUT
touching slide text/images, the webcam face, the robot, or the teacher's
INTENTIONAL annotations.

Why it is safe:
  * It only repairs pixels OUTSIDE the slide-content bounding box (text & images
    are never touched) and outside the robot.
  * Inside that margin / webcam-surround band it repairs only STATIC pixels (low
    temporal deviation) — so the moving face is never touched, and a persistent
    mark (which the median treats as the real background) is left alone.
  * The replacement is the per-pixel temporal MEDIAN — the genuine clean
    background at that spot (white margin, grey webcam halo, …) — so there is
    never a visible patch or colour mismatch.

Effect: a brief coloured stroke in a normally-blank spot is swapped for the clean
background it briefly covered; every other pixel is left exactly as it was.

Intentional annotations sit ON/near the text (inside the content box) and are
therefore never removed — distinguishing a stray mark from a deliberate underline
in the content area is not reliable, so we deliberately don't try.
"""
from __future__ import annotations

import subprocess

import cv2
import numpy as np

from . import config as C


def _probe(src: str, key: str, default: str) -> str:
    stream = "format=duration" if key == "duration" else f"stream={key}"
    sel = [] if key == "duration" else ["-select_streams", "v:0"]
    out = subprocess.run(["ffprobe", "-v", "error", *sel, "-show_entries", stream,
                          "-of", "csv=p=0", src], capture_output=True, text=True).stdout.strip()
    return out.splitlines()[0] if out else default


def _sample_frames(src: str, n: int, W: int, H: int) -> list:
    dur = float(_probe(src, "duration", "0") or 0)
    frames = []
    for i in range(n):
        t = dur * (i + 0.5) / n
        out = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.2f}", "-i", src,
                              "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "bgr24",
                              "pipe:1"], capture_output=True).stdout
        if len(out) == W * H * 3:
            frames.append(np.frombuffer(out, np.uint8).reshape(H, W, 3))
    return frames


def _robot_box(W: int, H: int) -> tuple:
    r = cv2.imread(C.ASSETS.robot, cv2.IMREAD_UNCHANGED)
    rw, rh = int(r.shape[1] * C.ROBOT_SCALE), int(r.shape[0] * C.ROBOT_SCALE)
    cx, cy = C.ROBOT_CENTER
    return (max(0, int(cx - rw / 2 - 24)), max(0, int(cy - rh / 2 - 24)),
            min(W, int(cx + rw / 2 + 24)), min(H, int(cy + rh / 2 + 48)))


def _content_box(frames: list, webmask: np.ndarray, robotbox: tuple) -> "tuple | None":
    """Union bounding box of slide ink across the video (excl. webcam & robot)."""
    xs_all, ys_all = [], []
    rx0, ry0, rx1, ry1 = robotbox
    for fr in frames:
        ink = fr.min(axis=2) < 232
        ink &= ~webmask
        ink[ry0:ry1, rx0:rx1] = False
        ys, xs = np.where(ink)
        if len(xs):
            xs_all.append(xs)
            ys_all.append(ys)
    if not xs_all:
        return None
    xs, ys = np.concatenate(xs_all), np.concatenate(ys_all)
    return (int(np.percentile(xs, 1)), int(np.percentile(ys, 1)),
            int(np.percentile(xs, 99)), int(np.percentile(ys, 99)))


def repair(src: str, dst: str, *, n_sample: int = 48, var_thresh: int = 12,
           diff_thresh: int = 26, margin: int = 44) -> bool:
    """Repair stray margin/webcam-surround artifacts in `src` -> `dst` (audio
    copied). Returns True if it ran (dst written), False if it passed through."""
    W, H = C.OUT_W, C.OUT_H
    fps_raw = _probe(src, "r_frame_rate", "30/1").strip().rstrip(", ")
    if "/" in fps_raw:
        a, b = fps_raw.split("/")
        fps = f"{float(a) / float(b):.4f}"
    else:
        fps = fps_raw or "30"
    frames = _sample_frames(src, n_sample, W, H)
    if len(frames) < 8:
        return False

    stack = np.stack(frames).astype(np.int16)            # (n,H,W,3)
    plate = np.empty((H, W, 3), np.uint8)
    mad = np.empty((H, W, 3), np.uint8)
    for y0 in range(0, H, 120):                          # band-wise to bound memory
        b = stack[:, y0:y0 + 120]
        med = np.median(b, axis=0)
        plate[y0:y0 + 120] = np.clip(med, 0, 255).astype(np.uint8)
        mad[y0:y0 + 120] = np.clip(np.median(np.abs(b - med), axis=0), 0, 255).astype(np.uint8)

    # Confine repair to the WEBCAM-SURROUND ring — the band straddling the bubble
    # edge out to ~80px beyond it. This is where the recurring stray strokes land,
    # and slide content is never authored on top of the presenter, so it is
    # reliably non-content. Everything else (text, images, the bottom-of-slide
    # picture, the sides) is left completely untouched — those areas can hold
    # per-slide content that a whole-video median would wrongly treat as an
    # outlier, so we deliberately don't repair there.
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    wcx, wcy, wr = C.WEBCAM_OUT_X * W, C.WEBCAM_OUT_Y * H, C.WEBCAM_OUT_R
    r2 = (xx - wcx) ** 2 + (yy - wcy) ** 2
    ring = (r2 > (wr - 22) ** 2) & (r2 < (wr + 80) ** 2)
    static = mad.max(axis=2) < var_thresh                # excludes the moving face/shoulder
    repair_zone = ring & static

    # Persistent bottom-right corner mark (the Acrobat "PDF" icon) — a small
    # non-white blob on white that never moves, so the temporal median keeps it.
    # Detect it on the plate (small, isolated, on white) and whiten it every frame.
    white_mask = np.zeros((H, W), bool)
    white_fill = (255, 255, 255)
    cz = plate[H - 110:H - 34, W - 190:W - 60]
    cnw = cz.min(axis=2) < 232
    if 15 < int(cnw.sum()) < 1500:                       # a small mark, not empty/an image
        ys, xs = np.where(cnw)
        wpx = cz[~cnw]
        if len(wpx):
            white_fill = tuple(int(v) for v in np.median(wpx, axis=0))
        white_mask[H - 110 + ys.min() - 5:H - 110 + ys.max() + 6,
                   W - 190 + xs.min() - 5:W - 190 + xs.max() + 6] = True

    act = repair_zone | white_mask
    if not act.any():
        return False
    # only the sub-region covering the ring + corner is ever modified
    ys, xs = np.where(act)
    ry0, ry1, rx0, rx1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    rz = repair_zone[ry0:ry1, rx0:rx1]
    wm = white_mask[ry0:ry1, rx0:rx1]
    rplate = plate[ry0:ry1, rx0:rx1].astype(np.int16)

    dec = subprocess.Popen(["ffmpeg", "-v", "error", "-i", src, "-an",
                            "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"],
                           stdout=subprocess.PIPE)
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y",
                            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
                            "-r", fps, "-i", "pipe:0", "-i", src,
                            "-map", "0:v", "-map", "1:a?", "-c:a", "copy",
                            "-c:v", "libx264", "-crf", "18", "-preset", "veryfast",
                            "-pix_fmt", "yuvj420p", dst], stdin=subprocess.PIPE)
    fbytes = W * H * 3
    while True:
        buf = dec.stdout.read(fbytes)
        if len(buf) < fbytes:
            break
        fr = np.frombuffer(buf, np.uint8).reshape(H, W, 3).copy()
        sub = fr[ry0:ry1, rx0:rx1]
        diff = np.abs(sub.astype(np.int16) - rplate).max(axis=2)
        m = rz & (diff > diff_thresh)
        if m.any():
            sub[m] = plate[ry0:ry1, rx0:rx1][m]
        if wm.any():
            sub[wm] = white_fill
        enc.stdin.write(fr.tobytes())
    enc.stdin.close()
    dec.wait()
    return enc.wait() == 0
