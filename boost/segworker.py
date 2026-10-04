"""
Per-segment background-removal worker (one OS process per CPU core).

Run as:  python -m boost.segworker <src> <start> <dur|-> <fps> <w> <h> <box|-> <out>
  box = "cx,cy,r" inpaint circle over the source webcam bubble, or
        "x0,y0,x1,y1" legacy white-fill rectangle, or "-" for none.

Launched by edit.clean_grey to parallelise the (single-threaded) OpenCV pass
across cores without relying on multiprocessing spawn semantics — robust under
uvicorn on Windows.
"""
from __future__ import annotations

import subprocess
import sys

import cv2
import numpy as np

from .detect import (remove_background, remove_ui_panel, remove_taskbar,
                     remove_pdf_badge, remove_top_toolbar, remove_cursor)


def main(argv: list[str]) -> int:
    src, start, dur, fps, w, h, box, out = argv
    fps_i, w_i, h_i = int(fps), int(w), int(h)
    box_t = None if box == "-" else tuple(int(v) for v in box.split(","))

    dec_cmd = ["ffmpeg", "-v", "error", "-ss", start, "-i", src]
    if dur != "-":
        dec_cmd += ["-t", dur]
    dec_cmd += ["-r", fps, "-an", "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"]
    dec = subprocess.Popen(dec_cmd, stdout=subprocess.PIPE)
    enc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y",
         "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w_i}x{h_i}", "-r", fps,
         "-i", "pipe:0", "-an",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
         "-pix_fmt", "yuv420p", out],
        stdin=subprocess.PIPE,
    )
    frame_bytes = w_i * h_i * 3

    # Pre-compute the (static) inpaint mask once instead of per frame.
    inp_mask = None
    if box_t is not None and len(box_t) == 3:
        bcx, bcy, brr = box_t
        inp_mask = np.zeros((h_i, w_i), np.uint8)
        cv2.circle(inp_mask, (bcx, bcy), brr, 255, -1)

    # Static-frame skipping: the webcam bubble (a face) moves every frame, but we
    # erase it anyway — so if the SLIDE (everything outside the bubble) is
    # unchanged, the cleaned frame is identical to the last one and we can reuse
    # it, skipping the expensive OpenCV/inpaint pass.
    #
    # CRITICAL: detect LOCAL changes, not the mean. A pen stroke / highlight is a
    # tiny area, so a mean-diff barely moves and the frame gets wrongly reused —
    # which made annotations appear late / all at once. We instead COUNT how many
    # pixels actually changed and only skip when essentially nothing did, so live
    # highlighting always renders in real time. Set BOOST_STATIC_SKIP=0 to disable.
    import os as _os
    STATIC_SKIP = _os.environ.get("BOOST_STATIC_SKIP", "1") == "1"
    DS = 4                                        # fine enough to catch thin strokes
    PIX_THRESH = 14                              # per-pixel change above encoder noise
    MAX_CHANGED = 3                              # skip only if ≤ this many px changed
    sw2, sh2 = max(1, w_i // DS), max(1, h_i // DS)
    excl = np.ones((sh2, sw2), bool)             # True = compare, False = bubble
    if box_t is not None:
        if len(box_t) == 3:
            m = np.ones((sh2, sw2), np.uint8)
            cv2.circle(m, (box_t[0] // DS, box_t[1] // DS),
                       box_t[2] // DS + 2, 0, -1)
            excl = m > 0
        else:
            x0, y0, x1, y1 = box_t
            excl[y0 // DS:y1 // DS, x0 // DS:x1 // DS] = False
    prev_small = None
    prev_out = None
    static_run = 0                               # consecutive static frames seen

    # Content-eating guard: the cleaners may only remove page CHROME (canvas,
    # toolbar, scrollbar) — all of which lives at the frame edges. If a pass
    # wipes a large amount of ink from the CENTRE of the page (where the
    # document content is), it just ate a table/diagram/annotation (the exact
    # bug the client kept catching), so those pixels are restored from the
    # original frame. The zone is well inside every viewer's chrome.
    gz_y0, gz_y1 = int(h_i * 0.15), int(h_i * 0.95)
    gz_x0, gz_x1 = int(w_i * 0.20), int(w_i * 0.80)
    if box_t is not None and len(box_t) == 3:    # keep the bubble circle out of it
        guard_excl = np.zeros((gz_y1 - gz_y0, gz_x1 - gz_x0), bool)
        bcx2, bcy2, brr2 = box_t
        gm = np.zeros((gz_y1 - gz_y0, gz_x1 - gz_x0), np.uint8)
        cv2.circle(gm, (bcx2 - gz_x0, bcy2 - gz_y0), brr2 + 8, 1, -1)
        guard_excl = gm.astype(bool)
    else:
        guard_excl = np.zeros((gz_y1 - gz_y0, gz_x1 - gz_x0), bool)

    while True:
        buf = dec.stdout.read(frame_bytes)
        if len(buf) < frame_bytes:
            break
        frame = np.frombuffer(buf, np.uint8).reshape(h_i, w_i, 3).copy()

        static = False
        if STATIC_SKIP:
            small = cv2.cvtColor(
                cv2.resize(frame, (sw2, sh2), interpolation=cv2.INTER_AREA),
                cv2.COLOR_BGR2GRAY).astype(np.int16)
            if prev_small is not None:
                d = np.abs(small - prev_small)
                changed = int(((d > PIX_THRESH) & excl).sum())
                static = changed <= MAX_CHANGED
            prev_small = small

        # Reuse the previous cleaned frame only after TWO consecutive static
        # detections: a single decode glitch used to freeze one corrupted frame
        # for seconds ("static-skip propagates one bad frame").
        static_run = static_run + 1 if static else 0
        if static_run >= 2 and prev_out is not None:
            enc.stdin.write(prev_out)            # reuse last cleaned frame
            continue

        zone_before = frame[gz_y0:gz_y1, gz_x0:gz_x1].copy()
        ink_before = (cv2.cvtColor(zone_before, cv2.COLOR_BGR2GRAY) < 200)
        ink_before &= ~guard_excl

        remove_taskbar(frame)                        # before bg removal: icons still coloured
        remove_pdf_badge(frame)                      # BEFORE bg removal too: the flood
        # eats the faded badge's pale body (neutral sat) and leaves a micro-core
        # below the detector's threshold — removing it while intact is reliable
        remove_ui_panel(frame)                       # BEFORE bg removal as well: its
        # light-background gate must see the RAW corner — after the flood the
        # margin around a dark poster is already white and the gate passed,
        # letting the detector eat the poster's left column
        remove_background(frame)
        if _os.environ.get("BOOST_REMOVE_CURSOR", "1") == "1":
            remove_cursor(frame)                 # parked mouse pointer on the page

        zone_after = frame[gz_y0:gz_y1, gz_x0:gz_x1]
        ink_after = (cv2.cvtColor(zone_after, cv2.COLOR_BGR2GRAY) < 200)
        eaten = ink_before & ~ink_after
        n_before = int(ink_before.sum())
        n_eaten = int(eaten.sum())
        if n_eaten > 4000 and n_eaten > 0.10 * max(1, n_before):
            # cleanup wiped a big block of page content — put those pixels back
            zone_after[eaten] = zone_before[eaten]

        if inp_mask is not None:
            frame[:] = cv2.inpaint(frame, inp_mask, 3, cv2.INPAINT_TELEA)
        elif box_t is not None:                  # legacy white-fill rectangle
            x0, y0, x1, y1 = box_t
            frame[y0:y1, x0:x1] = (255, 255, 255)
        prev_out = frame.tobytes()
        enc.stdin.write(prev_out)
    enc.stdin.close()
    dec.wait()
    return enc.wait()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
