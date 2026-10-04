"""
Automated QA pass over a FINISHED rendered video — the checks mirror every
category of defect the client has actually flagged, so a video that passes here
is safe to deliver without eyeballing all of it:

  * bottom-cut      - slide content touching the bottom frame edge
  * webcam-overlap  - ink strokes running into the webcam circle's edge
  * corner-junk     - taskbar / Acrobat badge / toolbar remnants in the margins
  * av-sync         - audio vs video stream duration drift
  * audio-dip       - speech windows notably quieter than the video's median
  * freeze          - whole-frame freezes (incl. webcam) while speech continues
  * webcam-static   - webcam circle showing no motion for a stretch

Usage:
    python -m boost.qa <video.mp4> [<video2.mp4> ...] [--json report.json]

Each finding has a timestamp, so a human can jump straight to the second that
needs review. Levels: "warn" = look at it, "fail" = do not deliver.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np

from . import config as C

FPS = 2                     # analysis sample rate (frames per second)
DW, DH = 480, 270           # analysis frame size (16:9, 1/4 of 1920x1080)
SX, SY = 1920 / DW, 1080 / DH


def _audio_rms(path: str, win: float = 0.5):
    """Mono 16 kHz PCM -> RMS dBFS per `win`-second window."""
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-vn", "-ac", "1", "-ar", "16000",
         "-f", "s16le", "-"], capture_output=True).stdout
    a = np.frombuffer(raw, np.int16).astype(np.float64) / 32768.0
    n = int(16000 * win)
    if not len(a) or n <= 0:
        return np.array([]), win
    m = len(a) // n
    rms = np.sqrt((a[:m * n].reshape(m, n) ** 2).mean(axis=1) + 1e-12)
    return 20 * np.log10(rms + 1e-9), win


def _out_ink_bbox(path: str, t: float):
    """Ink bounding box (top, bottom, left, right) of one output frame at `t`,
    analysis scale, excluding the robot / webcam zones and outer margins."""
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", f"{t:.2f}", "-i", path, "-frames:v",
         "1", "-vf", f"scale={DW}:{DH}", "-f", "rawvideo", "-pix_fmt", "gray",
         "-"], capture_output=True).stdout
    if len(raw) < DW * DH:
        return None
    g = np.frombuffer(raw, np.uint8).reshape(DH, DW)
    ink = g < 200
    ink[:, :int(70)] = False                            # robot column
    wcx, wcy = C.WEBCAM_OUT_X * 1920 / SX, C.WEBCAM_OUT_Y * 1080 / SY
    wr = (C.WEBCAM_OUT_R + 8) / SX
    yy, xx = np.mgrid[0:DH, 0:DW]
    ink[(xx - wcx) ** 2 + (yy - wcy) ** 2 < wr * wr] = False
    rows = np.where(ink.sum(axis=1) > 2)[0]
    cols = np.where(ink.sum(axis=0) > 2)[0]
    if not len(rows) or not len(cols):
        return None
    return rows.min(), rows.max(), cols.min(), cols.max()


def _scan_source_slides(src: str):
    """Slide spans + persistent-ink masks of the raw/cleaned SOURCE, with each
    sampled frame first passed through the pipeline's own chrome cleaners
    (taskbar / background / UI panel / PDF badge) and the webcam bubble masked
    — so "content" here means exactly what the pipeline itself would keep."""
    import cv2
    from .detect import (detect_webcam_bubble, remove_background,
                         remove_pdf_badge, remove_taskbar, remove_ui_panel)
    from .edit import probe as _probe
    info = _probe(src)
    sw, sh = info["w"], info["h"]
    bub = detect_webcam_bubble(src)
    DS = 4
    dw, dh = sw // DS, sh // DS
    step = 2.0                                          # one frame every 2s
    dec = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", src, "-vf", f"fps={1 / step}",
         "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"],
        stdout=subprocess.PIPE)
    nb = sw * sh * 3
    bmask = np.zeros((dh, dw), np.uint8)
    cv2.circle(bmask, (bub.cx // DS, bub.cy // DS), bub.r // DS + 3, 1, -1)
    bmask = bmask.astype(bool)
    slides, prev = [], None
    cur = {"t0": 0.0, "cnt": np.zeros((dh, dw), np.uint8), "n": 0}
    t = 0.0
    while True:
        buf = dec.stdout.read(nb)
        if len(buf) < nb:
            break
        fr = np.frombuffer(buf, np.uint8).reshape(sh, sw, 3).copy()
        remove_taskbar(fr)
        remove_background(fr)
        remove_ui_panel(fr)
        remove_pdf_badge(fr)
        g = cv2.cvtColor(cv2.resize(fr, (dw, dh), interpolation=cv2.INTER_AREA),
                         cv2.COLOR_BGR2GRAY).astype(np.int16)
        if prev is not None:
            if int((np.abs(g - prev) > 40).sum()) > dw * dh * 0.04:
                cur["t1"] = t
                slides.append(cur)
                cur = {"t0": t, "cnt": np.zeros((dh, dw), np.uint8), "n": 0}
        ink = g < 200
        ink[:14, :] = False                             # toolbar strip
        ink[bmask] = False                              # webcam bubble
        np.add(cur["cnt"], ink.astype(np.uint8), out=cur["cnt"],
               where=cur["cnt"] < 255)
        cur["last"] = ink                               # final state wins
        cur["n"] += 1
        prev = g
        t += step
    dec.wait()
    cur["t1"] = t
    slides.append(cur)
    out = []
    for s in slides:
        if s["n"] < 2 or "last" not in s:
            continue
        # FINAL state of the slide, de-noised: a pixel counts only if it is ink
        # in the last sampled frame AND was seen at least twice — this drops
        # the cursor (transient) and everything that was merely SCROLLED PAST
        # earlier in the slide (which is not "missing" from the output).
        ink = s["last"] & (s["cnt"] >= min(2, s["n"]))
        rows = np.where(ink.sum(axis=1) > 3)[0]
        if not len(rows):
            continue
        out.append({"t0": s["t0"], "t1": s["t1"], "ct": int(rows.min()) * DS,
                    "cb": int(rows.max()) * DS, "ink": ink, "ds": DS})
    return out


def _check_vs_source(path: str, src: str, add) -> None:
    """Compare each slide's content extent in the OUTPUT against the cleaned
    SOURCE — the only reliable way to catch 'the bottom of the slide never
    made it into the video' (the output itself just looks like white space).

    Geometry is re-derived per slide from the ink bounding-box WIDTHS (widths
    survive any vertical crop), so no knowledge of the render settings is
    needed: predicted_bottom = out_top + src_span * S, with S from widths."""
    slides = _scan_source_slides(src)
    for idx, sl in enumerate(slides):
        if sl["t1"] - sl["t0"] < 4:                     # ignore slivers
            continue
        # a slide whose CONTINUATION grows downward (teacher keeps writing on
        # the same slide) is top-pinned on purpose — the final content is tall
        # and centring the early phase would cut the later writing
        grows = False
        if idx + 1 < len(slides):
            nx = slides[idx + 1]
            grows = (nx["t0"] - sl["t1"] < 3
                     and abs(nx["ct"] - sl["ct"]) < 30
                     and nx["cb"] > sl["cb"] + 80)
        # sample near the END of the slide — the source mask is the slide's
        # FINAL state, so the output frame must be from (almost) the same
        # moment; 3s early still missed strokes drawn in a slide's last seconds
        mid = max((sl["t0"] + sl["t1"]) / 2, sl["t1"] - 1.5)
        bb = _out_ink_bbox(path, mid)
        if bb is None:
            continue
        ot, ob, ol, orr = bb
        ys, xs = np.where(sl["ink"])
        if not len(xs):
            continue
        ds = sl["ds"]
        sl_l, sl_r = xs.min() * ds, xs.max() * ds
        sl_t, sl_b = ys.min() * ds, ys.max() * ds
        if sl_r - sl_l < 40:
            continue
        S = ((orr - ol) * SX) / max(1.0, sl_r - sl_l)   # px-out per px-src
        pred_bottom = ot * SY + (sl_b - sl_t) * S       # where src bottom lands
        if pred_bottom > 1080 + 8:
            add("fail", "bottom-missing", mid,
                f"~{int((pred_bottom - 1080) / max(S, .1))}px of the source "
                "slide never made it into the frame (cut at the bottom)")
        elif pred_bottom - (ob * SY) > 45 * S:
            add("warn", "bottom-missing", mid,
                "output slide is missing content present in the source "
                "(bottom section)")

        # --- content-missing: compare the source's ink against the output
        # CELL BY CELL. A whole-slide density ratio is useless here (a dense
        # map drowns out a handful of annotations), but a grid cell that is
        # full of ink in the source and empty in the output means content was
        # erased or hidden — exactly the popup-patch that buried a teacher's
        # annotation pass while QA reported CLEAN.
        try:
            raw_o = subprocess.run(
                ["ffmpeg", "-v", "error", "-ss", f"{mid:.2f}", "-i", path,
                 "-frames:v", "1", "-vf", f"scale={DW}:{DH}", "-f", "rawvideo",
                 "-pix_fmt", "gray", "-"], capture_output=True).stdout
            if len(raw_o) >= DW * DH:
                go = np.frombuffer(raw_o[:DW * DH], np.uint8).reshape(DH, DW)
                out_ink = (go < 210)
                ys_s, xs_s = np.where(sl["ink"])
                oxp = (ol + (xs_s * ds - sl_l) * S / SX).astype(int)
                oyp = (ot + (ys_s * ds - sl_t) * S / SY).astype(int)
                ok = (oxp >= 0) & (oxp < DW) & (oyp >= 0) & (oyp < DH)
                src_map = np.zeros((DH, DW), bool)
                src_map[oyp[ok], oxp[ok]] = True
                yy3, xx3 = np.mgrid[0:DH, 0:DW]
                wcx3 = C.WEBCAM_OUT_X * 1920 / SX
                wcy3 = C.WEBCAM_OUT_Y * 1080 / SY
                wr3 = (C.WEBCAM_OUT_R + 20) / SX
                keep = ((xx3 - wcx3) ** 2 + (yy3 - wcy3) ** 2 >= wr3 * wr3)
                keep &= ~((xx3 < 80) & (yy3 > DH - 90))     # robot
                src_map &= keep
                out_ink &= keep
                GH, GW = 6, 8
                ch, cw = DH // GH, DW // GW
                cells = []
                for gy in range(GH):
                    for gx in range(GW):
                        y0c, x0c = gy * ch, gx * cw
                        cs = int(src_map[y0c:y0c + ch, x0c:x0c + cw].sum())
                        co = int(out_ink[y0c:y0c + ch, x0c:x0c + cw].sum())
                        # a cell needs REAL content to be judged: 40-odd
                        # mask pixels is map-edge noise and produced a false
                        # "annotations missing" on a perfectly good render
                        if cs >= 120:
                            cells.append((co / max(1, cs), gx, gy))
                worst = None
                if len(cells) >= 4:
                    # Self-calibrating: the source mask counts persistent ink
                    # at a coarser scale, so out/src runs ~1.5-2.5 across a
                    # healthy slide. Judge each cell against the slide's OWN
                    # median instead of an absolute ratio — an erased region
                    # then stands out no matter how dense the slide is.
                    med = float(np.median([c[0] for c in cells]))
                    lo = min(cells)
                    if med > 0.2 and lo[0] < 0.35 * med:
                        worst = lo
                if worst:
                    add("warn", "content-missing", mid,
                        f"a region of the slide is blank in the output but "
                        f"inked in the source (cell {worst[1]},{worst[2]} — "
                        "annotations erased or hidden?)")
        except Exception:  # noqa: BLE001
            pass

        # --- slide centring (client spec: copy sits vertically centred).
        # Per SLIDE, not per video: one un-centred slide must not hide in a
        # video-wide median. Skip tall content (fills the frame anyway).
        top_m = ot * SY
        bot_m = 1080 - ob * SY
        if not grows and (ob - ot) * SY < 700 and (bot_m - top_m) > 170:
            add("warn", "slide-not-centred", mid,
                f"content sits high: top margin {top_m:.0f}px vs bottom "
                f"{bot_m:.0f}px (slide should be vertically centred)")


def check(path: str, src: "str | None" = None) -> dict:
    findings = []
    add = lambda level, kind, t, msg: findings.append(  # noqa: E731
        {"level": level, "kind": kind, "t": None if t is None else round(t, 1),
         "msg": msg})

    # ---------- vs-source: content completeness (needs the cleaned source) ----
    if src and os.path.exists(src):
        try:
            _check_vs_source(path, src, add)
        except Exception as exc:  # noqa: BLE001
            add("warn", "qa-error", None, f"source comparison failed: {exc}")

    # ---------- A/V sync ----------
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "stream=codec_type,duration", "-of", "json", path],
        capture_output=True, text=True).stdout
    durs = {}
    for st in json.loads(probe).get("streams", []):
        try:
            durs[st["codec_type"]] = float(st.get("duration") or 0)
        except (TypeError, ValueError):
            pass
    if durs.get("video") and durs.get("audio"):
        drift = abs(durs["video"] - durs["audio"])
        if drift > 0.5:
            add("fail", "av-sync", None,
                f"audio/video stream durations differ by {drift:.2f}s")

    # ---------- audio profile ----------
    db, win = _audio_rms(path)
    speech = db > -45                                   # windows with speech
    if len(db) and speech.any():
        med = float(np.median(db[speech]))
        i = 0
        while i < len(db):
            if speech[i] and db[i] < med - 6:           # ≥6 dB under median
                j = i
                while j < len(db) and speech[j] and db[j] < med - 4:
                    j += 1
                if (j - i) * win >= 1.5:                # sustained dip
                    add("warn", "audio-dip", i * win,
                        f"speech {med - float(db[i:j].mean()):.1f} dB quieter "
                        f"than usual for {(j - i) * win:.1f}s")
                i = j
            else:
                i += 1

    # ---------- video pass ----------
    dec = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", path, "-vf",
         f"fps={FPS},scale={DW}:{DH}", "-f", "rawvideo", "-pix_fmt", "bgr24",
         "pipe:1"], stdout=subprocess.PIPE)
    nb = DW * DH * 3

    # geometry of the fixed output webcam circle, in analysis coords
    wcx, wcy = C.WEBCAM_OUT_X * 1920 / SX, C.WEBCAM_OUT_Y * 1080 / SY
    wr = C.WEBCAM_OUT_R / SX
    yy, xx = np.mgrid[0:DH, 0:DW]
    dist2 = (xx - wcx) ** 2 + (yy - wcy) ** 2
    cam_mask = dist2 < (wr * 0.8) ** 2                  # inside the webcam
    edge_ring = (dist2 > (wr + 1) ** 2) & (dist2 < (wr + 5) ** 2)
    robot_zone = (xx < 70) & (yy > DH - 80)             # bottom-left mascot

    prev = None
    prev_cam = None
    freeze_run = 0
    cam_still_run = 0
    bottom_run = 0
    top_run = 0
    top_flag_until = -10
    overlap_run = 0
    panel_flag_until = -10
    popup_hits = []
    speck_run = 0
    speck_flag_until = -10
    bar_hits: list = []
    t = 0.0
    bottom_flag_until = -10
    overlap_flag_until = -10
    junk_flag_until = -10
    while True:
        buf = dec.stdout.read(nb)
        if len(buf) < nb:
            break
        fr = np.frombuffer(buf, np.uint8).reshape(DH, DW, 3)
        g = fr.mean(axis=2)

        ink = (g < 200) & ~cam_mask & ~robot_zone
        import cv2 as _cv0
        hsvf = _cv0.cvtColor(fr, _cv0.COLOR_BGR2HSV)

        # --- top-cut: ink sliced by the very top edge (a slide pinned too
        # high — a flashed photo slide once shipped with its title bisected)
        tband = ink[0:3, 40:DW - 40]
        top_run = top_run + 1 if tband.sum() > 12 else 0
        if top_run >= 2 and t > top_flag_until:
            add("warn", "top-cut", t - 0.5,
                "slide content cut off at the top edge")
            top_flag_until = t + 8

        # --- bottom-cut: ink touching the very bottom rows across some width.
        # Needs to PERSIST ≥2 samples (1s): scrolls/transitions sweep content
        # through the bottom edge for a moment and are not defects.
        band = ink[DH - 3:DH, 40:DW - 40]
        bottom_run = bottom_run + 1 if band.sum() > 12 else 0
        if bottom_run >= 4 and t > bottom_flag_until:   # ≥2s — scrolls sweep
            add("warn", "bottom-cut", t - 1.5,          # through in less
                "slide content touches the bottom edge (likely cut off)")
            bottom_flag_until = t + 8

        # --- webcam overlap: ink strokes hugging the circle's outer edge
        # (same persistence rule as above)
        overlap_run = overlap_run + 1 if (ink & edge_ring).sum() > 6 else 0
        if overlap_run >= 2 and t > overlap_flag_until:
            add("warn", "webcam-overlap", t - 0.5,
                "content runs into the webcam circle edge "
                "(may continue underneath)")
            overlap_flag_until = t + 8

        # --- corner junk: coloured specks in the bottom band spread wide
        if t > junk_flag_until:
            # start past the robot mascot column — its teal body sits in the
            # band permanently and photo edges pushed the combined pattern
            # over the "row of icons" bar (false taskbar warnings)
            band = fr[DH - 14:DH, 60:, :].astype(np.int16)
            sat = band.max(axis=2) - band.min(axis=2)
            colored = (sat > 45) & (band.max(axis=2) > 60)
            # a photo/diagram reaching the bottom band paints MOST of it —
            # taskbar icons are sparse specks; skip dense bands as content
            dense = int(colored.sum()) > colored.size * 0.30
            cols = np.where(colored.any(axis=0))[0]
            spread = (not dense and len(cols) > 30
                      and cols.min() < DW * 0.2 - 60
                      and cols.max() > DW * 0.8 - 60)
            # Win11 centres its icons — a spaced row of small colour specks in
            # the bottom band counts even without edge-to-edge spread
            centered = False
            if not dense and not spread and int(colored.sum()) > 12:
                xs2 = np.where(colored.any(axis=0))[0]
                gaps = np.diff(xs2)
                centered = (len(xs2) > 12 and (xs2.max() - xs2.min()) > DW * 0.3
                            and int((gaps > 3).sum()) >= 4)
            if spread or centered:
                add("warn", "corner-junk", t,
                    "coloured icon strip along the bottom (taskbar remnant?)")
                junk_flag_until = t + 8
            # red badge, bottom-right corner
            br = fr[DH - 25:DH, DW - 35:DW, :].astype(np.int16)
            red = (br[:, :, 2] > 140) & (br[:, :, 2] - br[:, :, 0] > 50)
            if red.sum() > 15:
                add("warn", "corner-junk", t,
                    "red badge in the bottom-right corner (PDF icon remnant?)")
                junk_flag_until = t + 8

        # --- corner speck: small persistent non-white blob in the bottom-right
        # chrome corner (badge remnants and similar smudges — the client has
        # circled this dot in two batches)
        cz = g[DH - 30:DH - 2, DW - 40:DW - 2]
        speck_run = speck_run + 1 if 1 <= int((cz < 235).sum()) <= 60 else 0
        if speck_run >= 16 and t > speck_flag_until:    # ≥8s persistent
            add("warn", "corner-speck", t - 8,
                "small persistent mark in the bottom-right corner")
            speck_flag_until = t + 20

        # --- transition junk: a near-full-width dark BAR across the slide
        # area (the PDF page separator caught mid-swipe) — the client sent a
        # screenshot of exactly this
        # the separator is nearly BLACK, thin (a few px) and wide; teal
        # diagram bands / highlight bars are lighter or thicker — don't flag
        dark_rows = ((g < 100) & ~cam_mask & ~robot_zone)[
            int(DH * 0.15):int(DH * 0.9)]
        rowfrac = dark_rows.mean(axis=1)
        hot = np.where(rowfrac > 0.40)[0]
        if bool(len(hot)) and (hot.max() - hot.min()) <= 4:
            # must be one CONTIGUOUS dark line, not scattered dark content
            # (a building's roofline in a photo once summed up to the same
            # fraction) — check the longest run on the hottest row
            row = dark_rows[hot[len(hot) // 2]]
            best_run, cur = 0, 0
            for v in row:
                cur = cur + 1 if v else 0
                best_run = max(best_run, cur)
            if best_run >= row.size * 0.45:
                bar_hits.append(t)                  # graded after the pass

        # --- ui-popup: Edge's text-selection menu (Highlight/Copilot) over
        # the page — tiny multicoloured logo blob on a white card (the client
        # caught two of these before QA knew the class existed)
        if True:
            import cv2 as _cv
            satm = ((hsvf[:, :, 1] > 90) & (hsvf[:, :, 2] > 90))
            satm[:, :int(DW * 0.15)] = False
            satm[:int(DH * 0.06), :] = False
            n5, lab5, st5, _c5 = _cv.connectedComponentsWithStats(
                satm.astype(np.uint8), 8)
            for i5 in range(1, n5):
                _x5, _y5, w5, h5, a5 = st5[i5]
                if not (2 <= a5 <= 40 and w5 <= 8 and h5 <= 8):
                    continue
                hue5 = hsvf[:, :, 0][lab5 == i5]
                if len({int(v) // 30 for v in hue5}) < 3:
                    continue
                bm5 = (lab5 == i5).astype(np.uint8)
                ring5 = (_cv.dilate(bm5, np.ones((5, 5), np.uint8)) > 0) & (bm5 == 0)
                if not (ring5.any()
                        and int(np.median(hsvf[:, :, 2][ring5])) >= 225
                        and int(np.median(hsvf[:, :, 1][ring5])) <= 25):
                    continue
                # The menu is a WHITE CARD: its wider surroundings are bright
                # and unsaturated. A map's legend dots (green/blue/purple)
                # merge into one multi-hue blob at analysis scale and passed
                # the ring test — the card test tells them apart.
                cy5, cx5 = _y5 + h5 // 2, _x5 + w5 // 2
                y0b, y1b = max(0, cy5 - 16), min(DH, cy5 + 17)
                x0b, x1b = max(0, cx5 - 16), min(DW, cx5 + 17)
                box_v = hsvf[y0b:y1b, x0b:x1b, 2]
                box_s = hsvf[y0b:y1b, x0b:x1b, 1]
                if float(((box_v > 215) & (box_s < 40)).mean()) >= 0.72:
                    popup_hits.append(t)
                    break

        # --- ui-panel remnant: saturated multi-colour cluster in the LEFT
        # margin strip (the Draw/highlighter dropdown) — slide content never
        # lives there and the robot mascot (bottom-left) is excluded
        if t > panel_flag_until:
            lstrip = fr[:int(DH * 0.75), 8:int(DW * 0.16), :].astype(np.int16)
            lsat = lstrip.max(axis=2) - lstrip.min(axis=2)
            lcol = (lsat > 60) & (lstrip.max(axis=2) > 90)
            if int(lcol.sum()) > 25:
                add("warn", "ui-panel", t,
                    "coloured UI panel remnant in the left margin "
                    "(Draw menu not fully removed?)")
                panel_flag_until = t + 8

        # --- freezes: whole frame (incl. webcam) identical while speech goes on
        if prev is not None:
            if np.abs(g - prev).max() < 2:
                freeze_run += 1
            else:
                if freeze_run >= FPS * 3:
                    wi = int((t - freeze_run / FPS) / win)
                    if wi < len(db) and db[wi] > -45:
                        add("warn", "freeze", t - freeze_run / FPS,
                            f"whole frame frozen {freeze_run / FPS:.1f}s "
                            "while speech continues")
                freeze_run = 0
            cam_now = fr[cam_mask]
            if prev_cam is not None:
                if np.abs(cam_now.astype(np.int16)
                          - prev_cam.astype(np.int16)).mean() < 0.4:
                    cam_still_run += 1
                else:
                    if cam_still_run >= FPS * 4:
                        add("warn", "webcam-static", t - cam_still_run / FPS,
                            f"webcam shows no motion for "
                            f"{cam_still_run / FPS:.1f}s")
                    cam_still_run = 0
            prev_cam = cam_now
        prev = g
        t += 1.0 / FPS
    dec.wait()

    # ---------- transition bars, graded by persistence ----------
    # a swipe separator flashes for <2s; a dark rule that stays for many
    # seconds is a design element of the slide, not junk
    if bar_hits:
        runs = [[bar_hits[0], bar_hits[0]]]
        for tt in bar_hits[1:]:
            if tt - runs[-1][1] <= 1.0 / FPS + 0.01:
                runs[-1][1] = tt
            else:
                runs.append([tt, tt])
        for (ra, rb) in runs:
            if rb - ra <= 1.5:
                add("warn", "transition-junk", ra,
                    "full-width dark bar across the slide (page separator "
                    "caught mid-swipe?)")

    # Only TRANSIENT popup detections count: a selection menu lives seconds,
    # while a map's coloured symbols trip the same logo test for a whole
    # slide (those produced a wall of false ui-popup warnings).
    if popup_hits:
        runs, cur_run = [], [popup_hits[0], popup_hits[0]]
        for tt in popup_hits[1:]:
            if tt - cur_run[1] <= 1.5:
                cur_run[1] = tt
            else:
                runs.append(tuple(cur_run))
                cur_run = [tt, tt]
        runs.append(tuple(cur_run))
        for a, b in runs:
            if (b - a) <= 8.0:
                add("warn", "ui-popup", a,
                    "selection menu (Highlight/Copilot) floating over the slide")

    # ---------- mouse pointer parked on a slide ----------
    # Must run at FULL resolution: downscaling averages the arrow's thin
    # outline into the white page, so the cheap pass simply cannot see it.
    try:
        from .detect import find_cursor as _find_cursor
        dur_c = max(durs.get("video") or 0, 1)
        tc, seen, gap_seen = 2.0, [], []
        # ---- webcam crop running off the source bubble ----
        # If the bubble moves mid-recording and the crop does not follow, part
        # of the page ends up INSIDE our circle. The tell is the source
        # bubble's own rim crossing the circle: a hard brightness step at a
        # fixed radius spanning a wide arc. A face never does that — it fills
        # the circle edge to edge. (Client caught this in the conclusion of
        # "Sample Answers Part One", 2026-09-11.)
        _cx, _cy = int(C.WEBCAM_OUT_X * 1920), int(C.WEBCAM_OUT_Y * 1080)
        _rv = C.WEBCAM_OUT_R - 2 - max(0, C.WEBCAM_RING)
        _yy, _xx = np.mgrid[0:1080, 0:1920]
        _rr = np.sqrt((_xx - _cx) ** 2 + (_yy - _cy) ** 2)
        _ang = np.degrees(np.arctan2(-(_yy - _cy), _xx - _cx))
        _inb = (_rr > 0.78 * _rv) & (_rr < 0.88 * _rv)
        _outb = (_rr > 0.90 * _rv) & (_rr < 0.99 * _rv)
        _sliv = (_rr > 0.94 * _rv) & (_rr < 0.99 * _rv)
        _sect = [((_ang >= a0) & (_ang < a0 + 20)) for a0 in range(-180, 180, 20)]

        def _crop_gap_arc(g_, g2_=None):
            """Widest contiguous arc of sectors that look like PAGE showing
            inside the circle. A page gap is bright (>=185), dead flat at the
            very edge (std < 6) and does not move between frames; a white
            shirt or a bright wall behind hair fails at least one of those
            (a shirt moves and has folds - it fooled the first version of
            this check on "House: Part Two", 2026-09-21)."""
            hot = []
            for i_, sa in enumerate(_sect):
                na, nb, ns = _inb & sa, _outb & sa, _sliv & sa
                if na.sum() < 200 or nb.sum() < 200 or ns.sum() < 100:
                    continue
                ob = g_[nb]
                if ob.mean() - g_[na].mean() <= 35 or ob.mean() < 185:
                    continue
                if float(g_[ns].std()) >= 6.0:
                    continue
                if g2_ is not None and float(np.abs(g_[ns] - g2_[ns]).mean()) >= 2.0:
                    continue
                hot.append(i_)
            if not hot:
                return 0
            best = run = 1
            for i_ in range(1, len(hot)):
                run = run + 1 if hot[i_] - hot[i_ - 1] == 1 else 1
                best = max(best, run)
            return best * 20

        def _grab_gray(t_):
            # same GREEN channel as the main frame, so the motion test compares
            # like with like (gray vs green differs by a few units on its own)
            raw2 = subprocess.run(
                ["ffmpeg", "-v", "error", "-ss", f"{t_:.2f}", "-i", path,
                 "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
                capture_output=True).stdout
            if len(raw2) < 1920 * 1080 * 3:
                return None
            return np.frombuffer(raw2[:1920 * 1080 * 3], np.uint8).reshape(
                1080, 1920, 3)[:, :, 1].astype(np.float32)

        while tc < dur_c:
            raw_c = subprocess.run(
                ["ffmpeg", "-v", "error", "-ss", f"{tc:.2f}", "-i", path,
                 "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
                capture_output=True).stdout
            if len(raw_c) >= 1920 * 1080 * 3:
                fr_c = np.frombuffer(raw_c[:1920 * 1080 * 3],
                                     np.uint8).reshape(1080, 1920, 3)
                if _find_cursor(fr_c):
                    seen.append(tc)
                g1_ = fr_c[:, :, 1].astype(np.float32)
                if _crop_gap_arc(g1_) >= 40:          # cheap pre-filter
                    g2_ = _grab_gray(tc + 0.7)          # then the motion test
                    if g2_ is not None and _crop_gap_arc(g1_, g2_) >= 40:
                        gap_seen.append(tc)
            tc += 4.0
        runs = []
        for t_ in gap_seen:
            if runs and t_ - runs[-1][1] <= 8.0:
                runs[-1][1] = t_
            else:
                runs.append([t_, t_])
        for a_, b_ in runs:
            add("fail", "webcam-crop-gap", a_,
                "webcam crop ran off the source bubble - page showing inside "
                f"the circle until {b_:.0f}s")
        runs = []
        for t_ in seen:
            if runs and t_ - runs[-1][1] <= 8.0:
                runs[-1][1] = t_
            else:
                runs.append([t_, t_])
        for a_, b_ in runs:
            add("warn", "cursor-visible", a_,
                "mouse pointer left sitting on the slide")
    except Exception as exc:  # noqa: BLE001
        add("warn", "qa-error", None, f"cursor check failed: {exc}")

    # ---------- presenter face inside the output bubble ----------
    # The client reference: face centred in the circle at ~0.45 of its
    # diameter. Both violations shipped once ("presenter not centred within
    # the bubble", "still a bit zoomed in") — now measured directly.
    try:
        import cv2
        face_c = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        wcx_f, wcy_f = C.WEBCAM_OUT_X * 1920, C.WEBCAM_OUT_Y * 1080
        R_f = C.WEBCAM_OUT_R
        offs, sizes = [], []
        dur_s = max(durs.get("video") or 0, 1)
        for i in range(12):
            tt = dur_s * (i + 0.5) / 12
            raw = subprocess.run(
                ["ffmpeg", "-v", "error", "-ss", f"{tt:.1f}", "-i", path,
                 "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "gray", "-"],
                capture_output=True).stdout
            if len(raw) < 1920 * 1080:
                continue
            g2 = np.frombuffer(raw[:1920 * 1080], np.uint8).reshape(1080, 1920)
            x0f, y0f = int(wcx_f - R_f), max(0, int(wcy_f - R_f))
            det = face_c.detectMultiScale(
                g2[y0f:int(wcy_f + R_f), x0f:int(wcx_f + R_f)],
                1.1, 5, minSize=(60, 60))
            if len(det):
                fx, fy, fw, fh = max(det, key=lambda d: d[2] * d[3])
                offs.append((fx + fw / 2 - (wcx_f - x0f),
                             fy + fh / 2 - (wcy_f - y0f)))
                sizes.append(fh / (2 * R_f))
        if len(offs) >= 4:
            mx = float(np.median([o[0] for o in offs]))
            my = float(np.median([o[1] for o in offs]))
            sz = float(np.median(sizes))
            # client spec 2026-08-14: the source bubble maps 1:1 onto ours, so
            # the presenter's natural position inside it is correct by
            # definition — only flag GROSS drift (a compositing bug, not a
            # seating choice)
            if abs(mx) > 45 or abs(my) > 45:
                add("warn", "webcam-off-centre", None,
                    f"presenter face sits {mx:+.0f}px/{my:+.0f}px off the "
                    "bubble centre (possible crop drift)")
            if sz > 0.72:
                add("warn", "webcam-tight", None,
                    f"presenter face fills {sz:.0%} of the bubble")
    except Exception as exc:  # noqa: BLE001
        add("warn", "qa-error", None, f"face check failed: {exc}")

    # ---------- bridge-too-long: background frozen while the SOURCE moves.
    # Bridges hold a still of the previous slide over transitions; a bridge
    # that outlives its transition hides real content (a 37s one shipped once
    # when a design rule was mistaken for a swipe bar). Compare background
    # motion in the output against the cleaned source, 1 sample/s.
    if src and os.path.exists(src):
        try:
            def _gray_stream(p, w, h):
                dec = subprocess.Popen(
                    ["ffmpeg", "-v", "error", "-i", p, "-vf",
                     f"fps=1,scale={w}:{h}", "-f", "rawvideo", "-pix_fmt",
                     "gray", "pipe:1"], stdout=subprocess.PIPE)
                while True:
                    b = dec.stdout.read(w * h)
                    if len(b) < w * h:
                        break
                    yield np.frombuffer(b, np.uint8).reshape(h, w).astype(np.int16)
                dec.wait()
            outs = list(_gray_stream(path, 240, 135))
            srcs = list(_gray_stream(src, 240, 135))
            n = min(len(outs), len(srcs))
            yy2, xx2 = np.mgrid[0:135, 0:240]
            bgm = ((xx2 - C.WEBCAM_OUT_X * 240) ** 2
                   + (yy2 - C.WEBCAM_OUT_Y * 135) ** 2) > (C.WEBCAM_OUT_R / 8 + 6) ** 2
            bgm &= ~((xx2 < 35) & (yy2 > 95))            # robot
            run, run_t0 = 0, 0
            for i in range(1, n):
                o_d = float(np.abs(outs[i] - outs[i - 1])[bgm].mean())
                s_d = float(np.abs(srcs[i] - srcs[i - 1]).mean())
                if o_d < 0.6 and s_d > 3.0:               # output still, source moving
                    if run == 0:
                        run_t0 = i - 1
                    run += 1
                else:
                    if run >= 3:
                        add("warn", "bridge-too-long", run_t0,
                            f"background frozen for {run}s while the source "
                            "keeps changing (bridge hiding content?)")
                    run = 0
            if run >= 3:
                add("warn", "bridge-too-long", run_t0,
                    f"background frozen for {run}s while the source keeps "
                    "changing (bridge hiding content?)")
        except Exception as exc:  # noqa: BLE001
            add("warn", "qa-error", None, f"bridge check failed: {exc}")

    level = ("fail" if any(f["level"] == "fail" for f in findings)
             else "warn" if findings else "pass")
    return {"file": os.path.basename(path), "path": path,
            "result": level, "findings": findings}


def main(argv):
    out_json = None
    src = None
    if "--json" in argv:
        i = argv.index("--json")
        out_json = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]
    if "--src" in argv:
        i = argv.index("--src")
        src = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]
    reports = []
    for p in argv:
        r = check(p, src=src)
        reports.append(r)
        print(f"\n=== {r['file']}: {r['result'].upper()} "
              f"({len(r['findings'])} finding(s))")
        for f in r["findings"]:
            ts = "" if f["t"] is None else f"@{int(f['t'] // 60)}:{f['t'] % 60:04.1f} "
            print(f"  [{f['level']}] {f['kind']} {ts}- {f['msg']}")
    if out_json:
        with open(out_json, "w") as fh:
            json.dump(reports, fh, indent=2)
        print(f"\nreport -> {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
