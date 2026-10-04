"""
Computer-vision detectors used by the pipeline.

  * detect_webcam_bubble  -> locate the circular Loom webcam in the source frame
  * grey_to_white_mask    -> boolean mask of Word/PDF grey background pixels
  * remove_grey           -> repaint grey pixels white (in place)
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass

import cv2
import numpy as np

from . import config as C


@dataclass
class Bubble:
    cx: int          # centre x in source pixels
    cy: int          # centre y in source pixels
    r: int           # radius in source pixels
    detected: bool   # False if we fell back to the preset default
    kind: str = "circle"   # "panel" = bubble sits in a uniform side column;
    #                        composite may then reframe the face (shrink view)


def _sample_frames(video: str, n: int = 5) -> list[np.ndarray]:
    """Grab `n` frames spread across the clip via ffmpeg, decoded with OpenCV."""
    cap = cv2.VideoCapture(video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    frames = []
    for i in range(n):
        pos = int(total * (i + 1) / (n + 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ok, fr = cap.read()
        if ok:
            frames.append(fr)
    cap.release()
    return frames


def track_bubble(video: str, bubble: "Bubble", *, fps: float = 2.0,
                 min_shift: int = 6, min_len: float = 0.8) -> list:
    """Follow the webcam bubble over time and report where it JUMPS.

    detect_webcam_bubble samples a handful of frames and takes their median, so
    a bubble that moves mid-recording gets one averaged position and the crop
    is then wrong for part of the video: it takes page pixels from outside the
    bubble, which show as a pale crescent inside the output circle with the
    face pushed off-centre. Seen 2026-09-11 on "Sample Answers Part One" — the
    bubble dropped 15px at 570s, exactly the conclusion the client flagged.

    Measurement: the bubble is the only large DARK disc in its neighbourhood
    (wall + face sit well under 210, while the page around it is 230-250), so
    the largest such blob, checked for disc-like fill, gives its centre. The
    baseline is the MEDIAN position over the whole video rather than `bubble`,
    so a steady small offset is left alone and only real jumps are corrected.

    Returns [(t0, t1, dx, dy), ...]: how far the crop must move during each
    span. Empty list = it never moved, and the caller keeps the fixed crop.
    """
    cx, cy, r = int(bubble.cx), int(bubble.cy), int(bubble.r)
    cap = cv2.VideoCapture(video)
    sw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1620
    sh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
    cap.release()
    pad = 70
    ww, hh = min(sw, 2 * (r + pad)), min(sh, 2 * (r + pad))
    # ffmpeg CLAMPS a crop window that runs past the frame; mirror that here or
    # every measured centre is silently offset by the overhang.
    x0 = min(max(0, cx - r - pad), sw - ww)
    y0 = min(max(0, cy - r - pad), sh - hh)
    try:
        raw = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", video, "-r", f"{fps}",
             "-vf", f"crop={ww}:{hh}:{x0}:{y0},format=gray",
             "-f", "rawvideo", "-pix_fmt", "gray", "-"],
            capture_output=True, timeout=1800).stdout
    except Exception:  # noqa: BLE001
        return []
    n = len(raw) // (ww * hh)
    if n < 8:
        return []
    vid = np.frombuffer(raw[:n * ww * hh], np.uint8).reshape(n, hh, ww)

    ker = np.ones((7, 7), np.uint8)

    def _blob(gray_win):
        """centre of the bubble disc in the window, or None"""
        m = cv2.morphologyEx((gray_win < 210).astype(np.uint8),
                             cv2.MORPH_OPEN, ker)
        cnt, _lab, st, _cen = cv2.connectedComponentsWithStats(m, 8)
        if cnt < 2:
            return None
        k = 1 + int(np.argmax(st[1:, 4]))
        x, y, w, h, a = st[k]
        # A disc fills pi/4 of its box and is as wide as it is tall. These
        # limits must be TIGHT: the bubble is measured to a couple of pixels,
        # so a loose test lets through a frame where something dark on the
        # slide touched the bubble and stretched the box — the centre then
        # jumps by several px and the crop follows it, which is exactly the
        # up-and-down wobble the client saw ("Max Price", 2026-09-28: a real
        # disc measured |w-h| = 1 while the bad frames measured 11-27).
        if (abs(int(w) - int(h)) > max(3.0, r * 0.08)
                or not (0.70 < a / max(1, w * h) < 0.86)
                or abs((w + h) / 4 - r) > r * 0.10):
            return None
        return x + w / 2.0, y + h / 2.0

    def _at(t):
        """measure one frame at time t (used to pin phase boundaries)"""
        try:
            raw1 = subprocess.run(
                ["ffmpeg", "-v", "error", "-ss", f"{t:.2f}", "-i", video,
                 "-frames:v", "1", "-vf",
                 f"crop={ww}:{hh}:{x0}:{y0},format=gray",
                 "-f", "rawvideo", "-pix_fmt", "gray", "-"],
                capture_output=True, timeout=60).stdout
        except Exception:  # noqa: BLE001
            return None
        if len(raw1) < ww * hh:
            return None
        return _blob(np.frombuffer(raw1[:ww * hh], np.uint8).reshape(hh, ww))

    cxs: list = []
    cys: list = []
    for i in range(n):
        b_ = _blob(vid[i])
        cxs.append(b_[0] if b_ else None)
        cys.append(b_[1] if b_ else None)

    good = [i for i in range(n) if cxs[i] is not None]
    if len(good) < n * 0.5:
        return []                                # too unreliable to act on
    bx = float(np.median([cxs[i] for i in good]))
    by = float(np.median([cys[i] for i in good]))

    runs: list = []
    for i in range(n):
        dx = int(round((cxs[i] - bx) / 2.0) * 2) if cxs[i] is not None else None
        dy = int(round((cys[i] - by) / 2.0) * 2) if cys[i] is not None else None
        if dx is None:                           # carry the last known position
            dx, dy = (runs[-1][2], runs[-1][3]) if runs else (0, 0)
        if runs and runs[-1][2] == dx and runs[-1][3] == dy:
            runs[-1][1] = (i + 1) / fps
        else:
            runs.append([i / fps, (i + 1) / fps, dx, dy])

    out = []
    for a, b, dx, dy in runs:
        if b - a < min_len or (abs(dx) < min_shift and abs(dy) < min_shift):
            continue
        if out and abs(out[-1][1] - a) < 0.01 and out[-1][2:] == (dx, dy):
            out[-1] = [out[-1][0], b, dx, dy]
        else:
            out.append([round(a, 2), round(b, 2), dx, dy])

    # A genuine jump is CONSISTENT: every sample inside the span reads the new
    # position. Measurement noise is not — it wanders. Drop any phase whose own
    # samples do not agree with it, so one bad frame can never move the crop.
    kept = []
    for a, b, dx, dy in out:
        i0, i1 = int(a * fps), max(int(a * fps) + 1, int(b * fps))
        agree = tot = 0
        for i in range(i0, min(i1, n)):
            if cxs[i] is None:
                continue
            tot += 1
            if (abs((cxs[i] - bx) - dx) <= 2.5 and abs((cys[i] - by) - dy) <= 2.5):
                agree += 1
        if tot >= 2 and agree >= 0.8 * tot:
            kept.append([a, b, dx, dy])
    out = kept

    # The scan grid is coarse (2 fps), so a phase can start up to half a second
    # after the bubble actually jumped — and that half second renders with the
    # OLD crop, i.e. the very artefact we are fixing is still visible at the
    # start of the shot. Pin both edges to ~0.05s by stepping single frames.
    dur = n / fps

    def _matches(t, dx, dy):
        m_ = _at(t)
        if m_ is None:
            return None
        return (abs(round((m_[0] - bx) / 2) * 2 - dx) <= 2
                and abs(round((m_[1] - by) / 2) * 2 - dy) <= 2)

    for idx, ph in enumerate(out):
        a, b, dx, dy = ph
        # never let refinement grow a phase INTO its neighbour: overlapping
        # spans make the per-frame crop expression flip between two offsets
        lo = out[idx - 1][1] if idx else 0.0
        hi = out[idx + 1][0] if idx + 1 < len(out) else dur
        t_ = max(lo, a - 1.0)
        while t_ < a + 0.05:                     # walk forward to the first hit
            if _matches(t_, dx, dy):
                ph[0] = round(t_, 2)
                break
            t_ += 0.05
        t_ = min(hi, b + 1.0)
        while t_ > b - 0.05:                     # walk back to the last hit
            if _matches(t_, dx, dy):
                ph[1] = round(min(hi, t_ + 0.05), 2)
                break
            t_ -= 0.05
    return [tuple(ph) for ph in out]


def detect_webcam_bubble(video: str) -> Bubble:
    """Find the webcam circle (top-right quadrant).

    Hough alone is not enough: slide graphics are full of circles (clouds,
    diagram nodes, icons) and picking the right-most match once landed the
    "bubble" in the middle of a chemistry slide — the render then inpainted
    slide text and cropped a phantom webcam. The reliable tell is MOTION: the
    webcam interior (a live face) changes from frame to frame, slide graphics
    do not. So collect ALL plausible circles across sampled frames, cluster
    them by position, and keep the cluster whose interior actually moves."""
    frames = _sample_frames(video, 7)
    if not frames:
        fx, fy, fr = C.WEBCAM_FALLBACK
        return Bubble(int(fx * 1620), int(fy * 1080), int(fr * 1620),
                      detected=False)
    h, w = frames[0].shape[:2]

    cands: list[tuple[int, int, int]] = []
    for img in frames:
        roi = img[0:int(h * 0.5), int(w * 0.55):w]
        gray = cv2.medianBlur(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), 5)
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=120,
            param1=100, param2=40,
            minRadius=int(w * 0.04), maxRadius=int(w * 0.11),
        )
        if circles is None:
            continue
        for c in np.round(circles[0]).astype(int):
            cands.append((c[0] + int(w * 0.55), c[1], c[2]))

    if not cands:
        fx, fy, fr = C.WEBCAM_FALLBACK
        return Bubble(int(fx * w), int(fy * h), int(fr * w), detected=False)

    # cluster by centre proximity
    clusters: list[list[tuple[int, int, int]]] = []
    for c in cands:
        for cl in clusters:
            m = np.median(np.array(cl), axis=0)
            if abs(c[0] - m[0]) < 60 and abs(c[1] - m[1]) < 60:
                cl.append(c)
                break
        else:
            clusters.append([c])

    # motion must be measured across a SHORT gap (~0.2s): between far-apart
    # samples the slide itself changes and every slide circle "moves" too
    cap = cv2.VideoCapture(video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    gap = max(2, int(fps * 0.2))
    pairs = []
    for i in range(5):
        pos = int(total * (i + 1) / 6)
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ok1, f1 = cap.read()
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos + gap)
        ok2, f2 = cap.read()
        if ok1 and ok2:
            pairs.append((
                cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY).astype(np.int16),
                cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY).astype(np.int16)))
    cap.release()

    def motion(cx: int, cy: int, r: int) -> float:
        mask = np.zeros((h, w), np.uint8)
        cv2.circle(mask, (cx, cy), max(10, int(r * 0.7)), 1, -1)
        mb = mask.astype(bool)
        if not mb.any() or not pairs:
            return 0.0
        diffs = [float(np.abs(a - b)[mb].mean()) for a, b in pairs]
        return float(np.median(diffs))

    best, best_motion = None, -1.0
    for cl in clusters:
        arr = np.array(cl)
        cx, cy, r = (int(np.median(arr[:, 0])), int(np.median(arr[:, 1])),
                     int(np.median(arr[:, 2])))
        mv = motion(cx, cy, r)
        if mv > best_motion:
            best, best_motion = (cx, cy, r), mv

    # a live face moves; slide graphics sit still (typical noise level < 1)
    if best is None or best_motion < 1.5:
        fx, fy, fr = C.WEBCAM_FALLBACK
        return Bubble(int(fx * w), int(fy * h), int(fr * w), detected=False)

    # Refine the radius: Hough under-reads soft bubble rims (74 vs a real
    # ~105 once ballooned the cropped face). The bubble interior is TEXTURED
    # (face) while the page around it is uniform — walk rings outwards and
    # keep the last radius whose ring still shows texture.
    cx, cy, r0 = best
    g0 = pairs[0][0] if pairs else cv2.cvtColor(
        frames[0], cv2.COLOR_BGR2GRAY).astype(np.int16)
    cx0, cy0, _r0 = best
    g_still = pairs[0][0] if pairs else cv2.cvtColor(
        frames[0], cv2.COLOR_BGR2GRAY).astype(np.int16)
    # RECTANGULAR side-panel camera (some teachers record a full-height camera
    # column at the right edge instead of a round bubble). Signature: dark
    # columns spanning nearly the full height at the right edge, with LIGHT
    # page columns immediately left of them (a dark Acrobat theme is dark on
    # both sides and is skipped). Crop window = the column's full width
    # centred on the face — max pixels, full head in frame (client reference).
    dark_frac = (np.asarray(g_still) < 205).mean(axis=0)
    zone = int(w * 0.7)
    panel_cols = np.where(dark_frac[zone:] > 0.75)[0] + zone
    if len(panel_cols) > 120 and panel_cols.max() >= w - 8:
        x0p = int(panel_cols.min())
        left_probe = dark_frac[max(0, x0p - 60):max(1, x0p - 15)]
        if len(left_probe) and float(left_probe.mean()) < 0.3:
            r_p = (w - x0p) // 2 - 2
            if r_p > 60:
                # The camera column CONTAINS a round bubble (teal circle on a
                # uniform grey panel). Locate it exactly: panel background =
                # median colour of the column's bottom (empty) half; the
                # bubble is the one large blob that differs from it. Cropping
                # the column centre instead drifted the face off-centre.
                # The bubble's interior wall is nearly the SAME grey as the
                # panel (Δ≈5 levels) — colour thresholds can't find it, but a
                # contrast STRETCH around the panel level makes the rim crisp
                # enough for a bounded Hough. Centre must sit near the face.
                colg = np.asarray(g_still)[:, x0p:w].astype(np.float64)
                bg_v = float(np.median(colg[int(h * 0.65):]))
                st_img = np.clip((colg - (bg_v - 8)) * (255.0 / 60.0),
                                 0, 255).astype(np.uint8)
                st_img = cv2.medianBlur(st_img, 5)
                colw = w - x0p
                circ = cv2.HoughCircles(
                    st_img, cv2.HOUGH_GRADIENT, dp=1.2, minDist=300,
                    param1=80, param2=30,
                    minRadius=int(colw * 0.35), maxRadius=int(colw * 0.52))
                if circ is not None:
                    cands2 = sorted(
                        np.round(circ[0]).astype(int),
                        key=lambda c: abs(c[0] + x0p - cx0) + abs(c[1] - cy0))
                    bx2, by2, br2 = cands2[0]
                    return Bubble(x0p + int(bx2), int(by2),
                                  min(int(br2), r_p), detected=True,
                                  kind="panel")
                cy_p = max(r_p + 2, cy0 - int(r_p * 0.14))
                return Bubble((x0p + w) // 2, cy_p, r_p, detected=True,
                              kind="panel")

    # The live-noise boundary IS the rim: inside the bubble even a static
    # white wall flickers (sensor noise + codec), the page outside is a
    # decoded still — ring motion drops to ~0 exactly past the bubble edge.
    ang = np.linspace(0, 2 * np.pi, 90, endpoint=False)
    ca, sa = np.cos(ang), np.sin(ang)
    r_ref = r0
    for rr in range(max(20, int(r0 * 0.8)), int(w * 0.13), 2):
        xs2 = np.clip((cx + rr * ca).astype(int), 0, w - 1)
        ys2 = np.clip((cy + rr * sa).astype(int), 0, h - 1)
        mv = float(np.median([float(np.abs(a - b)[ys2, xs2].mean())
                              for a, b in pairs])) if pairs else 0.0
        if mv > 0.5:
            r_ref = rr + 4
    return Bubble(cx, cy, max(r0, r_ref), detected=True)


def remove_background(bgr: np.ndarray) -> np.ndarray:
    """Whiten the viewer 'canvas' around the document page, in place.

    Works for both viewer themes: the light Edge/PDF grey (~229) AND the dark
    Acrobat theme (~58). Strategy: build a mask of neutral *background-coloured*
    pixels (low saturation, in the light-grey OR dark band — excluding the white
    page and near-black text), then flood from the frame border and whiten only
    the connected region. The white page stops the flood, so interior text and
    diagrams are preserved even when they are dark/neutral.

    This also removes the top toolbar and any UI dropdowns (e.g. the dark 'Draw'
    panel) because they are neutral chrome connected to the frame edge.
    """
    h, w = bgr.shape[:2]
    b = bgr[:, :, 0].astype(np.int16)
    g = bgr[:, :, 1].astype(np.int16)
    r = bgr[:, :, 2].astype(np.int16)
    mx = np.maximum(np.maximum(b, g), r)
    mn = np.minimum(np.minimum(b, g), r)
    sat = mx - mn
    neutral = sat <= C.BG_MAX_SAT
    light = (mx >= C.BG_GREY_LO) & (mx <= C.BG_GREY_HI)
    dark = (mx >= C.BG_DARK_LO) & (mx <= C.BG_DARK_HI)
    mask = (neutral & (light | dark)).astype(np.uint8)

    # bridge thin gaps (icons/handles inside the chrome) so the canvas stays one
    # connected region reaching the border
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return bgr

    # Seed the flood from the LEFT/RIGHT/TOP edges only — never the bottom. The
    # viewer canvas wraps the page and is reached from the sides anyway, but slide
    # content is often scrolled so it runs off the BOTTOM of the frame; seeding
    # there made such content (e.g. a shaded table) "border-connected chrome" and
    # wiped the whole thing (client: "the table is not loading").
    border = np.concatenate([
        labels[0, :], labels[:, 0], labels[:, w - 1]
    ])
    border_labels = set(np.unique(border)) - {0}
    if not border_labels:
        return bgr

    keep = np.isin(labels, list(border_labels)).astype(np.uint8)
    # grow slightly so the thin page-border line that sits right at the
    # canvas/page boundary is consumed too (the page has a white margin, so a few
    # px of growth never reaches the document content)
    keep = cv2.dilate(keep, np.ones((7, 7), np.uint8))
    bgr[keep.astype(bool)] = (255, 255, 255)
    return bgr


def swatch_row(hsv: np.ndarray, scale: float = 1.0) -> bool:
    """True if the image region contains the Draw menu's giveaway: a ROW of
    ≥3 compact saturated blobs (the colour swatches) at a similar height.

    A simple "several saturated hues in the corner" rule false-fired on the
    teacher's own pen strokes in the page margin (a blue curve anti-aliases
    into 2 hue bins) and ERASED a drawing. A pen stroke is one elongated
    component — it can never look like a row of small circles.
    `scale` adapts the pixel thresholds when `hsv` is downsampled (1/4 => 0.25).
    """
    sat = ((hsv[:, :, 1] > 90) & (hsv[:, :, 2] > 90)).astype(np.uint8)
    if int(sat.sum()) < 120 * scale * scale:
        return False
    # The Draw menu is a WHITE card on the light page margin; a poster's icon
    # row sits on dark artwork. Requiring a light surround stops the detector
    # eating a poster's left column (it whitened x<332 of an infographic).
    nonsat = hsv[:, :, 2][sat == 0]
    if nonsat.size and int(np.median(nonsat)) < 160:
        return False
    _ring_check_needed = True
    n, _labels, stats, cents = cv2.connectedComponentsWithStats(sat, 8)
    blobs = []
    a_lo, a_hi = 60 * scale * scale, 2500 * scale * scale
    d_max = 60 * scale
    for i in range(1, n):
        bx, by, bw, bh, area = stats[i]
        if a_lo <= area <= a_hi and max(bw, bh) <= d_max \
                and 0.4 <= bw / max(1.0, bh) <= 2.5:
            blobs.append(float(cents[i][1]))            # y-centre
    if len(blobs) < 3:
        return False
    blobs.sort()
    band = 30 * scale                                   # same-row tolerance
    if not any(blobs[i + 2] - blobs[i] < band for i in range(len(blobs) - 2)):
        return False
    # LOCAL card test: real menu swatches sit on a WHITE card — the ring
    # right around each blob is bright and unsaturated. A poster's flag /
    # coloured banners sit on artwork, so their rings are dark or colourful.
    # (Corner-wide medians failed here: the page margin next to a poster is
    # white enough to fool any global brightness gate.)
    blob_mask = np.zeros(sat.shape, np.uint8)
    n3, lab3, st3, _c3 = cv2.connectedComponentsWithStats(sat, 8)
    a_lo2, a_hi2 = 60 * scale * scale, 2500 * scale * scale
    d_max2 = 60 * scale
    ok_rings = 0
    ker = np.ones((max(3, int(9 * scale)) | 1,) * 2, np.uint8)
    for i in range(1, n3):
        bx3, by3, bw3, bh3, area3 = st3[i]
        if not (a_lo2 <= area3 <= a_hi2 and max(bw3, bh3) <= d_max2
                and 0.4 <= bw3 / max(1.0, bh3) <= 2.5):
            continue
        bm = (lab3 == i).astype(np.uint8)
        ring = (cv2.dilate(bm, ker) > 0) & (bm == 0)
        if not ring.any():
            continue
        v_ring = hsv[:, :, 2][ring]
        s_ring = hsv[:, :, 1][ring]
        if int(np.median(v_ring)) >= 210 and int(np.median(s_ring)) <= 40:
            ok_rings += 1
    return ok_rings >= 3


def remove_ui_panel(bgr: np.ndarray) -> bool:
    """Whiten the light Draw/highlighter dropdown menu (colour swatches, stroke
    preview, thickness slider, toggle) that opens at the top-left of the viewer.

    Detected by its giveaway: a cluster of *multiple distinct hues* (the colour
    swatches) in the top-left corner — the single-colour slide title can't fake
    it. The whitened block extends right only until the gap before the title, so
    the title is never clipped. Call AFTER remove_background. Returns True if a
    menu was found and painted out.
    """
    h, w = bgr.shape[:2]
    # Search the whole top-HALF of the left margin strip: when the page is
    # scrolled, the menu hangs lower and its swatch row can be cut off by the
    # frame top (seen in production: only the bottom half of two swatches was
    # visible, so a ≥3-hues rule missed it and the whole panel stayed).
    cx, cy = int(w * 0.20), int(h * 0.45)   # panel reaches ~x306 at 1620w;
    # content starts ~x364, so a 0.20 strip (x<332) still can't touch it
    corner = bgr[:cy, :cx]
    hsv = cv2.cvtColor(corner, cv2.COLOR_BGR2HSV)
    if not swatch_row(hsv):
        return False

    # Whiten the panel's FULL vertical extent, not a fixed corner box (the menu
    # is taller than any fixed guess: swatches + stroke preview + thickness
    # slider + "Text only highlight" toggle). Walk down the left strip and stop
    # after a run of clean rows; the strip ends well left of slide content
    # (content starts ~x364 at 1620w), so text/annotations are never touched.
    strip_w = cx + 8
    g = cv2.cvtColor(bgr[:, :strip_w], cv2.COLOR_BGR2GRAY)
    s2 = cv2.cvtColor(bgr[:, :strip_w], cv2.COLOR_BGR2HSV)[:, :, 1]
    busy = ((g < 235) | (s2 > 30)).sum(axis=1)
    # The panel is the ONLY thing living in this strip (slide content starts
    # ~x364 at 1620w), so just whiten down to the strip's last busy row. No
    # top-down walk: remove_background runs first and whitens the toolbar, so
    # a walk from y=0 hit 'all quiet' rows and bailed before reaching the
    # panel body (which starts ~y55) — leaving the menu on screen.
    rows = np.where(busy[:int(h * 0.7)] > 2)[0]
    if not len(rows) or int(rows.min()) > 130:
        # the Draw menu HANGS FROM THE TOOLBAR — its card starts near the top.
        # Content that begins lower (a poster sliding in mid-transition once
        # matched the swatch test for a single frame, and static-skip froze
        # that eaten frame for the whole slide) is never the menu.
        return False
    last = int(rows.max())
    bgr[:min(h, last + 10), :strip_w] = (255, 255, 255)
    return True


def remove_top_toolbar(bgr: np.ndarray, h_strip: int = 52) -> bool:
    """Whiten the top viewer toolbar strip. The Acrobat/Edge toolbar sits in the top
    ~48px; slide content (titles) starts well below (~y130+), and in full-screen the
    strip is just the page's white top margin — so whitening it is always safe. This
    lets the composite CENTRE content vertically without the toolbar peeking in at
    reduced zoom. Returns True."""
    bgr[0:h_strip] = (255, 255, 255)
    return True


def remove_pdf_badge(bgr: np.ndarray) -> bool:
    """Whiten the Adobe Acrobat "PDF" badge that Microsoft Edge's PDF viewer floats
    at the bottom-right of the page. It is a red/pink rounded icon that sits there
    on every slide of the recording and survives remove_background (it is a
    saturated colour, not neutral grey), so it ends up in the bottom-right of the
    output. Detected as a red blob confined to the bottom-right corner (where slide
    content never is) and painted out. Returns True if it was removed."""
    h, w = bgr.shape[:2]
    y0z, x0z = h - 90, w - 130                       # bottom-right viewer-chrome zone
    zone = bgr[y0z:h, x0z:w]
    hsv = cv2.cvtColor(zone, cv2.COLOR_BGR2HSV)
    # Edge FADES the floating badge to semi-transparent after a moment of
    # mouse idle — the pale ghost ring failed a strict "red" test and survived
    # into deliveries as a lone mark on white (client flagged it). Thresholds
    # are loose (warm hue, barely saturated) — this zone is chrome-only, the
    # page never puts content in the bottom-right 130x90 corner.
    red = (((hsv[:, :, 0] < 25) | (hsv[:, :, 0] > 160))
           & (hsv[:, :, 1] > 18) & (hsv[:, :, 2] > 60))
    if int(red.sum()) < 25:
        return False
    ys, xs = np.where(red)
    # generous pad: the badge's anti-aliased edge and drop shadow reach well
    # past its red core — a 5px pad left a tiny grey dot that the client
    # circled in TWO batches running
    bgr[max(0, y0z + int(ys.min()) - 14):y0z + int(ys.max()) + 15,
        max(0, x0z + int(xs.min()) - 14):x0z + int(xs.max()) + 15] = (255, 255, 255)
    return True


def remove_taskbar(bgr: np.ndarray, detect_h: int = 48, wipe_h: int = 64) -> bool:
    """Whiten the Windows taskbar — the band of app icons at the very bottom that
    shows before the presenter goes full screen (its coloured icons survive
    remove_background, leaving them floating on white in the output).

    Call BEFORE remove_background, while the icons are still coloured. It is
    identified — and told apart from a slide image whose bottom edge reaches the
    frame bottom — by its unique layout: small coloured icons SPREAD from the far
    left to the far right of the strip, on a near-white background (low fill). A
    slide image is instead a dense, contiguous, centred colour block, so it is
    left untouched. Returns True if a taskbar was painted out."""
    h, w = bgr.shape[:2]
    strip = bgr[h - detect_h:h]
    hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)
    colored = (hsv[:, :, 1] > 60) & (hsv[:, :, 2] > 50) & (hsv[:, :, 2] < 250)
    total = int(colored.sum())
    if total < 1200:                                 # too little colour to be a taskbar
        return False
    bins = 24
    bw = w // bins
    cols = colored.sum(axis=0)
    occ = [b for b in range(bins) if cols[b * bw:(b + 1) * bw].sum() > 40]
    if len(occ) < 6:
        return False
    left_b, right_b = occ[0], occ[-1]
    fill = total / float(colored.size)               # coloured fraction of the strip
    spans_width = left_b <= 3 and right_b >= 17 and (right_b - left_b) >= 14
    if spans_width and fill < 0.35:                  # sparse icons edge-to-edge → taskbar
        bgr[h - wipe_h:h] = (255, 255, 255)
        return True

    # Windows 11 CENTRES its icons, so the edge-to-edge test misses it (client:
    # taskbar visible at 12:54). Second signature: a horizontal ROW of ≥5 small
    # multi-coloured blobs at the same height in the strip — app icons. A slide
    # graphic that low is a contiguous block, not a spaced row of tiny blobs.
    if fill < 0.35:
        n, _l, stats, cents = cv2.connectedComponentsWithStats(
            colored.astype(np.uint8), 8)
        blobs = []
        for i in range(1, n):
            _bx, _by, bw2, bh2, area = stats[i]
            if 30 <= area <= 1600 and max(bw2, bh2) <= 44:
                blobs.append((float(cents[i][0]), float(cents[i][1]), i))
        if len(blobs) >= 5:
            blobs.sort(key=lambda b: b[1])
            for i in range(len(blobs) - 4):
                row = blobs[i:i + 5]
                if row[-1][1] - row[0][1] < 14:      # same height
                    xs = sorted(b[0] for b in row)
                    if xs[-1] - xs[0] > 250:         # spaced across ≥250px
                        hsv2 = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)
                        hues = hsv2[:, :, 0][colored]
                        if len({int(x) for x in (hues // 20)}) >= 3:
                            bgr[h - wipe_h:h] = (255, 255, 255)
                            return True
                    break
    return False

def find_cursor(bgr) -> bool:
    """Detection-only twin of remove_cursor (used by QA so the check runs the
    exact same test the cleaner does, at full resolution)."""
    return remove_cursor(bgr.copy())


def remove_cursor(bgr, prev_bgr=None) -> bool:
    """Whiten the mouse pointer parked on the page (client request).

    The pointer is the only SMALL, ISOLATED dark shape on a clean page: text
    glyphs always have neighbours (they sit in words), pen ticks are drawn
    next to the line they mark. Requiring emptiness for 45px around the blob
    is what separates them — shape alone does not (a cursor and the letter
    "l" have identical fill ratios).
    """
    h, w = bgr.shape[:2]
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    # The pointer is a WHITE arrow with a thin GREY outline, not a dark blob:
    # at a "dark" threshold its outline breaks into fragments too small to
    # survive any size filter, which is why a 150 cut-off saw nothing at all.
    dark = (g < 215).astype(np.uint8)
    dark[:, :int(w * 0.08)] = 0                  # left chrome
    dark[:int(h * 0.05), :] = 0                  # toolbar strip
    n, lab, st, _c = cv2.connectedComponentsWithStats(dark, 8)
    hit = False
    for i in range(1, n):
        x, y, bw, bh, area = st[i]
        if not (25 <= area <= 400 and 6 <= bw <= 30 and 10 <= bh <= 40):
            continue
        if bh < bw * 1.25:                       # round -> a bullet point
            continue
        y0, y1 = max(0, y - 6), min(h, y + bh + 6)
        x0, x1 = max(0, x - 45), min(w, x + bw + 45)
        band = dark[y0:y1, x0:x1].copy()
        band[lab[y0:y1, x0:x1] == i] = 0
        if int(band.sum()) > 25:                 # has neighbours -> it is text
            continue
        halo = g[y0:y1, x0:x1]
        blob = (lab[y0:y1, x0:x1] == i)
        around = halo[~blob]
        if around.size and float((around > 235).mean()) < 0.90:
            continue                             # not on a clean page
        hsv_b = cv2.cvtColor(bgr[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
        if float(hsv_b[:, :, 1][blob].mean()) > 60:
            continue                             # coloured -> the teacher's pen
        bgr[max(0, y - 3):y + bh + 3, max(0, x - 3):x + bw + 3] = (255, 255, 255)
        hit = True
    return hit
