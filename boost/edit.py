"""
ffmpeg / OpenCV editing stages.

Pipeline (replaces the manual Premiere workflow described in the brief):

  1. clean_grey   - repaint Word/PDF grey background white (side margins AND the
                    scrolling page breaks) + normalise fps.  This single CV pass
                    replaces the manual "mask + keyframe every page break" step.
  2. composite    - background layer (scale 112, centred, toolbars cropped off),
                    white fill, enlarged webcam bubble (scale 165, circular),
                    robot mascot overlay, and Podcast-Voice-style audio cleanup.
  3. make_intro   - generated title card (stand-in for SC Intro.mogrt).
  4. assemble     - intro -> body -> outro with cross-fades, exported 1920x1080.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile

import cv2
import numpy as np

from . import config as C
from . import detect
from .detect import Bubble, remove_background
from .detect import swatch_row as _swatch_row

def _find_font() -> str:
    """A bold sans-serif font that exists, across Windows / macOS / Linux."""
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf",                               # Windows
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",         # macOS
        "/Library/Fonts/Arial Bold.ttf",                             # macOS (older)
        "/System/Library/Fonts/Helvetica.ttc",                       # macOS fallback
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",      # Linux
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return candidates[0]


# escape ':' (Windows drive letter) for ffmpeg drawtext's fontfile option
FONT = _find_font().replace("\\", "/").replace(":", "\\:")


def _run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"command failed ({proc.returncode}):\n{' '.join(cmd)}\n{proc.stderr[-2000:]}"
        )


def probe(video: str) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,r_frame_rate",
         "-show_entries", "format=duration", "-of", "json", video],
        capture_output=True, text=True,
    ).stdout
    data = json.loads(out)
    st = data["streams"][0]
    return {
        "w": int(st["width"]),
        "h": int(st["height"]),
        "duration": float(data["format"]["duration"]),
    }


# ---------------------------------------------------------------------------
# Stage 1 - grey background removal (OpenCV streamed through ffmpeg pipes)
# ---------------------------------------------------------------------------
def clean_grey(src: str, dst: str, fps: int = C.OUT_FPS,
               paint_bubble: "Bubble | None" = None,
               workers: "int | None" = None) -> None:
    """Background-removal stage, parallelised across CPU cores.

    The video is split into time-segments, each processed by its own
    `boost.segworker` process, then concatenated. Output is video-only (audio is
    taken from the original source later in `composite`, which keeps A/V in sync
    regardless of how the video segments are cut)."""
    info = probe(src)
    w, h, duration = info["w"], info["h"], info["duration"]

    # Erase the source Loom bubble by INPAINTING a tight circle around it (blends
    # into whatever's behind — white margin or table). The old white rectangle was
    # much bigger than the bubble and whited out a chunk of the slide around it,
    # which showed as a white "square" on coloured content next to the webcam.
    box = "-"
    if paint_bubble is not None:
        rr = int(paint_bubble.r * 1.12)     # cover bubble + its border, minimal excess
        box = f"{paint_bubble.cx},{paint_bubble.cy},{rr}"

    workers = workers or os.cpu_count() or 1
    workers = max(1, min(workers, int(duration // 8) or 1))   # >= ~8s per segment

    tmp = tempfile.mkdtemp(prefix="boost_seg_")
    seg_dur = duration / workers
    segs, procs = [], []
    for i in range(workers):
        start = i * seg_dur
        dur = "-" if i == workers - 1 else f"{seg_dur}"       # last decodes to EOF
        seg = os.path.join(tmp, f"seg{i:03d}.mp4")
        segs.append(seg)
        procs.append(subprocess.Popen(
            [sys.executable, "-m", "boost.segworker", src, f"{start}", dur,
             str(fps), str(w), str(h), box, seg],
            cwd=C.PROJECT_ROOT))
    for p in procs:
        if p.wait() != 0:
            raise RuntimeError("segworker failed")

    if workers == 1:
        os.replace(segs[0], dst)
        return

    listfile = os.path.join(tmp, "list.txt")
    with open(listfile, "w", encoding="utf-8") as f:
        for s in segs:
            f.write(f"file '{s.replace(chr(92), '/')}'\n")
    try:
        _run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0",
              "-i", listfile, "-c", "copy", dst])
    except RuntimeError:
        _run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0",
              "-i", listfile, *C.video_codec_args(), dst])


# ---------------------------------------------------------------------------
# Stage 2 - composite background + webcam + robot + audio
# ---------------------------------------------------------------------------
def _content_bounds_v(clean: str, sw: int, sh: int,
                      top_ignore: int = 52, left_ignore: int = 140,
                      N: int = 40) -> "tuple | None":
    """Sample frames of the grey-removed `clean` (ink on white) and return
    (content_top, content_bottom) in raw px: the HIGHEST slide content and the
    LOWEST slide content anywhere in the whole video. The composite fits this
    span into the frame so no slide (title at the top, image at the bottom) is
    ever cropped. Sampled densely with near-extreme percentiles so a rare
    tall/low slide is included but a one-frame speckle is not.
    """
    cap = cv2.VideoCapture(clean)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    tops, bottoms = [], []
    for i in range(N):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * (i + 0.5) / N))
        ok, fr = cap.read()
        if not ok:
            continue
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        ink = g < 225                       # non-white = document ink
        ink[:, :left_ignore] = False        # drop the left sidebar column
        ink[:, sw - 40:] = False            # and the right scrollbar column
        rowcount = ink.sum(axis=1)
        rows = np.where(rowcount > 15)[0]   # rows with real ink (not speckle)
        rows = rows[rows >= top_ignore]     # ignore the toolbar band
        if len(rows):
            tops.append(int(rows.min()))
            bottoms.append(int(rows.max()))
    cap.release()
    if not tops:
        return None
    # near-extremes: highest content, lowest content across slides (2%/98% guard
    # against a single stray frame while still catching a genuine extreme slide)
    return int(np.percentile(tops, 2)), int(np.percentile(bottoms, 98))


def _slide_scan(clean: str, sw: int, sh: int, step: float = 0.25,
                return_phases: bool = False):
    """One cheap decode pass over `clean` (downscaled grey @ 1/step fps).

    Returns a list of slides: dicts with t0/t1 (seconds), ct/cb (content top &
    bottom in raw source px) and `ink` — a downsampled boolean mask (1/DS scale)
    of every ink pixel seen on that slide (annotations accumulate over time, so
    the mask is OR-ed across the slide's frames). Used by composite() to find
    slides whose content collides with the fixed webcam circle or runs off the
    bottom at spec zoom — exactly the two recurring client complaints.
    """
    DS = 4
    dw, dh = sw // DS, sh // DS
    dec = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", clean, "-vf",
         f"fps={1 / step},scale={dw}:{dh}", "-f", "rawvideo",
         "-pix_fmt", "gray", "pipe:1"], stdout=subprocess.PIPE)
    nbytes = dw * dh
    slides, prev = [], None
    cur = {"t0": 0.0, "cnt": np.zeros((dh, dw), np.uint8), "n": 0, "tail": []}
    t = 0.0
    left_ig, top_ig = 140 // DS, 52 // DS

    def close(slide, t_end):
        # Un-count the LAST few samples: the end of a slide (and of the video)
        # is where fullscreen-exit reflows and transition blends live — the
        # recording tail showed reflowed page text as phantom "slide content"
        # and dragged whole slides into needless zoom-outs.
        for m in slide.pop("tail"):
            np.subtract(slide["cnt"], m, out=slide["cnt"], where=slide["cnt"] > 0)
        slide["t1"] = t_end
        slides.append(slide)
    while True:
        buf = dec.stdout.read(nbytes)
        if len(buf) < nbytes:
            break
        g = np.frombuffer(buf, np.uint8).reshape(dh, dw).astype(np.int16)
        if prev is not None:
            changed = int((np.abs(g - prev) > 40).sum())
            if changed > dw * dh * 0.04:            # hard cut = new slide
                close(cur, t)
                cur = {"t0": t, "cnt": np.zeros((dh, dw), np.uint8), "n": 0,
                       "tail": []}
        # Skip WINDOWED (non-fullscreen) frames entirely — when the teacher
        # drops out of fullscreen (usually to stop the recording) the browser
        # tab/address bars appear and the page text reflows to a different
        # position, which is NOT where the slide content lives. Signature: a
        # band of dark chrome pixels spanning nearly the full width of the
        # very top rows (a fullscreen slide has white/canvas there).
        topstrip = g[:max(1, 40 // DS)] < 128
        chrome = False
        if topstrip.mean() > 0.015:
            cols = np.where(topstrip.any(axis=0))[0]
            chrome = (len(cols) and cols.min() < dw * 0.15
                      and cols.max() > dw * 0.85)
        ink = g < 200
        ink[:, :left_ig] = False
        ink[:, dw - max(1, 40 // DS):] = False
        ink[:top_ig, :] = False
        # COUNT sightings instead of OR-ing: the mouse cursor sweeps the page
        # and would otherwise stamp phantom "content" everywhere it passes —
        # and slide-transition BLENDS stamp the previous slide's content onto
        # the new one. Skip the first sample after a cut (usually a blend) and
        # require a pixel to be inky in ≥3 samples (~0.75s) so only things
        # that STAY on the slide count (text, images, finished pen strokes).
        if cur["n"] > 0 and not chrome:
            m = ink.astype(np.uint8)
            np.add(cur["cnt"], m, out=cur["cnt"], where=cur["cnt"] < 255)
            cur["tail"].append(m)
            if len(cur["tail"]) > 3:
                cur["tail"].pop(0)
            cur["ink_seen"] = cur.get("ink_seen", 0) + int(m.sum())
            cur["ink_frames"] = cur.get("ink_frames", 0) + 1
            # SOFT mask for extent bookkeeping: pale imagery (a light-blue
            # sea map ~220 grey) is invisible to the strict ink threshold and
            # once got its bottom recentred off-frame. Anything darker than
            # the page counts towards the extents; the strict mask still
            # drives collision tests.
            soft = g < 238
            soft[:, :left_ig] = False
            soft[:, dw - max(1, 40 // DS):] = False
            soft[:top_ig, :] = False
            rws = np.where(soft.sum(axis=1) > 15 // DS)[0]
            if len(rws):
                cur.setdefault("tops", []).append(int(rws.min()) * DS)
                cur.setdefault("bots", []).append(int(rws.max()) * DS)
                # keep the extent as a TIME SERIES too: a slide the teacher
                # scrolls through has several stable phases, and centring the
                # whole window on their union leaves each phase looking
                # top-heavy (the client kept flagging exactly this).
                cur.setdefault("ext", []).append(
                    (t, int(rws.min()) * DS, int(rws.max()) * DS))
        cur["n"] += 1
        prev = g
        t += step
    dec.wait()
    close(cur, t)

    out = []
    phases = []                                     # dropped slivers (e.g. the
    # viewer's "page appears low, then snaps up" hold) — composite can place
    # a compensation window over them so the slide never visibly moves
    for s in slides:
        if s["n"] < 3:                              # blend/transition sliver
            if s["n"] >= 1:
                ink_s = s["cnt"] >= 1
                rows_s = np.where(ink_s.sum(axis=1) > 15 // DS)[0]
                if len(rows_s):
                    phases.append({"t0": s["t0"], "t1": s["t1"],
                                   "ct": int(rows_s.min()) * DS,
                                   "cb": int(rows_s.max()) * DS})
            continue
        ink = s["cnt"] >= min(3, max(1, s["n"] - 1))  # persistent ink only
        # Drop full-length straight lines: the PDF page BORDER runs the whole
        # height/width of the frame and would otherwise define the content
        # bounding box (real content never spans >50% as one 1px line).
        colf = ink.mean(axis=0)
        ink[:, colf > 0.5] = False
        rowf = ink.mean(axis=1)
        ink[rowf > 0.5, :] = False
        rows = np.where(ink.sum(axis=1) > 15 // DS)[0]
        if not len(rows):
            continue
        # a SCROLL phase masquerading as a slide: the content moves, so almost
        # nothing is "persistent" — the mask sees only whatever stood still
        # (e.g. a title) and geometry planned from it cropped the moving rest
        # off-frame. Persistent ink far below the per-frame visible ink marks
        # the window unstable; the planner then leaves it at base geometry.
        # Geometric test: a SCROLL moves the content top between samples; a
        # photo merely flickers around the ink threshold (which fooled the old
        # persistent-vs-visible ratio and left a poster slide un-rescued).
        # A scroll moves the content TOP or BOTTOM (a photo scrolling up under a
        # fixed title moves only the bottom) — either one drifting marks the
        # window unstable; a flickering photo edge moves neither by much.
        # UNION extent over every sample: geometry decisions (centring, fit)
        # must account for everything that was EVER visible in the window —
        # a photo scrolling in from below, a slide build. Planning from the
        # persistent mask alone once centred a lone title and pushed the
        # arriving photo off-frame; no stability heuristic could tell that
        # case from a flickering poster, the union can.
        tops = s.get("tops", [])
        bots = s.get("bots", [])
        ct_p, cb_p = int(rows.min()) * DS, int(rows.max()) * DS
        out.append({"t0": s["t0"], "t1": s["t1"], "ct": ct_p, "cb": cb_p,
                    "uct": min(tops + [ct_p]), "ucb": max(bots + [cb_p]),
                    "ext": s.get("ext", []),
                    "ink": ink, "ds": DS, "stable": True})
    if return_phases:
        return out, phases
    return out


def _bar_intervals(clean: str, sw: int, sh: int, step: float = 0.25):
    """Moments where a thin, dark, near-full-width BAR crosses the slide area
    — the PDF page separator caught mid-swipe. Catches swipes even between
    near-identical slides that the slide scan can't tell apart."""
    DS = 4
    dw, dh = sw // DS, sh // DS
    dec = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", clean, "-vf",
         f"fps={1 / step},scale={dw}:{dh}", "-f", "rawvideo",
         "-pix_fmt", "gray", "pipe:1"], stdout=subprocess.PIPE)
    nb = dw * dh
    hits, t = [], 0.0
    y0, y1 = int(dh * 0.1), int(dh * 0.95)
    while True:
        buf = dec.stdout.read(nb)
        if len(buf) < nb:
            break
        g = np.frombuffer(buf, np.uint8).reshape(dh, dw)
        rows = (g[y0:y1] < 100).mean(axis=1)
        hot = np.where(rows > 0.40)[0]
        if len(hot) and (hot.max() - hot.min()) <= 4:
            hits.append(t)
        t += step
    dec.wait()
    ivals = []
    for tt in hits:
        if ivals and tt - ivals[-1][1] <= 0.8:
            ivals[-1][1] = tt + step
        else:
            ivals.append([tt - 0.4, tt + step + 0.2])
    # a bar that PERSISTS is a design rule / table border (content), not a
    # swipe — once bridged 37s of a lecture behind a still. Swipes are brief.
    return [(max(0.0, a), b) for a, b in ivals if (b - a) <= 1.2]


def _adds_ink(clean: str, donor: float, t0: float, t1: float,
              margin: float = 1.06) -> bool:
    """True if the slide gains ink between `donor` and the [t0,t1] window —
    i.e. the teacher is writing. A still patched over such a window HIDES
    real content (a false-positive popup once hid a whole annotation pass).
    Any patch/bridge must refuse the window in that case."""
    def ink_at(t):
        raw = subprocess.run(
            ["ffmpeg", "-v", "error", "-ss", f"{max(0.0, t):.2f}", "-i", clean,
             "-frames:v", "1", "-vf", "scale=405:270", "-f", "rawvideo",
             "-pix_fmt", "gray", "-"], capture_output=True).stdout
        if len(raw) < 405 * 270:
            return None
        g = np.frombuffer(raw[:405 * 270], np.uint8).reshape(270, 405)
        return int((g < 200).sum())
    base = ink_at(donor)
    if not base:
        return False
    step = max(0.5, (t1 - t0) / 6.0)
    t = t0
    while t <= t1:
        cur = ink_at(t)
        if cur and cur > base * margin:
            return True
        t += step
    return False


def _popup_intervals(cleaned: str, sw: int, sh: int, step: float = 0.5):
    """Moments where Edge's text-SELECTION menu (Highlight / Add comment /
    Ask Copilot) floats over the page. Signature: the tiny multicoloured
    Copilot logo (3+ hues in a compact blob) sitting on a WHITE card, in the
    content area — a colourful photo fails the white-ring test. Patched like
    the Draw menu: a still from just before the popup opened."""
    half_w, half_h = sw // 2, sh // 2
    dec = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", cleaned, "-vf",
         f"fps={1 / step},scale={half_w}:{half_h}", "-f", "rawvideo",
         "-pix_fmt", "bgr24", "pipe:1"], stdout=subprocess.PIPE)
    nb = half_w * half_h * 3
    hits, t = [], 0.0
    while True:
        buf = dec.stdout.read(nb)
        if len(buf) < nb:
            break
        fr = np.frombuffer(buf, np.uint8).reshape(half_h, half_w, 3)
        hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
        sat = ((hsv[:, :, 1] > 90) & (hsv[:, :, 2] > 90))
        sat[:, :int(half_w * 0.08)] = False
        sat[:int(half_h * 0.05), :] = False
        n, lab, st, _c = cv2.connectedComponentsWithStats(
            sat.astype(np.uint8), 8)
        found = False
        for i in range(1, n):
            x, y, w2, h2, a = st[i]
            if not (3 <= a <= 120 and w2 <= 16 and h2 <= 16):
                continue
            hue = hsv[:, :, 0][lab == i]
            if len({int(v) // 30 for v in hue}) < 3:    # multicolour logo
                continue
            bm = (lab == i).astype(np.uint8)
            ring = (cv2.dilate(bm, np.ones((9, 9), np.uint8)) > 0) & (bm == 0)
            if not ring.any():
                continue
            if (int(np.median(hsv[:, :, 2][ring])) >= 225
                    and int(np.median(hsv[:, :, 1][ring])) <= 25):
                found = True
                break
        if found:
            hits.append(t)
        t += step
    dec.wait()
    ivals = []
    for tt in hits:
        if ivals and tt - ivals[-1][1] <= 1.5:
            ivals[-1][1] = tt + step
        else:
            ivals.append([tt - 0.6, tt + step + 0.3])
    # Pad the END generously: the menu FADES OUT, so its logo drops below the
    # detector threshold while the card is still faintly on screen (a popup
    # survived 0.25s past a patch window and the client would have seen it).
    # A selection menu lives SECONDS. A "popup" lasting longer is slide
    # content that merely looks like one (a map's tiny colour symbols on
    # white passed the logo test and produced a 24s patch that hid the
    # teacher's whole annotation pass — client caught it).
    return [(max(0.0, a), b + 1.2) for a, b in ivals if (b - a) <= 6.0]


def _panel_intervals(cleaned: str, sw: int, sh: int, step: float = 0.5):
    """Time intervals where the Draw/highlighter dropdown menu is open in the
    RAW (pre-grey-clean) source. The menu is an opaque card that sits ON TOP
    of the slide text, so whitening it still leaves the text underneath
    missing — those intervals get patched in composite() with a still of the
    same slide taken while the menu is closed."""
    DS = 4
    dw, dh = sw // DS, sh // DS
    dec = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", cleaned, "-vf",
         f"fps={1 / step},scale={dw}:{dh}", "-f", "rawvideo",
         "-pix_fmt", "bgr24", "pipe:1"], stdout=subprocess.PIPE)
    nb = dw * dh * 3
    xs_lim, ys_lim = int(dw * 0.20), int(dh * 0.45)
    hits, t = [], 0.0
    while True:
        buf = dec.stdout.read(nb)
        if len(buf) < nb:
            break
        fr = np.frombuffer(buf, np.uint8).reshape(dh, dw, 3)
        corner = cv2.cvtColor(fr[:ys_lim, :xs_lim], cv2.COLOR_BGR2HSV)
        # same swatch-ROW signature as detect.remove_ui_panel — a plain
        # "colours in the corner" rule matched the teacher's own pen strokes
        hits.append((t, _swatch_row(corner, scale=1.0 / DS)))
        t += step
    dec.wait()
    ivals, start = [], None
    for tt, ok in hits:
        if ok and start is None:
            start = tt
        elif not ok and start is not None:
            ivals.append((max(0.0, start - 0.4), tt + 0.4))
            start = None
    if start is not None:
        ivals.append((max(0.0, start - 0.4), t))
    return ivals


def _prerender_still(png_in: str, S_g: float, bx_g: int, by_g: int,
                     mt_g: int, sw: int, sh: int) -> str:
    """Bake a bridge/patch still into a ready 1920x1080 frame ONCE (scale,
    crop, white pad, page-number mask) so the ffmpeg graph only overlays a
    static image. Scaling every looped still on every frame made a composite
    with 20-40 bridges 5-10x slower than before."""
    OW, OH = C.OUT_W, C.OUT_H
    img = cv2.imread(png_in)
    zw, zh = max(1, round(sw * S_g)), max(1, round(sh * S_g))
    img = cv2.resize(img, (zw, zh), interpolation=cv2.INTER_AREA
                     if S_g < 1 else cv2.INTER_LINEAR)
    canvas = np.full((OH, OW, 3), 255, np.uint8)
    crop_top = -min(0, by_g)
    py_g = max(0, by_g)
    px_g = max(0, bx_g)
    src_h = min(zh - crop_top, OH - py_g)
    src_w = min(zw, OW - px_g)
    if src_h > 0 and src_w > 0:
        canvas[py_g:py_g + src_h, px_g:px_g + src_w] = \
            img[crop_top:crop_top + src_h, 0:src_w]
    if mt_g and mt_g > 2:
        canvas[:min(OH, mt_g), :] = 255
    out = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
    cv2.imwrite(out, canvas)
    try:
        os.unlink(png_in)
    except OSError:
        pass
    return out


def _refine_boundary(clean: str, t_approx: float, fps: int,
                     win: float = 0.8, mode: str = "onset") -> float:
    """Frame-accurate slide-change time near `t_approx`. The zoom splice must
    switch geometry on EXACTLY the first frame of the new slide — the coarse
    scan is ±0.25s, and a splice that lands a few frames early/late shows the
    same slide visibly zooming (client: "we can see the slide zoom out").
    Returns a timestamp just before the first new-slide frame."""
    tstart = max(0.0, t_approx - win)
    dec = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-ss", f"{tstart:.3f}", "-t", f"{2 * win:.3f}",
         "-i", clean, "-vf", "scale=270:180", "-f", "rawvideo",
         "-pix_fmt", "gray", "pipe:1"], stdout=subprocess.PIPE)
    nb = 270 * 180
    frames = []
    while True:
        buf = dec.stdout.read(nb)
        if len(buf) < nb:
            break
        frames.append(np.frombuffer(buf, np.uint8).astype(np.int16))
    dec.wait()
    if len(frames) < 3:
        return t_approx
    diffs = [float(np.abs(frames[i] - frames[i - 1]).mean())
             for i in range(1, len(frames))]
    mx = max(diffs)
    if mx < 4.0:                                    # no real cut nearby
        return t_approx
    # Snap to the ONSET of the change, not its peak: slide decks animate
    # transitions (push/rise over ~1s), and a boundary at the settle point
    # made the slide finish its own animation and THEN jump to the new
    # geometry. At the onset, the whole animation plays under the incoming
    # slide's geometry and lands exactly in place. For hard cuts the onset
    # and the peak are the same frame, so nothing changes there.
    thr = max(4.0, 0.35 * mx)
    if mode == "settle":
        # LAST significant change + 1 — where the animation finishes (used
        # for bridge ends: the still must hold until the swipe has landed)
        k = max(i for i, d in enumerate(diffs) if d >= thr)
        return tstart + (k + 2) / fps - 0.5 / fps
    k = next(i for i, d in enumerate(diffs) if d >= thr)
    return tstart + (k + 1) / fps - 0.5 / fps


def _plan_slide_overrides(slides, sw: int, sh: int,
                          S_base: float, bx_base: int, by_base: int):
    """For each scanned slide decide whether the BASE geometry (spec zoom,
    top-pinned) breaks it: content under the webcam circle, or content cropped
    at the bottom/top. Broken slides get their own centred, zoomed-out geometry
    — the smallest zoom-out that fixes the slide, never below BG_SCALE_FLOOR.
    Everything else keeps the spec look untouched.

    Returns [(t0, t1, S, bx, by, mask_top)] — mask_top is the y limit of a
    white strip painted over the revealed page margin (kills page numbers)."""
    OW, OH = C.OUT_W, C.OUT_H
    wcx, wcy = C.WEBCAM_OUT_X * OW, C.WEBCAM_OUT_Y * OH
    # Circle + generous safety margin: the ink mask is built on a 1/4-scale
    # image, so thin stroke ENDS get averaged away and the mask underestimates
    # the true ink extent by up to ~10px ("GoodsOut" ran into the circle even
    # after a rescue planned with a 12px margin).
    r_chk = C.WEBCAM_OUT_R + 24
    # resolution-normalised floor, same convention as composite (720p sources
    # need a proportionally larger factor to fill the same canvas fraction)
    s_floor = C.BG_SCALE_FLOOR * (OH / float(sh))

    def collides(S, bx, by, ys, xs):
        ox = bx + xs * S
        oy = by + ys * S
        return bool((((ox - wcx) ** 2 + (oy - wcy) ** 2) < r_chk * r_chk).any())

    overrides = []
    for sl in slides:
        if sl["t1"] - sl["t0"] < 1.0:               # sub-second sliver — skip
            continue
        # short flashed slides (1-2.5s) still get RESCUED if broken: a quickly
        # flipped photo slide once shipped with its title cut at the top; the
        # geometry switch sits on the slide's own cuts, so nothing flickers
        ct, cb = sl.get("uct", sl["ct"]), sl.get("ucb", sl["cb"])
        ys, xs = np.where(sl["ink"])
        if not len(ys):
            continue
        ys = ys.astype(np.float64) * sl["ds"]
        xs = xs.astype(np.float64) * sl["ds"]

        bottom_out = by_base + cb * S_base          # where slide bottom lands
        top_out = by_base + ct * S_base
        broken = (collides(S_base, bx_base, by_base, ys, xs)
                  or bottom_out > OH - 6            # cropped at the bottom
                  or top_out < 2)                   # cropped at the top
        if not broken:
            continue

        avail = OH - C.BG_TITLE_MARGIN - C.BG_BOTTOM_MARGIN
        s_fit = avail / max(1.0, cb - ct)
        S = min(S_base, s_fit)
        pick = None
        while S >= s_floor - 1e-9:
            bw = sw * S
            bx = (OW - bw) / 2
            by = (OH - (cb - ct) * S) / 2 - ct * S  # centre the slide content
            if not collides(S, bx, by, ys, xs):
                pick = (S, round(bx), round(by))
                break
            S -= 0.01
        if pick is None:                            # best effort at the floor
            S = s_floor
            bw = sw * S
            pick = (S, round((OW - bw) / 2),
                    round((OH - (cb - ct) * S) / 2 - ct * S))
        S, bx, by = pick
        mask_top = max(0, round(by + ct * S) - 12)
        overrides.append((sl["t0"], sl["t1"], S, bx, by, mask_top))

    # --- centering pass (client spec 2026-08: copy sits vertically CENTRED).
    # Healthy slides whose content is short get their own centred `by` at the
    # SAME scale; long copy centres to ~the same place as the top-pin and is
    # skipped (nothing may move enough to cut or to crawl under the webcam).
    recenters = []
    for sl in slides:
        if sl["t1"] - sl["t0"] < 2.5:
            continue
        mid = (sl["t0"] + sl["t1"]) / 2
        if any(o[0] <= mid <= o[1] for o in overrides):
            continue                                # zoom-rescued = centred already
        ys, xs = np.where(sl["ink"])
        if not len(ys):
            continue
        ys = ys.astype(np.float64) * sl["ds"]
        xs = xs.astype(np.float64) * sl["ds"]

        # Split the window into PHASES of stable content extent, so a slide
        # that grows/scrolls is centred phase by phase instead of once on the
        # union of everything (which left every phase visibly top-heavy).
        for (p0, p1, pct, pcb) in _extent_phases(sl):
            by_c = round((OH - (pcb - pct) * S_base) / 2 - pct * S_base)
            if by_c <= by_base + 24:                # negligible shift
                continue
            if collides(S_base, bx_base, by_c, ys, xs):
                continue                            # centring would hit the webcam
            mask_top = max(0, round(by_c + pct * S_base) - 12)
            recenters.append((p0, p1, by_c, mask_top))
    return overrides, recenters


def _extent_phases(sl, *, tol: int = 70, min_len: float = 2.5):
    """Break a slide into runs whose content extent is stable within `tol` px.
    Falls back to one phase spanning the slide when nothing varies."""
    ext = sl.get("ext") or []
    if len(ext) < 3:
        return [(sl["t0"], sl["t1"], sl.get("uct", sl["ct"]),
                 sl.get("ucb", sl["cb"]))]
    phases, run = [], [ext[0]]
    for e in ext[1:]:
        base_b = run[0][2]
        if abs(e[2] - base_b) > tol:
            phases.append(run)
            run = [e]
        else:
            run.append(e)
    phases.append(run)
    out = []
    for r in phases:
        t0 = r[0][0] if not out else r[0][0]
        t1 = r[-1][0]
        if t1 - t0 < min_len:                       # too short to re-seat
            if out:                                 # extend the previous phase
                p = out[-1]
                out[-1] = (p[0], t1, p[2], p[3])
            continue
        out.append((t0, t1, min(x[1] for x in r), max(x[2] for x in r)))
    if not out:
        return [(sl["t0"], sl["t1"], sl.get("uct", sl["ct"]),
                 sl.get("ucb", sl["cb"]))]
    out[0] = (sl["t0"], out[0][1], out[0][2], out[0][3])
    out[-1] = (out[-1][0], sl["t1"], out[-1][2], out[-1][3])
    return out


def _plan_panel_view(cam_src: str, bubble: Bubble, cside: int, D: int):
    """For side-panel cameras: how to place the (full-bubble) crop inside the
    output circle so the FACE ends up centred at the client's reference size.

    Returns (d2, ox, oy, wall_hex) — the crop is scaled to d2 (<= D), placed
    at (ox, oy) on a canvas filled with the bubble's own wall colour (the
    wall is uniform, so the seam is invisible), bottom-aligned so the body
    stays grounded at the circle's bottom edge.
    """
    cap = cv2.VideoCapture(cam_src)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    face_c = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = []
    frame0 = None
    for i in range(5):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * (i + 1) / 6))
        ok, fr = cap.read()
        if not ok:
            continue
        if frame0 is None:
            frame0 = fr
        x0 = max(0, bubble.cx - bubble.r)
        y0 = max(0, bubble.cy - bubble.r)
        box = fr[y0:bubble.cy + bubble.r, x0:bubble.cx + bubble.r]
        g = cv2.cvtColor(box, cv2.COLOR_BGR2GRAY)
        det = face_c.detectMultiScale(g, 1.1, 5, minSize=(50, 50))
        for (fx, fy, fw, fh) in det:
            faces.append((x0 + fx + fw / 2, y0 + fy + fh / 2, fh))
    cap.release()

    # wall colour: ring just inside the rim, upper half (no hair/shoulders)
    wall = (48, 48, 48)
    if frame0 is not None:
        ring_px = []
        for a in np.linspace(-2.7, -0.4, 40):
            px = int(bubble.cx + (bubble.r - 8) * np.cos(a))
            py = int(bubble.cy + (bubble.r - 8) * np.sin(a))
            if 0 <= py < frame0.shape[0] and 0 <= px < frame0.shape[1]:
                ring_px.append(frame0[py, px])
        if ring_px:
            wall = tuple(int(v) for v in np.median(np.array(ring_px), axis=0))
    wall_hex = f"0x{wall[2]:02x}{wall[1]:02x}{wall[0]:02x}"   # BGR -> RRGGBB

    if faces:
        fcx = float(np.median([f[0] for f in faces]))
        fcy = float(np.median([f[1] for f in faces]))
        fh = float(np.median([f[2] for f in faces]))
    else:                                        # no face found — keep as-is
        return D, 0, 0, wall_hex

    target = float(os.environ.get("BOOST_FACE_FRAC", "0.45"))
    d2 = int(round(target * D * cside / max(1.0, fh)))
    d2 = max(int(D * 0.72), min(D, d2))
    # place so the FACE lands on the circle centre (a touch above for
    # headroom); vertical lift is capped so the bubble stays bottom-heavy
    # (her body must stay grounded at the circle's bottom edge)
    scale2 = d2 / max(1, cside)
    face_dx = (fcx - bubble.cx) * scale2
    ox = int(round((D - d2) / 2 - face_dx))
    ox = max(0, min(D - d2, ox))
    face_in_crop_y = (fcy - (bubble.cy - cside / 2)) * scale2
    oy = int(round(D / 2 - 6 - face_in_crop_y))
    oy = max(int((D - d2) * 0.5), min(D - d2, oy))   # never above half-gap
    return d2, ox, oy, wall_hex


def composite(clean: str, cam_src: str, bubble: Bubble, dst: str) -> None:
    """Build the body: background (from `clean`, grey-removed + bubble painted),
    enlarged circular webcam (cropped from `cam_src`, the untouched source),
    robot overlay and Podcast-Voice audio."""
    info = probe(clean)
    sw, sh = info["w"], info["h"]
    OW, OH = C.OUT_W, C.OUT_H

    # --- background layer geometry. Fit ALL slide content into the frame so
    #     nothing is ever cropped: measure the highest + lowest content across
    #     the video, pick the largest scale (up to BG_SCALE) whose content span
    #     still fits between a top and bottom margin, and pin the content top just
    #     under the top margin (which also crops off the browser toolbar). If the
    #     content already fits at BG_SCALE we keep the original zoom. ---
    # BG_SCALE (1.12) is the SPEC zoom for the standard 1620x1080 recordings —
    # i.e. it describes how much of the canvas a slide should fill, not an
    # absolute factor. Normalise it by the source height so a 720p recording
    # fills the frame the same way (unnormalised, 1.12 rendered a 1080x720
    # lesson as a small slide floating in white space).
    res_norm = OH / float(sh)                           # 1.0 for 1080p sources
    S_spec = C.BG_SCALE * res_norm
    S_floor = C.BG_SCALE_FLOOR * res_norm
    S = S_spec
    bounds = _content_bounds_v(clean, sw, sh) if C.BG_ADAPTIVE else None
    if bounds:
        ct, cb = bounds                                 # raw px: top & bottom of content
        avail = OH - C.BG_TITLE_MARGIN - C.BG_BOTTOM_MARGIN
        s_fit = avail / max(1.0, cb - ct)               # scale at which content just fits
        S = max(S_floor, min(S_spec, s_fit))
    bw, bh = round(sw * S), round(sh * S)
    if bounds:
        by = round(C.BG_TITLE_MARGIN - ct * S)          # content top pinned below top margin
    else:
        by = round((OH - bh) / 2)                       # centred (no bounds / adaptive off)
    # Optional per-video vertical nudge (px, positive = down) — same idea as the
    # X offset below: lets a single re-render re-seat content (e.g. centre a
    # zoomed-out slide) without touching the spec geometry for everything else.
    by += round(float(os.environ.get("BOOST_BG_Y_OFFSET", "0")))
    # Optional per-video horizontal nudge (px, negative = left). Keeps the spec
    # scale (1.12) untouched — used to slide the background left just enough that a
    # handwritten annotation clears the fixed webcam, without zooming anything.
    bx = round((OW - bw) / 2 + float(os.environ.get("BOOST_BG_X_OFFSET", "0")))

    # --- per-slide rescue: slides whose content sits under the webcam circle or
    #     runs off the bottom at the spec zoom get their own centred zoom-out,
    #     spliced in exactly for that slide's time range. Every other slide keeps
    #     the spec geometry bit-for-bit. BOOST_SLIDE_ZOOM=0 disables. ---
    overrides = []
    recenters = []
    slides = []
    if os.environ.get("BOOST_SLIDE_ZOOM", "1") == "1" and bounds:
        try:
            slides = _slide_scan(clean, sw, sh)
            overrides, recenters = _plan_slide_overrides(slides, sw, sh, S, bx, by)
            overrides = overrides[:8]
            # snap both edges of every window to the EXACT slide-change frame,
            # so any geometry switch is hidden inside the content cut
            overrides = [
                (_refine_boundary(clean, t0, C.OUT_FPS),
                 _refine_boundary(clean, t1, C.OUT_FPS), S_i, bx_i, by_i, mt)
                for (t0, t1, S_i, bx_i, by_i, mt) in overrides]
            recenters = [
                (_refine_boundary(clean, t0, C.OUT_FPS),
                 _refine_boundary(clean, t1, C.OUT_FPS), by_i, mt)
                for (t0, t1, by_i, mt) in recenters]
            # TILE adjacent windows: fade transitions get dropped by the scan
            # as slivers, leaving a hole where BASE geometry leaked in — the
            # next slide opened misplaced and then "animated" into its centred
            # position (client flagged it). A ≤2s gap is a transition, not a
            # real slide: stretch the earlier window to meet the next one.
            recenters.sort(key=lambda r_: r_[0])
            for i in range(1, len(recenters)):
                pa, pb, pby, pmt = recenters[i - 1]
                a = recenters[i][0]
                if 0 < a - pb <= 2.0:
                    recenters[i - 1] = (pa, a, pby, pmt)
            for (t0, t1, S_i, _, _, _) in overrides:
                print(f"[slide-zoom] {t0:.2f}-{t1:.2f}s -> scale {S_i:.2f}",
                      file=sys.stderr, flush=True)
            if recenters:
                print(f"[slide-centre] {len(recenters)} slide(s) centred",
                      file=sys.stderr, flush=True)
        except Exception as exc:  # noqa: BLE001 — never fail the render on this
            print(f"[slide-zoom] scan failed: {exc}", file=sys.stderr, flush=True)
            overrides = []
            recenters = []
            slides = []

    # --- transition bridges: page swipes/holds between slides leak junk into
    #     the render (page-separator bar mid-screen, half-swiped pages, the
    #     "appears low then snaps up" hold). Bridge every inter-slide gap with
    #     a STILL of the previous slide — the viewer sees the old slide until
    #     the new one has settled, then a clean instant cut. Background only;
    #     webcam and audio stay live. BOOST_BRIDGE=0 disables. ---
    bridges = []                                    # (t0, t1, png, S,bx,by,mask)
    if os.environ.get("BOOST_BRIDGE", "1") == "1" and slides:
        def _geom_at(t):
            for (ra, rb, rby, rmt) in recenters:
                if ra <= t <= rb:
                    return S, bx, rby, rmt
            for (oa, ob2, oS, obx, oby, omt) in overrides:
                if oa <= t <= ob2:
                    return oS, obx, oby, omt
            return S, bx, by, 0
        for i in range(1, len(slides)):
            gap0, gap1 = slides[i - 1]["t1"], slides[i]["t0"]
            if gap1 - gap0 > 1.2:
                continue                            # a real hole (animated build,
                # slow scroll) — bridging it hid 5s of the next slide once
            a = _refine_boundary(clean, gap0, C.OUT_FPS) - 0.35
            # even a contiguous boundary hides a few swipe frames around the
            # cut (the scan's 0.25s grid can't see them) — bridge them too
            b2 = (_refine_boundary(clean, gap1, C.OUT_FPS, mode="settle")
                  if gap1 - gap0 > 0.05 else a + 0.35)
            if b2 - a < 0.15:
                continue
            donor = max(slides[i - 1]["t0"] + 0.3, gap0 - 0.8)
            png = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
            _run(["ffmpeg", "-v", "error", "-y", "-ss", f"{donor:.2f}",
                  "-i", clean, "-frames:v", "1", png])
            bridges.append((a, b2, png) + _geom_at(donor))
            if len(bridges) >= 48:
                break
        # sub-2s SCROLL-HOLD chunks: content MOVED through the window (the
        # union extent is far taller than the persistent one — e.g. a title
        # bisected mid-scroll held for 1.5s, which the client flags). Static
        # content flashes have union == persistent and are KEPT (client rule:
        # every original moment stays).
        for i in range(1, len(slides)):
            sl = slides[i]
            span_p = max(1, sl["cb"] - sl["ct"])
            span_u = sl.get("ucb", sl["cb"]) - sl.get("uct", sl["ct"])
            if (sl["t1"] - sl["t0"] < 2.0 and span_u > 1.3 * span_p):
                a = _refine_boundary(clean, sl["t0"], C.OUT_FPS) - 0.35
                b2 = _refine_boundary(clean, sl["t1"], C.OUT_FPS,
                                      mode="settle")
                if b2 - a < 0.15 or any(
                        x[0] - 0.4 <= a and b2 <= x[1] + 0.4 for x in bridges):
                    continue
                donor = max(slides[i - 1]["t0"] + 0.3,
                            slides[i - 1]["t1"] - 0.8)
                png = tempfile.NamedTemporaryFile(suffix=".png",
                                                  delete=False).name
                _run(["ffmpeg", "-v", "error", "-y", "-ss", f"{donor:.2f}",
                      "-i", clean, "-frames:v", "1", png])
                bridges.append((a, b2, png) + _geom_at(donor))
                if len(bridges) >= 48:
                    break

        # swipe bars at boundaries the slide scan can't see (near-identical
        # slides): find the bar frames directly and bridge over them too
        try:
            for (ba, bb) in _bar_intervals(clean, sw, sh):
                if any(x[0] - 0.3 <= ba and bb <= x[1] + 0.3 for x in bridges):
                    continue                        # already covered
                donor = max(0.3, ba - 0.6)
                png = tempfile.NamedTemporaryFile(suffix=".png",
                                                  delete=False).name
                _run(["ffmpeg", "-v", "error", "-y", "-ss", f"{donor:.2f}",
                      "-i", clean, "-frames:v", "1", png])
                bridges.append((ba, bb, png) + _geom_at(donor))
                if len(bridges) >= 48:
                    break
        except Exception as exc:  # noqa: BLE001
            print(f"[bridge] bar scan failed: {exc}",
                  file=sys.stderr, flush=True)
        for (a, b2, _p, *_g) in bridges:
            print(f"[bridge] {a:.2f}-{b2:.2f}s <- still of prev slide",
                  file=sys.stderr, flush=True)

    # --- panel patch: while the Draw menu is OPEN it covers slide text in the
    #     raw pixels, so whitening it still leaves a hole in the text. For each
    #     open interval, overlay a STILL of the same slide grabbed while the
    #     menu is closed — background only; webcam & audio stay live. ---
    patches = []                                    # (t0, t1, png_path)
    if os.environ.get("BOOST_PANEL_PATCH", "1") == "1" and bounds and slides:
        try:
            _iv = list(_panel_intervals(cam_src, sw, sh))[:4]
            _iv += list(_popup_intervals(cam_src, sw, sh))[:6]
            for (pa, pb) in _iv:
                mid = (pa + pb) / 2
                sl = next((s for s in slides
                           if s["t0"] <= mid <= s["t1"]), None)
                if sl is None:
                    continue
                if any(o[0] <= mid <= o[1] for o in overrides):
                    continue                        # zoomed slide — rare, skip
                donor = None
                if pb + 0.8 < sl["t1"] - 0.3:
                    donor = pb + 0.8                # after the menu closes
                elif pa - 0.8 > sl["t0"] + 0.3:
                    donor = pa - 0.8                # before it opened
                if donor is None:
                    continue                        # menu open the whole slide
                if _adds_ink(clean, donor, pa, pb):
                    print(f"[panel-patch] SKIP {pa:.1f}-{pb:.1f}s — content is "
                          "being added in this window", file=sys.stderr,
                          flush=True)
                    continue
                png = tempfile.NamedTemporaryFile(suffix=".png",
                                                  delete=False).name
                _run(["ffmpeg", "-v", "error", "-y", "-ss", f"{donor:.2f}",
                      "-i", clean, "-frames:v", "1", png])
                patches.append((pa, pb, png))
                print(f"[panel-patch] {pa:.1f}-{pb:.1f}s <- still @{donor:.1f}s",
                      file=sys.stderr, flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[panel-patch] failed: {exc}", file=sys.stderr, flush=True)
            patches = []

    # --- webcam: crop slightly inside the detected bubble (drops grey rim),
    #     then scale to a FIXED output diameter so size is consistent.
    #     Panel-kind bubbles crop at 1.0 — client spec 2026-08-14: "have their
    #     original bubble fit perfectly in our new bubble" (the output ring
    #     overlay hides the source rim line at the boundary). ---
    if getattr(bubble, "kind", "circle") == "panel":
        r = max(1, int(bubble.r))
    else:
        r = max(1, int(bubble.r * C.WEBCAM_MASK_SHRINK))
    cx0 = max(0, bubble.cx - r)
    cy0 = max(0, bubble.cy - r)
    cside = min(2 * r, sw - cx0, sh - cy0)

    # The bubble can JUMP mid-recording (a browser bar appears, the window is
    # nudged). A single crop position then takes page pixels from outside the
    # bubble for part of the video — a pale crescent inside our circle with the
    # face shoved off-centre, which is what the client saw in the conclusion of
    # "Sample Answers Part One". Follow it instead: crop x/y become per-frame
    # expressions for the spans where it moved.
    shifts = []
    if getattr(bubble, "kind", "circle") != "panel":
        try:
            shifts = detect.track_bubble(cam_src, bubble)
        except Exception as exc:  # noqa: BLE001
            print(f"[bubble-track] failed: {exc}", file=sys.stderr, flush=True)
    if shifts:
        for (sa, sb, sdx, sdy) in shifts:
            print(f"[bubble-track] {sa:.1f}-{sb:.1f}s shift ({sdx:+d},{sdy:+d})px",
                  file=sys.stderr, flush=True)
        cxe = "".join(f"if(between(t,{sa:.2f},{sb:.2f}),"
                      f"{max(0, min(sw - cside, cx0 + sdx))},"
                      for (sa, sb, sdx, sdy) in shifts) + str(cx0) + ")" * len(shifts)
        cye = "".join(f"if(between(t,{sa:.2f},{sb:.2f}),"
                      f"{max(0, min(sh - cside, cy0 + sdy))},"
                      for (sa, sb, sdx, sdy) in shifts) + str(cy0) + ")" * len(shifts)
        # no eval=frame: crop's x/y are runtime-tunable in this build, so the
        # expressions are already evaluated per frame (and `eval` is not an
        # option here — passing it kills the whole filtergraph)
        cam_crop = f"crop=w={cside}:h={cside}:x='{cxe}':y='{cye}'"
    else:
        cam_crop = f"crop={cside}:{cside}:{cx0}:{cy0}"
    D = 2 * C.WEBCAM_OUT_R                       # fixed output diameter
    r_outer = D // 2 - 2                         # circle footprint (unchanged)
    r_inner = max(1, r_outer - max(0, C.WEBCAM_RING))   # video circle (inset by ring)

    # video alpha mask = the (outer) circle — exactly the original, proven path.
    omask = np.zeros((D, D), np.uint8)
    cv2.circle(omask, (D // 2, D // 2), r_outer, 255, -1)
    # ring overlay = an RGBA white ANNULUS (inner..outer), transparent everywhere
    # else, so it can NEVER show as a square — it only paints the thin ring.
    ring = np.zeros((D, D, 4), np.uint8)
    cv2.circle(ring, (D // 2, D // 2), r_outer, (255, 255, 255, 255), -1)
    cv2.circle(ring, (D // 2, D // 2), r_inner, (0, 0, 0, 0), -1)
    ofile = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
    rfile = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
    cv2.imwrite(ofile, omask)
    cv2.imwrite(rfile, ring)

    # side-panel camera: place a SHRUNK full-bubble crop on a wall-coloured
    # canvas so the face sits centred at the reference size (client: "zoom
    # the presenter out"). The wall is uniform, so the seam is invisible.
    panel_view = None
    sfile = None
    # Shrink-reframe is OFF by default — the client chose the original bubble
    # mapped 1:1 onto the output circle ("fit perfectly"). BOOST_PANEL_REFRAME=1
    # re-enables the synthetic zoom-out if they ever change their mind.
    if (getattr(bubble, "kind", "circle") == "panel"
            and os.environ.get("BOOST_PANEL_REFRAME", "0") == "1"):
        d2, ox_p, oy_p, wall_hex = _plan_panel_view(cam_src, bubble, cside, D)
        if d2 < D:
            smask = np.zeros((d2, d2), np.uint8)
            cv2.circle(smask, (d2 // 2, d2 // 2), d2 // 2 - 1, 255, -1)
            smask = cv2.GaussianBlur(smask, (7, 7), 0)   # feathered seam
            sfile = tempfile.NamedTemporaryFile(suffix=".png",
                                                delete=False).name
            cv2.imwrite(sfile, smask)
            panel_view = (d2, ox_p, oy_p, wall_hex)
            print(f"[panel-view] bubble {cside}px -> {d2}px on {wall_hex} "
                  f"at ({ox_p},{oy_p})", file=sys.stderr, flush=True)

    wcx, wcy = C.WEBCAM_OUT_X * OW, C.WEBCAM_OUT_Y * OH
    wx, wy = round(wcx - D / 2), round(wcy - D / 2)

    # --- robot geometry ---
    robot = cv2.imread(C.ASSETS.robot, cv2.IMREAD_UNCHANGED)
    rh0, rw0 = robot.shape[:2]
    rw, rh = round(rw0 * C.ROBOT_SCALE), round(rh0 * C.ROBOT_SCALE)
    rcx, rcy = C.ROBOT_CENTER
    rx, ry = round(rcx - rw / 2), round(rcy - rh / 2)

    # background: base branch + one zoom branch per rescued slide. Zoom branches
    # are built in RGB so the white pad/mask stays true white regardless of the
    # source's YUV range, then overlaid only inside their slide's time window —
    # a video-only splice, the audio path is untouched (sync can't move).
    # centred slides shift the SAME scaled layer vertically — a per-frame `y`
    # expression on the base overlay, not extra scale branches (cheap even for
    # a dozen centred slides per video)
    if recenters:
        y_expr = "".join(f"if(between(t,{a:.2f},{b:.2f}),{by_i},"
                         for (a, b, by_i, _mt) in recenters)
        y_expr += str(by) + ")" * len(recenters)
        base_y = f"'{y_expr}'"
    else:
        base_y = str(by)

    n_ov = len(overrides)
    if n_ov:
        bg_split = (f"[0:v]setpts=PTS-STARTPTS,split={n_ov + 1}"
                    + "".join(f"[bgs{i}]" for i in range(n_ov + 1)) + ";")
        bg_base = f"[bgs0]scale={bw}:{bh},setsar=1[bgv];"
        zoom_branches = ""
        zoom_chain = ""
        prev_lbl = "bg0"
        for i, (t0, t1, S_i, bx_i, by_i, mask_top) in enumerate(overrides):
            zw, zh = round(sw * S_i), round(sh * S_i)
            px = max(0, bx_i)
            py_ = max(0, by_i)
            crop_top = -min(0, by_i)                # frame top above the screen
            ch = min(zh - crop_top, OH - py_)       # and clip anything below it
            zoom_branches += (
                f"[bgs{i + 1}]scale={zw}:{zh},format=rgb24,"
                f"crop={zw}:{ch}:0:{crop_top},"
                f"pad={OW}:{OH}:{px}:{py_}:white"
                + (f",drawbox=x=0:y=0:w={OW}:h={mask_top}:color=white:t=fill"
                   if mask_top > 0 else "")
                + f",format=yuv420p,setsar=1[z{i}];")
            nxt = f"bg{i + 1}"
            zoom_chain += (f"[{prev_lbl}][z{i}]overlay="
                           f"enable='between(t,{t0:.2f},{t1:.2f})'[{nxt}];")
            prev_lbl = nxt
        bg_graph = (bg_split + bg_base
                    + f"[base0][bgv]overlay={bx}:{base_y}[bg0];"
                    + zoom_branches + zoom_chain)
        bg_out = prev_lbl
    else:
        bg_graph = (f"[0:v]setpts=PTS-STARTPTS,scale={bw}:{bh},setsar=1[bgv];"
                    f"[base0][bgv]overlay={bx}:{base_y}[bg0];")
        bg_out = "bg0"

    # white strip above centred content (page numbers / margin junk revealed
    # by the downward shift), per centred window; RGB roundtrip keeps the
    # painted white true white regardless of the stream's YUV range
    rc_masks = [(a, b, mt) for (a, b, _byi, mt) in recenters if mt > 2]
    if rc_masks:
        chain = "format=rgb24" + "".join(
            f",drawbox=x=0:y=0:w={OW}:h={mt}:color=white:t=fill:"
            f"enable='between(t,{a:.2f},{b:.2f})'"
            for (a, b, mt) in rc_masks) + ",format=yuv420p"
        bg_graph += f"[{bg_out}]{chain}[bgrc];"
        bg_out = "bgrc"

    # panel-patch stills: base-geometry overlays active only inside their
    # menu-open window (inputs 6+i, added to the command below)
    for i, (pa, pb, _png) in enumerate(patches):
        # trim ends the looped still shortly after its window — without it the
        # infinite image input keeps the encode alive after the video ends.
        # If the patched window sits inside a CENTRED slide, the still must
        # land at that slide's centred y, not the base top-pin.
        mid_p = (pa + pb) / 2
        by_p = next((by_i for (a, b, by_i, _mt) in recenters
                     if a <= mid_p <= b), by)
        baked = _prerender_still(_png, S, bx, by_p, 0, sw, sh)
        patches[i] = (pa, pb, baked)
        bg_graph += (
            f"[{6 + i}:v]trim=end={pb + 2:.2f},format=yuv420p,setsar=1[pp{i}];"
            f"[{bg_out}][pp{i}]overlay=0:0:eof_action=pass:"
            f"enable='between(t,{pa:.2f},{pb:.2f})'[ppo{i}];")
        bg_out = f"ppo{i}"

    # transition-bridge stills: previous slide held over the swipe window,
    # rendered with that slide's own geometry (full-frame, pad-white)
    for j, (ba, bb, _png, S_g, bx_g, by_g, mt_g) in enumerate(bridges):
        idx = 6 + len(patches) + j
        baked = _prerender_still(_png, S_g, bx_g, by_g, mt_g, sw, sh)
        bridges[j] = (ba, bb, baked, S_g, bx_g, by_g, mt_g)
        bg_graph += (
            f"[{idx}:v]trim=end={bb + 2:.2f},format=yuv420p,setsar=1[br{j}];"
            f"[{bg_out}][br{j}]overlay=0:0:eof_action=pass:"
            f"enable='between(t,{ba:.2f},{bb:.2f})'[bro{j}];")
        bg_out = f"bro{j}"

    if panel_view:
        d2, ox_p, oy_p, wall_hex = panel_view
        smask_idx = 6 + len(patches) + len(bridges)   # sfile appended last
        cam_graph = (
            f"[2:v]setpts=PTS-STARTPTS,{cam_crop},"
            f"scale={d2}:{d2},setsar=1[camsm];"
            f"[{smask_idx}:v]format=gray,scale={d2}:{d2}[smask];"
            f"[camsm][smask]alphamerge[camsf];"
            f"color=c={wall_hex}:s={D}x{D}:r={C.OUT_FPS},format=yuv420p,"
            f"setsar=1[wallbg];"
            f"[wallbg][camsf]overlay={ox_p}:{oy_p}:shortest=1[camraw];")
    else:
        cam_graph = (
            f"[2:v]setpts=PTS-STARTPTS,{cam_crop},"
            f"scale={D}:{D},setsar=1[camraw];")

    fc = (
        f"[1:v]scale={OW}:{OH},setsar=1[base0];"
        + bg_graph + cam_graph +
        f"[4:v]format=gray,scale={D}:{D}[omask];"
        f"[camraw][omask]alphamerge[camc];"
        f"[{bg_out}][camc]overlay={wx}:{wy}[withcam0];"
        f"[5:v]format=rgba,scale={D}:{D}[ring];"
        f"[withcam0][ring]overlay={wx}:{wy}[withcam];"
        f"[3:v]scale={rw}:{rh}[robot];"
        f"[withcam][robot]overlay={rx}:{ry}[outv];"
        # audio from the ORIGINAL (input 2 = cam_src)
        + (
            # "Podcast Voice"-style cleanup: denoise, band-limit, compress, normalise
            "[2:a]highpass=f=90,lowpass=f=12000,afftdn=nr=12,"
            "acompressor=threshold=-18dB:ratio=3:attack=5:release=120,"
            "loudnorm=I=-16:TP=-1.5:LRA=11[outa]"
            if C.PROCESS_AUDIO else
            # raw audio, untouched (just resampled for the container)
            "[2:a]aresample=44100[outa]"
        )
    )

    patch_inputs = []
    for (_pa, _pb, png) in patches:
        patch_inputs += ["-loop", "1", "-framerate", "1", "-i", png]
    for (_ba, _bb, png, *_g) in bridges:
        patch_inputs += ["-loop", "1", "-framerate", "1", "-i", png]
    if sfile:
        patch_inputs += ["-i", sfile]
    _run([
        "ffmpeg", "-v", "error", "-y",
        "-i", clean, "-i", C.ASSETS.white_bg, "-i", cam_src,
        "-i", C.ASSETS.robot, "-i", ofile, "-i", rfile,
        *patch_inputs,
        "-filter_complex", fc, "-map", "[outv]", "-map", "[outa]",
        "-r", str(C.OUT_FPS),
        *C.final_codec_args(), "-c:a", "aac", "-b:a", "192k", dst,
    ])
    os.unlink(ofile)
    os.unlink(rfile)
    if sfile:
        try:
            os.unlink(sfile)
        except OSError:
            pass
    for (_pa, _pb, png) in patches:
        try:
            os.unlink(png)
        except OSError:
            pass
    for (_ba, _bb, png, *_g) in bridges:
        try:
            os.unlink(png)
        except OSError:
            pass
    assert_av_sync(dst)


def assert_av_sync(path: str, tolerance: float = 0.5) -> None:
    # NB: ~0.2s of stream-duration skew is normal container/codec padding (AAC
    # priming + fps rounding) — known-good deliveries show it. Real conform-mux
    # desync was measured in whole seconds, so 0.5s cleanly separates the two.
    """Fail loudly if the audio and video stream durations have drifted apart.
    Audio and video are always cut together from the same input in this
    pipeline, so any drift means a bug — better to kill the render than to
    deliver a silently desynced video."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,duration",
         "-of", "json", path], capture_output=True, text=True).stdout
    durs = {}
    for st in json.loads(out).get("streams", []):
        try:
            durs[st["codec_type"]] = float(st.get("duration") or 0)
        except (TypeError, ValueError):
            pass
    if "video" in durs and "audio" in durs and durs["video"] and durs["audio"]:
        drift = abs(durs["video"] - durs["audio"])
        if drift > tolerance:
            raise RuntimeError(
                f"A/V duration drift {drift:.2f}s in {os.path.basename(path)} "
                f"(video {durs['video']:.2f}s vs audio {durs['audio']:.2f}s)")


# ---------------------------------------------------------------------------
# Stage 3 - intro title card (stand-in for SC Intro.mogrt)
# ---------------------------------------------------------------------------
def make_intro(title: str, dst: str, duration: float = C.INTRO_DURATION) -> None:
    """Title card matching the SC Intro / SCI_FINAL look: teal background, the
    lesson topic in white, "studyclix" beneath."""
    OW, OH = C.OUT_W, C.OUT_H

    def esc(s: str) -> str:
        return (s.replace("\\", r"\\").replace(":", r"\:")
                 .replace("'", "’").replace("%", r"\%"))

    # shrink the font if the title is long so it always fits on one line
    fontsize = 96 if len(title) <= 28 else max(54, int(96 * 28 / len(title)))
    safe = esc(title)

    fc = (
        f"color=c={C.INTRO_TEAL}:s={OW}x{OH}:r={C.OUT_FPS}:d={duration}[bg];"
        f"[bg]drawtext=fontfile='{FONT}':text='{safe}':fontcolor=white:"
        f"fontsize={fontsize}:x=(w-text_w)/2:y=h*0.40[t1];"
        f"[t1]drawtext=fontfile='{FONT}':text='studyclix':fontcolor=white:"
        f"fontsize=58:x=(w-text_w)/2:y=h*0.56[t2];"
        f"[t2]fade=t=in:st=0:d=0.5,fade=t=out:st={duration-0.6}:d=0.6[outv]"
    )
    _run([
        "ffmpeg", "-v", "error", "-y",
        "-f", "lavfi", "-i", f"anullsrc=r=44100:cl=stereo:d={duration}",
        "-filter_complex", fc, "-map", "[outv]", "-map", "0:a",
        *C.video_codec_args(), "-c:a", "aac", dst,
    ])


def has_audio(video: str) -> bool:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a",
         "-show_entries", "stream=index", "-of", "csv=p=0", video],
        capture_output=True, text=True,
    ).stdout.strip()
    return bool(out)


def _normalize(src: str, dst: str) -> None:
    """Conform any clip to OUT_W x OUT_H @ OUT_FPS, always with a stereo track."""
    vf = (f"scale={C.OUT_W}:{C.OUT_H}:force_original_aspect_ratio=decrease,"
          f"pad={C.OUT_W}:{C.OUT_H}:(ow-iw)/2:(oh-ih)/2:white,"
          f"fps={C.OUT_FPS},setsar=1")
    if has_audio(src):
        cmd = ["ffmpeg", "-v", "error", "-y", "-i", src, "-vf", vf,
               "-af", "aformat=sample_rates=44100:channel_layouts=stereo",
               "-map", "0:v:0", "-map", "0:a:0"]
    else:
        cmd = ["ffmpeg", "-v", "error", "-y", "-i", src,
               "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
               "-vf", vf, "-map", "0:v:0", "-map", "1:a:0", "-shortest"]
    cmd += [*C.video_codec_args(), "-c:a", "aac", dst]
    _run(cmd)


# ---------------------------------------------------------------------------
# Stage 4 - assemble intro -> body -> outro with cross-fades
# ---------------------------------------------------------------------------
def assemble(intro: str, body: str, outro_src: str, dst: str, xfade: float = 0.6) -> None:
    tmp = tempfile.mkdtemp(prefix="boost_")
    outro = os.path.join(tmp, "outro_norm.mp4")
    _normalize(outro_src, outro)

    di = probe(intro)["duration"]
    db = probe(body)["duration"]
    o1 = di - xfade            # intro->body transition start
    o2 = o1 + db - xfade       # body->outro transition start

    fc = (
        f"[0:v][1:v]xfade=transition=fade:duration={xfade}:offset={o1}[v01];"
        f"[v01][2:v]xfade=transition=fade:duration={xfade}:offset={o2}[outv];"
        f"[0:a][1:a]acrossfade=d={xfade}[a01];"
        f"[a01][2:a]acrossfade=d={xfade}[outa]"
    )
    _run([
        "ffmpeg", "-v", "error", "-y",
        "-i", intro, "-i", body, "-i", outro,
        "-filter_complex", fc, "-map", "[outv]", "-map", "[outa]",
        *C.final_codec_args(), "-c:a", "aac", "-b:a", "192k",
        "-r", str(C.OUT_FPS), dst,
    ])


# ---------------------------------------------------------------------------
# Auto-cut dead time (silent scroll-throughs / long pauses)
# ---------------------------------------------------------------------------
def _detect_silences(src: str, noise_db: float, min_sil: float):
    p = subprocess.run(
        ["ffmpeg", "-hide_banner", "-i", src, "-af",
         f"silencedetect=noise={noise_db}dB:d={min_sil}", "-f", "null", "-"],
        capture_output=True, text=True)
    t = p.stderr
    starts = [float(x) for x in re.findall(r"silence_start: (-?[0-9.]+)", t)]
    ends = [float(x) for x in re.findall(r"silence_end: (-?[0-9.]+)", t)]
    return starts, ends


def _silencedetect_keep(src: str, dur: float, noise_db: float,
                        min_sil: float, pad: float) -> list:
    """Fallback keep-list from amplitude silencedetect (used if VAD unavailable)."""
    starts, ends = _detect_silences(src, noise_db, min_sil)
    removed = []
    for i, s in enumerate(starts):
        e = ends[i] if i < len(ends) else dur
        a, b = s + pad, e - pad
        if b - a > 0.2:
            removed.append((a, b))
    keep, cur = [], 0.0
    for a, b in sorted(removed):
        if a > cur:
            keep.append((cur, a))
        cur = max(cur, b)
    if cur < dur:
        keep.append((cur, dur))
    # Drop short "kept" islands (isolated clicks / head-silence slivers).
    return [(a, b) for a, b in keep if b - a >= C.AUTOCUT_MIN_KEEP]


def compute_keep(src: str, min_sil: float, pad: float,
                 noise_db: float) -> list:
    """Keep-ranges for the auto-cut. VAD (speech vs non-speech) is primary; it
    cleanly drops start fumbling / dead air without clipping quiet speech. Falls
    back to amplitude silencedetect only if VAD is unavailable."""
    dur = probe(src)["duration"]
    try:
        from . import speech
        segs = speech.detect_speech(src)
        keep = speech.speech_keep_ranges(segs, dur, min_sil, pad)
        if keep:
            return keep
    except Exception as e:  # noqa: BLE001 — degrade gracefully to silencedetect
        print(f"[autocut] VAD unavailable ({e}); using silencedetect", flush=True)
    return _silencedetect_keep(src, dur, noise_db, min_sil, pad)


def autocut(src: str, dst: str, noise_db: float = None,
            min_sil: float = None, pad: float = None) -> bool:
    """Remove dead time (start fumbling, long pauses, dead scroll time), keeping
    `pad` seconds around speech and natural pauses shorter than `min_sil`. Cuts
    video AND audio together. Returns True if a cut was made, False otherwise."""
    noise_db = C.AUTOCUT_NOISE_DB if noise_db is None else noise_db
    min_sil = C.AUTOCUT_MIN_SILENCE if min_sil is None else min_sil
    pad = C.AUTOCUT_PAD if pad is None else pad

    dur = probe(src)["duration"]
    keep = compute_keep(src, min_sil, pad, noise_db)
    if not keep:
        return False
    # Nothing meaningful trimmed (one block spanning ~the whole clip) → skip.
    if len(keep) == 1 and keep[0][0] < 0.3 and keep[0][1] > dur - 0.3:
        return False
    _render_keep(src, dst, keep)
    return True


def _render_keep(src: str, dst: str, keep: list, mute: list = None) -> None:
    """Re-assemble `src` from the kept (a, b) time-ranges, video + audio together.
    `mute` = (s, e) ranges (in source time) whose audio is silenced in place
    (e.g. breaths) — timing is unchanged, only the inhale is muted."""
    mute = mute or []
    parts, labels = [], ""
    for i, (a, b) in enumerate(keep):
        parts.append(f"[0:v]trim={a}:{b},setpts=PTS-STARTPTS[v{i}]")
        # breaths that fall in this kept segment, shifted to the segment's 0-base
        seg = [(max(ms, a) - a, min(me, b) - a) for ms, me in mute
               if min(me, b) - max(ms, a) > 0.02]
        afilt = f"[0:a]atrim={a}:{b},asetpts=PTS-STARTPTS"
        if seg:
            en = "+".join(f"between(t,{s:.3f},{e:.3f})" for s, e in seg)
            afilt += f",volume=enable='{en}':volume=0"
        parts.append(f"{afilt}[a{i}]")
        labels += f"[v{i}][a{i}]"
    parts.append(f"{labels}concat=n={len(keep)}:v=1:a=1[vo][ao]")
    _run(["ffmpeg", "-v", "error", "-y", "-i", src,
          "-filter_complex", ";".join(parts), "-map", "[vo]", "-map", "[ao]",
          *C.video_codec_args(), "-c:a", "aac", dst])


def _ranges_to_keep(cut: list, dur: float, min_keep: float = 0.05) -> list:
    """Complement of `cut` (sorted, merged removal ranges) over [0, dur]."""
    keep, cur = [], 0.0
    for a, b in sorted(cut):
        a, b = max(0.0, a), min(dur, b)
        if a > cur:
            keep.append((cur, a))
        cur = max(cur, b)
    if cur < dur:
        keep.append((cur, dur))
    return [(a, b) for a, b in keep if b - a > min_keep]


def _subtract_ranges(keep: list, cut: list) -> list:
    """Remove `cut` ranges from `keep` segments, splitting where they overlap."""
    cut = sorted(cut)
    out = []
    for ks, ke in keep:
        segs = [(ks, ke)]
        for cs, ce in cut:
            if ce <= ks or cs >= ke:
                continue
            nxt = []
            for s, e in segs:
                if ce <= s or cs >= e:
                    nxt.append((s, e))
                else:
                    if cs > s:
                        nxt.append((s, min(cs, e)))
                    if ce < e:
                        nxt.append((max(ce, s), e))
            segs = nxt
        out.extend(segs)
    return [(a, b) for a, b in out if b - a > 0.05]


def content_cleanup(src: str, dst: str, *, use_cleanvoice: bool = None,
                    min_sil: float = None, pad: float = None,
                    noise_db: float = None, warn=None) -> bool:
    """One-pass content cleanup: local VAD dead-time removal + (optional)
    Cleanvoice filler/cough/stutter removal, combined into a single re-encode.
    Returns True if anything was cut (dst written)."""
    use_cleanvoice = C.CLEANVOICE if use_cleanvoice is None else use_cleanvoice
    min_sil = C.AUTOCUT_MIN_SILENCE if min_sil is None else min_sil
    pad = C.AUTOCUT_PAD if pad is None else pad
    noise_db = C.AUTOCUT_NOISE_DB if noise_db is None else noise_db

    dur = probe(src)["duration"]
    keep = compute_keep(src, min_sil, pad, noise_db)
    if not keep:
        return False

    mute: list = []
    if use_cleanvoice:
        try:
            from . import cleanvoice
            tmp_audio = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False).name
            _run(["ffmpeg", "-v", "error", "-y", "-i", src,
                  "-ac", "1", "-ar", "44100", "-b:a", "128k", tmp_audio])
            try:
                cut, mute = cleanvoice.cut_and_mute_ranges(tmp_audio)
            finally:
                os.unlink(tmp_audio)
            if cut:
                keep = _subtract_ranges(keep, cut)
            # Protect speech: Cleanvoice sometimes mislabels a word stumble / word
            # edge as a breath. Only mute the parts of a breath that fall in a VAD
            # non-speech gap (padded), so actual words are never silenced.
            if mute:
                try:
                    from . import speech
                    sp = [(max(0.0, s - 0.12), e + 0.12)
                          for s, e in speech.detect_speech(src)]
                    mute = _subtract_ranges(mute, sp)
                except Exception as e:  # noqa: BLE001 — no VAD → don't risk words
                    print(f"[content_cleanup] breath-mute gating off ({e})", flush=True)
                    mute = []
        except Exception as e:  # noqa: BLE001 — never fail the job on the API
            print(f"[content_cleanup] Cleanvoice skipped ({e})", flush=True)
            mute = []
            if warn:
                warn("Cleanvoice unreachable — audio was NOT cleaned (silence "
                     "trim still applied). Re-render to retry.")

    if not keep:
        return False
    if len(keep) == 1 and keep[0][0] < 0.3 and keep[0][1] > dur - 0.3 and not mute:
        return False
    _render_keep(src, dst, keep, mute=mute)
    return True


def cleanvoice_cut(src: str, dst: str, config: dict = None) -> bool:
    """Cut Cleanvoice-flagged fillers / coughs / stutters / dead air from `src`.
    Returns True if anything was cut (dst written), False otherwise."""
    from . import cleanvoice

    # Cleanvoice only needs the audio.
    tmp_audio = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False).name
    _run(["ffmpeg", "-v", "error", "-y", "-i", src,
          "-ac", "1", "-ar", "44100", "-b:a", "128k", tmp_audio])
    try:
        cut = cleanvoice.cut_ranges(tmp_audio, config)
    finally:
        os.unlink(tmp_audio)
    if not cut:
        return False

    dur = probe(src)["duration"]
    keep = _ranges_to_keep(cut, dur)
    if not keep or (len(keep) == 1 and keep[0][1] - keep[0][0] > dur - 0.1):
        return False
    _render_keep(src, dst, keep)
    return True
