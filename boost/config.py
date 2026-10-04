"""
Central configuration for the Boost editing pipeline.

All geometric constants were decoded from the two Premiere Pro presets that the
manual workflow uses:
    Premiere Pro Presets - JC Boost/JC Lesson - Background.prfpset.xml
    Premiere Pro Presets - JC Boost/JC Lesson - Webcam.prfpset.xml

The presets store AE.ADBE Motion (Position / Scale / Anchor) values in
normalized [0..1] coordinates, which we reproduce here with ffmpeg/OpenCV.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

# Project root = the "black camel project" folder (parent of this package).
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def asset(*parts: str) -> str:
    return os.path.join(PROJECT_ROOT, *parts)


# ---------------------------------------------------------------------------
# Video encoder: switchable CPU (libx264) <-> GPU (NVIDIA NVENC) at runtime.
#   Affects the main full-length encodes (composite / intro / assemble / editor).
#   The segmented background pass always encodes its tiny segments on CPU — its
#   encode time is negligible and 24 parallel NVENC sessions would exceed the
#   GPU's session limit. GPU mode mainly speeds the big single-stream encodes.
# ---------------------------------------------------------------------------
ENCODER = os.environ.get("BOOST_ENCODER", "cpu").lower()   # "cpu" | "gpu"
GPU_VCODEC = "h264_nvenc"


def video_codec_args(mode: "str | None" = None) -> list:
    """Fast args for INTERMEDIATE encodes (segments / plates that get re-encoded)."""
    m = (mode or ENCODER or "cpu").lower()
    if m == "gpu":
        return ["-c:v", GPU_VCODEC, "-preset", "p5", "-rc", "vbr",
                "-cq", "23", "-b:v", "0", "-pix_fmt", "yuv420p"]
    return ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-pix_fmt", "yuv420p"]


# High-bitrate H.264 for the FINAL user-facing export (crisp, generous bitrate).
FINAL_MAXRATE = os.environ.get("BOOST_MAXRATE", "24M")
FINAL_BUFSIZE = os.environ.get("BOOST_BUFSIZE", "48M")


def final_codec_args(mode: "str | None" = None) -> list:
    """High-quality / high-bitrate H.264 args for the exported video."""
    m = (mode or ENCODER or "cpu").lower()
    if m == "gpu":
        return ["-c:v", GPU_VCODEC, "-preset", "p6", "-rc", "vbr",
                "-cq", "19", "-b:v", "12M",
                "-maxrate", FINAL_MAXRATE, "-bufsize", FINAL_BUFSIZE,
                "-pix_fmt", "yuv420p"]
    return ["-c:v", "libx264", "-preset", "medium", "-crf", "17",
            "-maxrate", FINAL_MAXRATE, "-bufsize", FINAL_BUFSIZE,
            "-pix_fmt", "yuv420p"]


# ---------------------------------------------------------------------------
# Output sequence
# ---------------------------------------------------------------------------
OUT_W = 1920
OUT_H = 1080
OUT_FPS = 30  # source is 60fps; 30 is plenty for screen lessons and halves work

# ---------------------------------------------------------------------------
# Background layer (JC Lesson - Background preset)
#   Scale 112%, centered. Scaling up pushes the Loom/PDF toolbars off the edges.
# ---------------------------------------------------------------------------
# Default/maximum background scale (Premiere preset "JC Lesson - Background" =
# 112%). The raw (1620x1080) and output (1920x1080) share the same height, so
# scale > 1 overflows vertically and is cropped 50/50 (top toolbar + bottom).
BG_SCALE = 1.12
# Fit-to-content: per video, measure the highest + lowest slide content and pick
# the largest scale (up to BG_SCALE) whose whole content span fits between the top
# and bottom margins, then pin the content top under the top margin (also crops the
# toolbar). Content that already fits keeps BG_SCALE. This guarantees no slide is
# cropped top or bottom (recurring client feedback: images cut off at the edge).
# DISABLED (2026-07-13). The temporal-median "stray artifact" repair in the
# webcam-surround ring (boost/artifacts.py) turned out to ERASE the teacher's
# handwritten annotations: he writes right next to the bubble, inside the ring,
# and a short-lived annotation looks exactly like a transient outlier against the
# whole-video median — so it got wiped, partially (leaving a grey ghost/faded
# tail). Client-reported as "bubble covering annotations" / "grey shadowy mark".
# The premise (that ring = never content) is simply wrong. Off until it can tell
# an annotation from an artifact, which per-pixel median cannot.
REPAIR_ARTIFACTS = os.environ.get("BOOST_REPAIR_ARTIFACTS", "0") == "1"
BG_ADAPTIVE = os.environ.get("BOOST_BG_ADAPTIVE", "1") == "1"
BG_SCALE_FLOOR = 0.85          # allow zooming out this far to fit tall slides
BG_TITLE_MARGIN = 24           # output px of breathing room above the top content
BG_BOTTOM_MARGIN = 24          # output px of breathing room below the bottom content

# ---------------------------------------------------------------------------
# Webcam layer (JC Lesson - Webcam preset)
#   Source bubble is cropped, scaled 165% and re-placed at a fixed output point.
#   The *input* bubble location varies per recording, so it is auto-detected;
#   the *output* placement below is fixed (preset Position 0.891 : 0.305).
# ---------------------------------------------------------------------------
# Webcam bubble is placed at a FIXED output size + position (matched to the
# client's reference edits), independent of the detected source bubble size.
WEBCAM_OUT_X = 0.88333  # normalized centre X = 1696px → 42px gap to right edge (client spec)
WEBCAM_OUT_Y = 0.2032  # normalized centre Y (measured from reference.mp4: 220px)
WEBCAM_OUT_R = 184     # fixed output radius in px (measured from reference.mp4)
# Crop INSIDE the detected bubble so the Loom bubble's own white border ring is
# excluded (it otherwise shows as a light "contour" that overlaps background
# objects). 0.90 cuts inside that ring while keeping the whole head. Output size
# is unaffected — the crop is scaled up to WEBCAM_OUT_R.
WEBCAM_MASK_SHRINK = 0.98   # was 0.90 — cropped the top of the presenter's
# head and threw away ~17% of the source pixels (client reference shows the
# FULL bubble). The output ring overlay hides the bubble's own rim.
# Clean white ring around the webcam: invisible on the white slide (white-on-
# white) but cleanly separates the circle from any coloured content (e.g. a
# zoomed-in table) so the camera never looks like it sits "on" an object. The
# overall footprint is unchanged (ring is inset); only the video is ~this many px
# smaller in radius. 0 = no ring.
WEBCAM_RING = 8
# Fallback bubble location (normalized, in source) if detection fails.
WEBCAM_FALLBACK = (0.915, 0.195, 0.07)  # (cx, cy, radius) normalized to width
                                        # measured from example.mp4 raw (same teacher setup)

# ---------------------------------------------------------------------------
# Grey Word/PDF background removal
#   The document background is a flat neutral grey (~229,229,229); the page is
#   white (255). We treat low-saturation pixels in this luminance band as grey
#   and repaint them white. Covers both side margins and scrolling page breaks.
# ---------------------------------------------------------------------------
# Edge-connected background removal (handles light AND dark viewer themes).
BG_MAX_SAT = 26        # max saturation to count a pixel as neutral chrome
BG_GREY_LO = 125       # light/medium viewer canvas band (canvas measures 228)
# Cap includes the viewer SCROLLBAR track (250) so it is removed; the white page
# (253) stays out and still blocks the flood. Light slide fills (~246) fall in the
# band but are INTERIOR — the flood is seeded only from the left/right/top edges
# (never the bottom), so scrolled content running off the frame bottom is safe.
BG_GREY_HI = 250
BG_DARK_LO = 32        # dark viewer canvas band (Acrobat theme ~58)
BG_DARK_HI = 124       # (excludes near-black document text < ~32); meets grey band

# ---------------------------------------------------------------------------
# Robot mascot overlay (pose1-smile.png), Premiere scale 20, position 145 x 950.
#   Premiere position is the clip *centre* in sequence pixels.
# ---------------------------------------------------------------------------
ROBOT_SCALE = 0.20
ROBOT_CENTER = (145.0, 950.0)   # (x, y) centre in output pixels — bottom-left
ROBOT_CORNER = "bottom-left"    # confirmed left by reference (SCI_FINAL)

# ---------------------------------------------------------------------------
# Intro title card (matches SC Intro / SCI_FINAL look)
#   Teal background, white lesson topic, "studyclix" beneath.
# ---------------------------------------------------------------------------
INTRO_TEAL = "0x0AAE9F"     # RGB(10,174,159) brand teal
INTRO_DURATION = 4.0

# Whether to prepend the intro title card / append the outro. Disabled for now
# (requested) — set True (or env BOOST_INTRO / BOOST_OUTRO=1) to re-enable.
ADD_INTRO = os.environ.get("BOOST_INTRO", "0") == "1"
ADD_OUTRO = os.environ.get("BOOST_OUTRO", "0") == "1"

# ---------------------------------------------------------------------------
# Audio: "Podcast Voice"-style processing. Disabled for now (requested) — the
# original audio is passed through untouched.
# ---------------------------------------------------------------------------
PROCESS_AUDIO = os.environ.get("BOOST_PROCESS_AUDIO", "0") == "1"

# ---------------------------------------------------------------------------
# Auto-cut dead time: removes silent gaps (scroll-throughs where the presenter
# says nothing, and long pauses). Speech is kept with a little padding so it
# never sounds clipped. Tunable; only gaps longer than MIN_SILENCE are cut.
# ---------------------------------------------------------------------------
AUTOCUT = os.environ.get("BOOST_AUTOCUT", "1") == "1"
AUTOCUT_NOISE_DB = -30      # below this level counts as "silence"
AUTOCUT_MIN_SILENCE = 1.5   # seconds — only cut dead gaps longer than this
AUTOCUT_PAD = 0.30          # seconds of silence kept on each side of speech
# Drop "kept" islands shorter than this (isolated clicks/throat-noises in dead
# zones, and the leading sliver of head silence). Real speech is rarely a <0.4s
# island flanked by silence on both sides. NOTE: this is a safe band-aid; the
# proper fix for fragmented start dead-zones is VAD (planned).
AUTOCUT_MIN_KEEP = 0.40

# Cleanvoice content-cleanup pass (fillers / coughs / mouth sounds / stutters).
# Sends audio to the Cleanvoice API; needs CLEANVOICE_API_KEY (.env). Combined
# with the local VAD cut in a single render. Degrades gracefully (logs + skips)
# if the key is missing or the API errors. NOTE: this uploads the client's audio
# to a third party — confirm that's acceptable per project.
CLEANVOICE = os.environ.get("BOOST_CLEANVOICE", "1") == "1"

# Repeated-line / false-start detection (stage 3): Whisper transcript + Gemini.
# Produces CUT CANDIDATES for human review (not auto-applied). Whisper model:
# tiny/base/small/medium/large-v3 — bigger = more accurate, slower.
# distil-large-v3: ~5-9x faster than large-v3 on CPU at ~the same English
# accuracy (faster-whisper/CTranslate2 has no Apple-GPU support, so this runs on
# CPU). Override with BOOST_WHISPER_MODEL=large-v3 if you want max accuracy.
WHISPER_MODEL = os.environ.get("BOOST_WHISPER_MODEL", "distil-large-v3")
GEMINI_MODEL = os.environ.get("BOOST_GEMINI_MODEL", "gemini-2.5-flash")
# Repeat detection is text-similarity based (deterministic), not LLM. A line is
# flagged when it is at least this similar (0..1) to a nearby line. Lower =
# catches looser/paraphrased repeats but more false positives; higher = only
# near-verbatim. A human confirms every cut, so lean toward catching more.
REPEAT_SIMILARITY = float(os.environ.get("BOOST_REPEAT_SIMILARITY", "0.65"))

# ---------------------------------------------------------------------------
# Asset file paths
# ---------------------------------------------------------------------------
@dataclass
class Assets:
    white_bg: str = field(default_factory=lambda: asset("assets", "White Background.jpg"))
    robot: str = field(default_factory=lambda: asset("assets", "pose1-smile.png"))
    outro: str = field(default_factory=lambda: asset("assets", "Outro Fade with logo.mov"))
    # SC Intro.mogrt is Premiere-only; we render a title card instead (see edit.py)
    intro_logo: str = field(default_factory=lambda: asset("assets", "Studyclix_robot.png"))

    def missing(self) -> list[str]:
        out = []
        for name in ("white_bg", "robot", "outro"):
            p = getattr(self, name)
            if not os.path.exists(p):
                out.append(p)
        return out


ASSETS = Assets()
