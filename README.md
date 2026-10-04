# Boost Video Editor — automated pipeline

Automates the manual Premiere Pro workflow described in **Boost Editing Guidance.docx**:
turns a raw Loom lesson recording (screen + webcam baked into one file) into a clean,
branded **1920×1080** video — without Premiere, using **ffmpeg + OpenCV**.

## What it does (maps 1:1 to the brief)

| Brief step | Pipeline stage |
|---|---|
| Crop Loom toolbars, scale background | `composite` — background layer scaled 112 %, centred |
| White background between tracks | `composite` — `White Background.jpg` base layer |
| Enlarge webcam bubble (scale 165) | `detect` (auto-locate bubble) + `composite` (crop → 165 % → circular) |
| Mask grey page breaks (rectangles + keyframes) | `clean_grey` — auto-repaints **all** neutral grey to white, per-frame |
| Podcast Voice audio | `composite` — denoise + band-limit + compress + loudnorm |
| Robot overlay (pose1-smile.png) | `composite` — overlay at preset scale/position |
| SC intro with title + Outro fade | `make_intro` (title card) + `assemble` (cross-fades) |
| Export 1920×1080, named by title | final encode |

The geometric constants in `boost/config.py` were **decoded from the two Premiere
presets** (`JC Lesson - Background/Webcam.prfpset.xml`): background scale 112 %,
webcam scale 165 %, output position 0.891 × 0.305, etc.

> The grey-removal pass automatically covers side margins **and** the scrolling
> page breaks, so it replaces the manual "draw a rectangle and keyframe every
> page break" step entirely.

## Install

```bash
pip install -r requirements.txt
# ffmpeg + ffprobe must be on PATH
```

## CLI

```bash
python -m boost.cli "raw.mp4" --title "Lesson Title" --out ./output
python -m boost.cli "raw.mp4" --title "Quick test" --out ./output --clip 20   # first 20s only
python -m boost.cli "raw.mp4" --title "No branding" --out ./output --no-intro --no-outro
```

## Web app

```bash
python -m uvicorn boost.web.app:app --port 8000
# open http://127.0.0.1:8000/  → upload mp4 + title → progress → download
```

**Queue:** one job is processed at a time, using *all* CPU cores. While a video
renders, newly submitted jobs wait and the UI shows their position
("in queue · N ahead"). `queue_position` comes from `/api/jobs/{id}`.

### Visual editor (CapCut-style)

After a render completes, the page opens a visual editor on the result:

- **Timeline** with a ruler, zoom (±), draggable **playhead**, and a video track.
- **Trim** by dragging the dark handles; trimmed spans are dimmed.
- **Cut out** — pick the ✂ tool and drag on the track to mark spans to remove
  (each shows a × to delete it).
- **Text / Image overlays** — added at the playhead, then **dragged and resized
  with the mouse directly on the video**; their coloured **clip** on the timeline
  is dragged/edge-resized to set when and how long they appear. An inspector
  edits text/colour/size (or image width).

The editor sends an edit spec (explicit 1920×1080 pixel coords + asset files) to
`/api/jobs/{id}/edit`; `boost/editor.py` renders overlays first (so their times
match the preview), then applies trim/cuts. Results can be edited again.

## Layout

```
boost/              python package (the pipeline)
  config.py         constants (preset geometry, encoder, intro/outro flags)
  detect.py         webcam-bubble detection + edge-connected background removal
  edit.py           ffmpeg/OpenCV stages: clean_grey, composite, intro, assemble
  segworker.py      per-segment background-removal worker (one process per core)
  editor.py         in-browser edit renderer (trim/cut/text/image/audio)
  pipeline.py       end-to-end orchestration with progress callbacks
  cli.py            command-line entry point
  web/              FastAPI app + single-page visual editor
assets/             pipeline assets (robot, white bg, outro, presets, …)
                    NOT in this repository — the branded intro, outro, robot
                    and backgrounds belong to the client. Supply your own
                    files under the names in config.py to run the pipeline.
samples/            raw sample inputs + SCI_FINAL.mp4 reference + the brief .docx
scripts/            dev tools (run_batch.py)
output/             rendered videos
```

## Notes

- **Intro / outro are OFF by default** (`config.ADD_INTRO` / `ADD_OUTRO`, or env
  `BOOST_INTRO=1` / `BOOST_OUTRO=1`, or CLI `--intro` / `--outro`). The export
  starts straight on the lesson content.
- **Final export** is high-bitrate H.264 (`config.final_codec_args`): libx264
  `crf 17` (CPU) or `h264_nvenc` (GPU), capped at `BOOST_MAXRATE` (default 24M).
  Intermediate/segment encodes stay fast.

## Background removal

`clean_grey` calls `detect.remove_background`, which keys out the viewer canvas
by flood-filling neutral *background-coloured* pixels (light Edge/PDF grey ~229
**and** dark Acrobat-theme ~58) that are connected to the frame border, then
whitening them. The white page stops the flood, so interior text and diagrams —
even dark ones — are preserved. This single pass removes: side margins, scrolling
page breaks, the top toolbar, and dark UI dropdowns (e.g. the "Draw" panel).

## Known limitations / next steps

- **Webcam position** is auto-detected per video (output placement is fixed to
  match the reference); if detection fails it falls back to `WEBCAM_FALLBACK`.
- **Browser pop-ups (white UI)**: transient moments where a teacher opens the
  Edge "…" menu / shows the taskbar are *white* chrome over content and can't be
  keyed out safely. The reference workflow treats teacher fumbles as cut points;
  these would need a manual/auto cut, which is out of scope for "no cutting".
- **Podcast Voice** is an approximation of the Premiere Essential Sound preset.
- **Intro** matches the SC Intro look (teal + topic + "studyclix"); the real
  `SC Intro.mogrt` could be pre-rendered for a frame-exact match.
- **Speed**: the background pass is parallelised across CPU cores
  (`boost/segworker.py`, one process per ~8 s segment).
- **Encoder toggle (CPU/GPU)**: `config.video_codec_args()` returns libx264
  (CPU) or `h264_nvenc` (NVIDIA GPU). Switch at runtime via the UI buttons or
  `POST /api/encoder {"encoder":"gpu"|"cpu"}` (also `BOOST_ENCODER` env default).
  GPU (NVENC) accelerates the main full-length encodes; the segmented background
  pass stays multi-core CPU (its encode is negligible and dozens of parallel
  NVENC sessions would exceed the GPU limit). So GPU mode is a partial speed-up,
  not a full one — the OpenCV background pass is CPU either way.
```
