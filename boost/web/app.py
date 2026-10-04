"""
FastAPI web front-end for the Boost editing pipeline.

Upload a raw Loom mp4 + a title -> the job is queued, processed by the pipeline,
and the finished 1920x1080 video can be downloaded.  A single background worker
processes jobs sequentially (video editing is CPU-heavy; one at a time is right
for a test deployment).

Run with:
    uvicorn boost.web.app:app --reload --port 8000
"""
from __future__ import annotations

import glob
import os
import re
import shutil
import sys
import tempfile
import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import json

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

import subprocess

from .. import config as C
from .. import repeats as repeats_mod
from ..editor import render_edit
from ..pipeline import run_pipeline


def _gpu_available() -> bool:
    try:
        out = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"],
                             capture_output=True, text=True).stdout
        return C.GPU_VCODEC in out
    except Exception:  # noqa: BLE001
        return False


_GPU_OK = _gpu_available()

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(HERE, "static")
# Persistent job storage. NOT the system temp dir — macOS periodically purges
# /var/folders/.../T, which was silently deleting finished videos. Defaults to
# ~/boost_jobs; override with BOOST_DATA_DIR.
DATA = os.environ.get("BOOST_DATA_DIR") or os.path.join(
    os.path.expanduser("~"), "boost_jobs")
os.makedirs(DATA, exist_ok=True)

app = FastAPI(title="Boost Video Editor")


# Simple access key for remote (tunnel) use: with BOOST_ACCESS_KEY set, every
# request must carry the key — once via ?key=..., then a cookie keeps the
# session. Without the env var the app stays open (local-only use).
ACCESS_KEY = os.environ.get("BOOST_ACCESS_KEY", "")


@app.middleware("http")
async def _no_cache_api(request, call_next):
    """Never let the browser cache API responses. /api/jobs had no cache headers,
    so a browser could serve a stale (or momentarily-empty, during a restart) job
    list and the real jobs would appear 'missing' even after a reload."""
    if ACCESS_KEY:
        supplied = (request.query_params.get("key")
                    or request.cookies.get("boost_key"))
        if supplied != ACCESS_KEY:
            from fastapi.responses import PlainTextResponse
            return PlainTextResponse("Forbidden", status_code=403)
    resp = await call_next(request)
    if ACCESS_KEY and request.query_params.get("key") == ACCESS_KEY:
        resp.set_cookie("boost_key", ACCESS_KEY, max_age=90 * 24 * 3600,
                        httponly=True, samesite="lax")
    if request.url.path.startswith("/api/"):
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
        resp.headers["Pragma"] = "no-cache"
    return resp


# ONE worker: each video is processed on all CPU cores, jobs run one-by-one.
_executor = ThreadPoolExecutor(max_workers=1)
_lock = threading.Lock()


@dataclass
class Job:
    id: str
    title: str
    kind: str = "render"          # render | edit
    status: str = "queued"        # queued | running | done | error
    progress: float = 0.0
    message: str = "Queued…"
    output: Optional[str] = None
    error: Optional[str] = None
    bubble_detected: bool = False
    repeats: Optional[list] = None      # repeated-line candidates (kind="repeats")
    detect_repeats: bool = False        # run repeat detection during the render
    drive_parent: Optional[str] = None  # Drive folder of the source (for upload-back)
    uploaded: bool = False              # has been uploaded to Drive 'final' folder
    superseded: bool = False            # a re-render/edit replaced this job —
    #                                     do NOT upload this version to Drive
    note: str = ""                      # human note (e.g. which job replaced it)


JOBS: Dict[str, Job] = {}
JOB_ORDER: list[str] = []         # submission order, for queue position
REGISTRY = os.path.join(DATA, "registry.json")


def _save_registry() -> None:
    """Persist finished render/edit jobs so the list survives a server restart.

    MERGES with the on-disk registry instead of overwriting it: rows for jobs this
    process doesn't have in JOB_ORDER are kept as long as their output file still
    exists. This makes it safe against a partial JOB_ORDER (or a stray second
    server instance) silently wiping everyone else's jobs."""
    try:
        merged = {}
        if os.path.exists(REGISTRY):
            try:
                with open(REGISTRY, encoding="utf-8") as f:
                    for row in json.load(f):
                        merged[row["id"]] = row
            except Exception:  # noqa: BLE001
                merged = {}
        for jid in JOB_ORDER:
            j = JOBS.get(jid)
            if not j or j.kind not in ("render", "edit"):
                continue
            merged[j.id] = {"id": j.id, "title": j.title, "status": j.status,
                            "kind": j.kind,
                            "output": j.output, "bubble_detected": j.bubble_detected,
                            "drive_parent": j.drive_parent,
                            "detect_repeats": j.detect_repeats,
                            "uploaded": j.uploaded,
                            "superseded": j.superseded, "note": j.note}
            # Also drop a sidecar INSIDE the job dir. The registry is a single
            # shared file: delete a job (or lose the row any other way) and the
            # Drive link is gone, so a restored job comes back with no upload
            # target and relink-by-title cannot find it (the recovered title is
            # the output filename, not the Drive name). The sidecar travels with
            # the folder, so restoring the folder restores the link.
            try:
                jd = os.path.join(DATA, j.id)
                if os.path.isdir(jd):
                    with open(os.path.join(jd, "job.json"), "w",
                              encoding="utf-8") as jf:
                        json.dump({"title": j.title, "kind": j.kind,
                                   "drive_parent": j.drive_parent,
                                   "detect_repeats": j.detect_repeats}, jf)
            except Exception:  # noqa: BLE001
                pass
        # keep only rows whose output file actually exists on disk. This both
        # drops finished jobs whose file was removed AND stops interrupted/errored
        # rows (no output) from lingering in the registry and reappearing on every
        # restart. Queued/running jobs have no output yet and are re-added from the
        # live JOB_ORDER above each save, so they are not lost by this.
        # Keep finished jobs (their file exists) AND jobs still waiting to
        # render. A restart used to throw the whole queue away, so applying a
        # code fix mid-batch meant re-adding every remaining video by hand.
        rows = []
        for r in merged.values():
            if r.get("output") and os.path.exists(r["output"]):
                rows.append(r)
            elif (r.get("status") in ("queued", "running")
                  and os.path.exists(os.path.join(DATA, r["id"], "input.mp4"))):
                rows.append(r)
        with open(REGISTRY, "w", encoding="utf-8") as f:
            json.dump(rows, f)
    except Exception:  # noqa: BLE001
        pass


def _load_registry() -> None:
    """On startup, restore finished jobs from the registry, then self-heal any
    that the registry lost by scanning the on-disk outputs."""
    rows = []
    if os.path.exists(REGISTRY):
        try:
            with open(REGISTRY, encoding="utf-8") as f:
                rows = json.load(f)
        except Exception:  # noqa: BLE001
            rows = []
    resume = []
    for r in rows:
        out = r.get("output")
        done = r.get("status") == "done" and out and os.path.exists(out)
        inp = os.path.join(DATA, r["id"], "input.mp4")
        pending = (not done and r.get("status") in ("queued", "running")
                   and os.path.exists(inp))
        if pending:
            JOBS[r["id"]] = Job(
                id=r["id"], title=r.get("title", "lesson"),
                kind=r.get("kind", "render"), status="queued", progress=0.0,
                message="Queued… (resumed after restart)",
                bubble_detected=r.get("bubble_detected", False),
                drive_parent=r.get("drive_parent"),
                detect_repeats=r.get("detect_repeats", False),
            )
            JOB_ORDER.append(r["id"])
            resume.append((r["id"], inp))
            continue
        JOBS[r["id"]] = Job(
            id=r["id"], title=r.get("title", "lesson"),
            kind=r.get("kind", "render"),
            status="done" if done else "error",
            progress=1.0 if done else 0.0,
            message="Done." if done else "Interrupted (server restarted).",
            output=out if done else None,
            error=None if done else "interrupted",
            bubble_detected=r.get("bubble_detected", False),
            drive_parent=r.get("drive_parent"),
            detect_repeats=r.get("detect_repeats", False),
            uploaded=r.get("uploaded", False),
            superseded=r.get("superseded", False), note=r.get("note", ""),
        )
        JOB_ORDER.append(r["id"])
    _recover_from_disk()
    # Hand the unfinished queue back to the worker pool, in submission order.
    # A job caught mid-render restarts, but reuses its cleaned audio if that
    # stage had already finished — Cleanvoice is paid per minute.
    for jid, inp in resume:
        cached = os.path.join(DATA, jid, "cleaned.mp4")
        _executor.submit(_process, jid, inp,
                         cached if os.path.exists(cached) else None)
    if resume:
        print(f"[resume] re-queued {len(resume)} unfinished render(s)",
              file=sys.stderr, flush=True)


def _recover_from_disk() -> None:
    """Self-heal: add any finished job whose output .mp4 is still on disk but is
    missing from the registry (e.g. registry got truncated / a crash). Makes the
    on-disk files the source of truth so nothing silently disappears from the
    list. Recovered jobs lose their Drive link (re-link to restore upload)."""
    try:
        for jd in sorted(glob.glob(os.path.join(DATA, "*"))):
            jid = os.path.basename(jd)
            if jid in JOBS or not os.path.isdir(jd):
                continue
            outs = [o for o in glob.glob(os.path.join(jd, "out", "*.mp4"))
                    if not o.endswith(".orig.mp4")]
            if not outs:
                continue
            out = max(outs, key=os.path.getmtime)
            title = os.path.splitext(os.path.basename(out))[0]
            meta = {}
            try:                                  # sidecar written by _save_registry
                with open(os.path.join(jd, "job.json"), encoding="utf-8") as mf:
                    meta = json.load(mf)
            except Exception:  # noqa: BLE001
                meta = {}
            JOBS[jid] = Job(id=jid, title=meta.get("title") or title,
                            kind=meta.get("kind") or "render", status="done",
                            progress=1.0, message="Done.", output=out,
                            bubble_detected=True,
                            drive_parent=meta.get("drive_parent"),
                            detect_repeats=bool(meta.get("detect_repeats")))
            JOB_ORDER.append(jid)
    except Exception:  # noqa: BLE001
        pass


def _queue_position(job_id: str) -> int:
    """How many jobs are ahead of this one still waiting/running (0 = next/now)."""
    pos = 0
    for jid in JOB_ORDER:
        if jid == job_id:
            break
        j = JOBS.get(jid)
        if j and j.status in ("queued", "running"):
            pos += 1
    return pos


def _process(job_id: str, src_path: str,
             precleaned: Optional[str] = None) -> None:
    job = JOBS[job_id]
    job.status = "running"
    out_dir = os.path.join(DATA, job_id, "out")

    def progress(msg: str, frac: float) -> None:
        job.message = msg
        job.progress = frac

    try:
        res = run_pipeline(src_path, job.title, out_dir, progress=progress,
                           detect_repeats=job.detect_repeats,
                           precleaned=precleaned,
                           save_cleaned=os.path.join(DATA, job_id, "cleaned.mp4"))
        job.output = res.output
        job.bubble_detected = res.bubble_detected
        if res.repeats is not None:
            job.repeats = res.repeats
            _save_repeats(job_id, res.repeats, res.transcript)   # persist to disk
        job.status = "done"
        warns = list(res.warnings or [])
        if res.repeats_error:
            warns.append(f"Repeat detection failed: {res.repeats_error}")
        job.message = "Done." if not warns else "⚠ " + " ".join(warns)
        job.progress = 1.0
        # Automated QA on every server render — findings land in the job row
        # so nothing ships silently (client kept catching what we never saw).
        try:
            job.message += " · QA…"
            from .. import qa as _qa
            _rep = _qa.check(res.output,
                             src=os.path.join(DATA, job_id, "cleaned.mp4"))
            _vis = [f for f in _rep["findings"] if f["kind"] != "audio-dip"]
            with open(os.path.join(DATA, job_id, "qa.json"), "w") as _fh:
                json.dump(_rep, _fh)
            if _vis:
                _lst = "; ".join(
                    f"{f['kind']}@{int(f['t'] // 60)}:{f['t'] % 60:04.1f}"
                    if f.get("t") is not None else f["kind"]
                    for f in _vis[:6])
                job.message = (job.message.replace(" · QA…", "")
                               + f" · ⚠ QA: {_lst}")
            else:
                job.message = job.message.replace(" · QA…", " · QA: clean")
        except Exception as _exc:  # noqa: BLE001 — QA must never fail a render
            job.message = job.message.replace(" · QA…", f" · QA error: {_exc}")
        # Auto-upload only if explicitly enabled (hands-off batches). By default
        # the user edits first, then uploads the final version with the button.
        if os.environ.get("BOOST_DRIVE_AUTOUPLOAD") == "1":
            _maybe_upload_to_drive(job)
    except Exception as exc:  # noqa: BLE001
        job.status = "error"
        job.error = f"{exc}"
        job.message = "Failed."
        traceback.print_exc()
    _save_registry()


def _maybe_upload_to_drive(job: "Job") -> None:
    """Upload the finished video into the source folder's 'final' subfolder
    (needs a service account with write access). No-op otherwise."""
    if not job.drive_parent or not job.output:
        return
    try:
        from .. import drive
        if not drive.has_write():
            job.message += " · (Drive upload off: not authorized — run authorize_drive.py)"
            return
        job.message = "Uploading to Google Drive…"
        fid = drive.find_subfolder(job.drive_parent, ["final", "david final"],
                                   drive.key())
        if not fid:
            job.message = "Done · ⚠ no 'final' subfolder found to upload to."
            return
        name = os.path.basename(job.output)
        dup = drive.find_in_folder(fid, name)
        if dup:
            # a stalled connection can land the file yet never return; a retry
            # would then add a duplicate. Replace the old copy instead.
            try:
                drive._service(fresh=True).files().delete(
                    fileId=dup, supportsAllDrives=True).execute()
            except Exception:  # noqa: BLE001
                pass
        job._uploading = True
        job.message = "Uploading to Google Drive… 0%"
        drive.upload(job.output, fid, name,
                     progress=lambda p: setattr(
                         job, "message",
                         f"Uploading to Google Drive… {int(p * 100)}%"))
        job.uploaded = True
        job.message = "Done · ✅ uploaded to Drive 'final' folder."
        _save_registry()
    except Exception as exc:  # noqa: BLE001
        job.message = f"Done · ⚠ Drive upload failed: {exc}"
        traceback.print_exc()
    finally:
        job._uploading = False


@app.post("/api/jobs")
async def create_job(file: UploadFile = File(...),
                     title: str = Form(""),
                     detect_repeats: str = Form("")) -> dict:
    # title is optional now (no intro); default to the uploaded file name
    title = title.strip() or os.path.splitext(file.filename or "")[0] or "lesson"
    want_repeats = str(detect_repeats).lower() in ("1", "true", "on", "yes")
    job_id = uuid.uuid4().hex[:12]
    job_dir = os.path.join(DATA, job_id)
    os.makedirs(job_dir, exist_ok=True)
    src_path = os.path.join(job_dir, "input.mp4")
    with open(src_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    with _lock:
        JOBS[job_id] = Job(id=job_id, title=title.strip(), kind="render",
                           detect_repeats=want_repeats)
        JOB_ORDER.append(job_id)
    _save_registry()
    _executor.submit(_process, job_id, src_path)
    return {"id": job_id}


# ---------------------------------------------------------------------------
# Add jobs from Google Drive links (a file, or a whole folder = a batch).
# The file/folder must be shared "Anyone with the link can view".
# ---------------------------------------------------------------------------
_VIDEO_EXT = (".mp4", ".mov", ".m4v", ".mkv", ".webm")


def _download_drive_file(url: str, job_dir: str) -> tuple:
    import gdown
    got = gdown.download(url, output=job_dir + os.sep, quiet=True, fuzzy=True)
    if not got or not os.path.exists(got) or os.path.getsize(got) < 10000:
        raise RuntimeError("download failed — share the file as "
                           "'Anyone with the link can view'")
    title = os.path.splitext(os.path.basename(got))[0]
    src = os.path.join(job_dir, "input.mp4")
    if os.path.abspath(got) != os.path.abspath(src):
        shutil.move(got, src)
    return src, title


def _process_url(job_id: str, url: str) -> None:
    from .. import drive
    job = JOBS[job_id]
    job.status = "running"
    job.message = "Downloading from Google Drive…"
    job_dir = os.path.join(DATA, job_id)
    src = os.path.join(job_dir, "input.mp4")
    try:
        k = drive.key()
        fid = drive.file_id(url)
        if k and fid:                                    # reliable Drive API path
            drive.download(fid, src, k)
        else:                                            # gdown fallback
            src, title = _download_drive_file(url, job_dir)
            if title:
                job.title = title
        if not os.path.exists(src) or os.path.getsize(src) < 10000:
            raise RuntimeError("download failed — check the link/sharing")
        _save_registry()
    except Exception as exc:  # noqa: BLE001
        job.status = "error"
        job.error = f"{exc}"
        job.message = f"Download failed: {exc}"
        traceback.print_exc()
        _save_registry()
        return
    _process(job_id, src)


def _fetch_folder(fjob_id: str, url: str, detect_repeats: bool) -> None:
    """Find every video in a shared Drive folder (recursively, incl. the batch's
    10 subfolders) and queue each as a render. Uses the Drive API when a key is
    set (reliable, account-agnostic for public folders); else falls back to gdown.
    `fjob_id` is a visible status row so the user sees progress / errors."""
    from .. import drive
    fjob = JOBS[fjob_id]
    k = drive.key()
    fid = drive.folder_id(url)
    if not (k and fid):
        _fetch_folder_gdown(fjob, url, detect_repeats)
        return
    try:
        fjob.message = "Listing folder + subfolders via Drive API…"
        vids = drive.find_videos(fid, k)
        if not vids:
            fjob.status = "error"
            fjob.message = ("No videos found. Check the folder is shared "
                            "“Anyone with the link can view” and has .mp4 files.")
            return
        fjob.message = f"Found {len(vids)} video(s) — downloading…"
        n = 0
        for v in vids:
            job_id = uuid.uuid4().hex[:12]
            jd = os.path.join(DATA, job_id)
            os.makedirs(jd, exist_ok=True)
            src = os.path.join(jd, "input.mp4")
            # Job title = the human-readable lesson name in (brackets) of the
            # containing Drive folder — the raw file names are unreadable
            # ("Microsoft_ Edge - CH-T3-S… - 27 July 2026"). Greedy match keeps
            # nested brackets intact ("Hess's Law (& conservation of energy)").
            title = os.path.splitext(os.path.basename(v["name"]))[0]
            pname = v.get("parent_name") or ""
            m = re.search(r"\((.*)\)", pname)
            if m and m.group(1).strip():
                title = m.group(1).strip()
            elif pname.strip():
                title = pname.strip()
            with _lock:
                JOBS[job_id] = Job(id=job_id, title=title, kind="render",
                                   status="queued", message="Downloading from Drive…",
                                   detect_repeats=detect_repeats,
                                   drive_parent=v.get("parent"))
                JOB_ORDER.append(job_id)
            try:
                drive.download(v["id"], src, k)
            except Exception as exc:  # noqa: BLE001
                JOBS[job_id].status = "error"
                JOBS[job_id].message = f"Download failed: {exc}"
                continue
            JOBS[job_id].message = "Queued…"
            _executor.submit(_process, job_id, src)
            n += 1
        _save_registry()
        fjob.status = "done"
        fjob.progress = 1.0
        fjob.message = f"Queued {n} video(s) from the folder."
    except Exception as exc:  # noqa: BLE001
        fjob.status = "error"
        fjob.error = f"{exc}"
        fjob.message = f"Folder fetch failed: {exc}"
        traceback.print_exc()


def _fetch_folder_gdown(fjob, url: str, detect_repeats: bool) -> None:
    import gdown
    tmp = tempfile.mkdtemp(prefix="boost_dl_")
    try:
        fjob.message = "Listing folder (gdown)…"
        files = gdown.download_folder(url=url, output=tmp, quiet=False,
                                      use_cookies=False, remaining_ok=True) or []
        vids = [f for f in files if f and f.lower().endswith(_VIDEO_EXT)]
        if not vids:
            fjob.status = "error"
            fjob.message = ("No videos found (set DRIVE_API_KEY for reliable "
                            "folder access, or share as 'Anyone with the link').")
            return
        n = 0
        for f in vids:
            job_id = uuid.uuid4().hex[:12]
            jd = os.path.join(DATA, job_id)
            os.makedirs(jd, exist_ok=True)
            src = os.path.join(jd, "input.mp4")
            shutil.move(f, src)
            title = os.path.splitext(os.path.basename(f))[0]
            with _lock:
                JOBS[job_id] = Job(id=job_id, title=title, kind="render",
                                   detect_repeats=detect_repeats)
                JOB_ORDER.append(job_id)
            _executor.submit(_process, job_id, src)
            n += 1
        _save_registry()
        fjob.status = "done"
        fjob.progress = 1.0
        fjob.message = f"Queued {n} video(s) from the folder."
    except Exception as exc:  # noqa: BLE001
        fjob.status = "error"
        fjob.error = f"{exc}"
        fjob.message = f"Folder download failed: {exc}"
        traceback.print_exc()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _norm_title(s: str) -> str:
    import re
    s = (s or "").lower()
    for suf in ("(taskbar fixed)", "(re-render)", "(edited)"):
        s = s.replace(suf, "")
    return re.sub(r"[^a-z0-9]", "", s)


@app.post("/api/jobs/relink")
async def relink_drive(request: Request) -> dict:
    """Re-associate already-rendered jobs with their Google Drive parent folder by
    matching titles against the videos in a shared folder — restores the "Upload
    to Drive" target WITHOUT re-downloading or re-rendering anything."""
    body = await request.json()
    url = (body.get("url") or "").strip()
    if not url:
        raise HTTPException(400, "missing folder url")
    from .. import drive
    k = drive.key()
    fid = drive.folder_id(url)
    if not (k and fid):
        raise HTTPException(400, "need a valid folder link and DRIVE_API_KEY set")
    vids = drive.find_videos(fid, k)
    # Match on the FOLDER name as well as the file name. Job titles are built
    # from the lesson folder's parenthesised part ("FRJC-... (Pets & Animals)"
    # -> "Pets & Animals"), so matching only the video filename missed every
    # job imported from a batch — relink reported 0 links while every title
    # was sitting right there in the folder names (2026-10-02).
    tmap = {}
    for v in vids:
        if not v.get("parent"):
            continue
        keys = [os.path.splitext(os.path.basename(v["name"]))[0]]
        pname = v.get("parent_name") or ""
        if pname:
            keys.append(pname)
            m = re.search(r"\((.*)\)", pname)
            if m and m.group(1).strip():
                keys.append(m.group(1).strip())
        for t in keys:
            tmap.setdefault(_norm_title(t), v["parent"])
    linked = []
    with _lock:
        for jid in JOB_ORDER:
            j = JOBS.get(jid)
            if not j or j.kind != "render":
                continue
            parent = tmap.get(_norm_title(j.title))
            if parent:
                j.drive_parent = parent
                linked.append(j.title)
    _save_registry()
    return {"linked": len(linked), "titles": linked, "videos_in_folder": len(vids)}


@app.post("/api/jobs/url")
async def create_job_url(request: Request) -> dict:
    body = await request.json()
    url = (body.get("url") or "").strip()
    want_repeats = bool(body.get("detect_repeats"))
    if not url:
        raise HTTPException(400, "missing url")
    if "/folders/" in url or "/drive/folders" in url:   # a whole batch folder
        fjob_id = uuid.uuid4().hex[:12]
        with _lock:
            JOBS[fjob_id] = Job(id=fjob_id, title="📁 Drive folder", kind="folder",
                                status="running", message="Fetching folder…",
                                detect_repeats=want_repeats)
            JOB_ORDER.append(fjob_id)
        threading.Thread(target=_fetch_folder,
                         args=(fjob_id, url, want_repeats), daemon=True).start()
        return {"id": fjob_id, "folder": True}
    job_id = uuid.uuid4().hex[:12]
    os.makedirs(os.path.join(DATA, job_id), exist_ok=True)
    with _lock:
        JOBS[job_id] = Job(id=job_id, title="Downloading…", kind="render",
                           detect_repeats=want_repeats)
        JOB_ORDER.append(job_id)
    _save_registry()
    _executor.submit(_process_url, job_id, url)
    return {"id": job_id}


@app.get("/api/jobs")
async def list_jobs() -> list:
    """All render jobs in submission order — lets the UI restore the list on
    refresh (the server is the source of truth)."""
    out = []
    for jid in JOB_ORDER:
        j = JOBS.get(jid)
        if not j or j.kind not in ("render", "folder", "edit"):
            continue
        out.append({
            "id": j.id, "title": j.title, "status": j.status,
            "kind": j.kind,
            "progress": round(j.progress, 3),
            "message": j.message,
            "queue_position": _queue_position(jid) if j.status == "queued" else 0,
            "download": f"/api/jobs/{j.id}/download" if (j.status == "done" and j.output) else None,
            "drive": bool(j.drive_parent),
            "uploaded": j.uploaded,
            "superseded": j.superseded, "note": j.note,
        })
    return out


@app.delete("/api/jobs/{job_id}")
async def delete_job(job_id: str) -> dict:
    with _lock:
        job = JOBS.pop(job_id, None)
        if job_id in JOB_ORDER:
            JOB_ORDER.remove(job_id)
    _save_registry()
    # Soft-delete: MOVE the job dir to .trash instead of destroying it, so an
    # accidental delete is recoverable (move it back + restart). Skipped for a
    # running job (don't yank an active render). .trash is hidden from the
    # disk-recovery scan (glob "*" ignores dot-dirs).
    if job and job.status != "running":
        src = os.path.join(DATA, job_id)
        if os.path.isdir(src):
            trash = os.path.join(DATA, ".trash")
            os.makedirs(trash, exist_ok=True)
            dst = os.path.join(trash, job_id)
            shutil.rmtree(dst, ignore_errors=True)
            try:
                os.replace(src, dst)
            except OSError:
                shutil.move(src, dst)
    return {"ok": True}


@app.get("/api/jobs/{job_id}")
async def job_status(job_id: str) -> dict:
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    position = _queue_position(job_id) if job.status == "queued" else 0
    return {
        "id": job.id,
        "title": job.title,
        "status": job.status,
        "progress": round(job.progress, 3),
        "message": job.message,
        "error": job.error,
        "bubble_detected": job.bubble_detected,
        "queue_position": position,
        "download": f"/api/jobs/{job.id}/download" if (job.status == "done" and job.output) else None,
        "drive": bool(job.drive_parent),
        "uploaded": job.uploaded,
        "superseded": job.superseded, "note": job.note,
    }


@app.get("/api/jobs/{job_id}/download")
async def download(job_id: str):
    job = JOBS.get(job_id)
    if not job or job.status != "done" or not job.output:
        raise HTTPException(404, "not ready")
    fname = os.path.basename(job.output)
    return FileResponse(job.output, media_type="video/mp4", filename=fname)


@app.get("/api/jobs/{job_id}/stream")
async def stream(job_id: str):
    """Inline (range-enabled) playback for the in-browser editor."""
    job = JOBS.get(job_id)
    if not job or job.status != "done" or not job.output:
        raise HTTPException(404, "not ready")
    return FileResponse(job.output, media_type="video/mp4")


def _process_edit(edit_id: str, src: str, spec: dict, assets: Dict[str, str]) -> None:
    job = JOBS[edit_id]
    job.status = "running"
    out_dir = os.path.join(DATA, edit_id, "out")
    os.makedirs(out_dir, exist_ok=True)
    safe = "".join(c for c in job.title if c.isalnum() or c in " -_").strip() or "edited"
    out = os.path.join(out_dir, f"{safe}.mp4")

    def progress(msg: str, frac: float) -> None:
        job.message, job.progress = msg, frac

    try:
        render_edit(src, spec, assets, out, progress=progress)
        job.output = out
        job.status = "done"
        job.message = "Done."
        job.progress = 1.0
    except Exception as exc:  # noqa: BLE001
        job.status = "error"
        job.error = f"{exc}"
        job.message = "Failed."
        traceback.print_exc()


@app.post("/api/jobs/{job_id}/edit")
async def edit_job(job_id: str, request: Request) -> dict:
    src_job = JOBS.get(job_id)
    if not src_job or src_job.status != "done" or not src_job.output:
        raise HTTPException(404, "source job not ready")

    form = await request.form()
    if "spec" not in form:
        raise HTTPException(400, "missing spec")
    spec = json.loads(form["spec"])

    # Remember which proposed cuts the editor accepted and which they kept —
    # the only reliable signal for telling a real stumble from a teaching
    # restatement (see boost/feedback.py).
    try:
        from .. import feedback as _fb
        fb = list(spec.get("repeat_feedback") or [])
        # A MANUAL cut is the strongest lesson available: the editor removed
        # something the detector never proposed. Turn each one into a positive
        # example by pulling the words spoken in that span from the saved
        # transcript, so the same phrasing gets proposed next time.
        man = spec.get("manual_cuts") or []
        if man:
            saved = _load_repeats(job_id) or {}
            tr = saved.get("transcript") or []
            for mc in man:
                a0, b0 = float(mc.get("start", 0)), float(mc.get("end", 0))
                if b0 - a0 < 0.4:
                    continue
                said = " ".join(
                    s.get("text", "") for s in tr
                    if s.get("end", 0) > a0 and s.get("start", 0) < b0).strip()
                if said:
                    fb.append({"text": said[:400], "type": "manual",
                               "confidence": 1.0, "approved": True})
        if fb:
            _fb.record(fb)
    except Exception:  # noqa: BLE001
        pass

    # Resolve inserted clips (the UI sends job ids) to rendered files. Done
    # here, never client-side: the browser must not be able to name a path.
    ins_ok = []
    for ins in (spec.get("inserts") or []):
        src_id = str(ins.get("job") or "")
        at = float(ins.get("at") or 0)
        cj = JOBS.get(src_id)
        if src_id == job_id:
            raise HTTPException(400, "cannot insert a video into itself")
        if not cj or cj.status != "done" or not cj.output or not os.path.exists(cj.output):
            raise HTTPException(400, f"insert clip {src_id} is not a finished render")
        ins_ok.append({"src": cj.output, "at": at, "job": src_id,
                       "title": cj.title})
    spec["inserts"] = ins_ok

    edit_id = uuid.uuid4().hex[:12]
    assets_dir = os.path.join(DATA, edit_id, "assets")
    os.makedirs(assets_dir, exist_ok=True)
    # Persist the edit decisions next to the job. They used to live only in the
    # browser, so a re-render of the SOURCE job silently produced a video
    # without the editor's cuts and there was no way to get the list back
    # (2026-09-10 — recovered only by aligning the edited audio against the
    # original). With this on disk, any re-render can re-apply them.
    try:
        with open(os.path.join(DATA, edit_id, "edit_spec.json"), "w",
                  encoding="utf-8") as f:
            json.dump({"source_job": job_id, "source_output": src_job.output,
                       "title": src_job.title, "spec": spec}, f, indent=1)
    except Exception:  # noqa: BLE001
        pass
    assets: Dict[str, str] = {}
    for key, val in form.multi_items():
        if key == "spec":
            continue
        if hasattr(val, "filename") and getattr(val, "filename", None):
            path = os.path.join(assets_dir, f"{key}_{val.filename}")
            with open(path, "wb") as f:
                shutil.copyfileobj(val.file, f)
            assets[key] = path

    # An edit can name itself. Default stays "<source> (edited)", but a
    # scripted edit (e.g. splicing in a replacement clip) needs a title that
    # stands out in a list where several lessons share a name.
    edit_title = (str(spec.get("title") or "").strip()
                  or f"{src_job.title} (edited)")[:160]
    with _lock:
        JOBS[edit_id] = Job(id=edit_id, title=edit_title, kind="edit",
                            drive_parent=src_job.drive_parent)   # keep Drive target
        JOB_ORDER.append(edit_id)
        src_job.superseded = True
        src_job.note = "Sent to edit — do not upload this version."
        for ins in ins_ok:                 # a clip spliced in must not ship alone
            cj = JOBS.get(ins["job"])
            if cj:
                cj.superseded = True
                cj.note = f"Inserted into \"{edit_title}\" — do not upload separately."
    _save_registry()
    _executor.submit(_process_edit, edit_id, src_job.output, spec, assets)
    return {"id": edit_id}


def _process_rerender(job_id: str, src_path: str, precleaned: Optional[str]) -> None:
    job = JOBS[job_id]
    job.status = "running"
    out_dir = os.path.join(DATA, job_id, "out")

    def progress(msg: str, frac: float) -> None:
        job.message, job.progress = msg, frac

    try:
        res = run_pipeline(src_path, job.title, out_dir, progress=progress,
                           detect_repeats=False, precleaned=precleaned,
                           save_cleaned=os.path.join(DATA, job_id, "cleaned.mp4"))
        job.output = res.output
        job.bubble_detected = res.bubble_detected
        job.status = "done"
        reused = " · reused cleaned audio (no Cleanvoice)" if precleaned else ""
        job.message = f"Done (visual re-render){reused}."
        job.progress = 1.0
    except Exception as exc:  # noqa: BLE001
        job.status = "error"
        job.error = f"{exc}"
        job.message = "Failed."
        traceback.print_exc()
    _save_registry()


@app.post("/api/jobs/{job_id}/rerender")
async def rerender_job(job_id: str) -> dict:
    """Re-render a finished job through the current pipeline (e.g. to pick up a
    visual fix like taskbar removal) while REUSING the cached cleaned audio /
    timeline from the first render — so Cleanvoice is not called again and no
    credits are spent. Falls back to a full clean if no cache exists yet."""
    src_job = JOBS.get(job_id)
    if not src_job or src_job.kind != "render":
        raise HTTPException(404, "source render job not found")
    src_path = os.path.join(DATA, job_id, "input.mp4")
    if not os.path.exists(src_path):
        raise HTTPException(400, "original input.mp4 no longer on disk — cannot re-render")

    cached = os.path.join(DATA, job_id, "cleaned.mp4")
    precleaned = cached if os.path.exists(cached) else None

    new_id = uuid.uuid4().hex[:12]
    new_dir = os.path.join(DATA, new_id)
    os.makedirs(new_dir, exist_ok=True)
    # Hardlink (instant, no copy) the big inputs into the new job so it can also
    # be re-rendered again later; fall back to a copy across filesystems.
    def _link(srcf: str, dstf: str) -> None:
        try:
            os.link(srcf, dstf)
        except OSError:
            shutil.copyfile(srcf, dstf)
    _link(src_path, os.path.join(new_dir, "input.mp4"))
    new_precleaned = None
    if precleaned:
        new_precleaned = os.path.join(new_dir, "cleaned.mp4")
        _link(precleaned, new_precleaned)

    with _lock:
        JOBS[new_id] = Job(id=new_id, title=f"{src_job.title} (re-render)",
                           kind="render", drive_parent=src_job.drive_parent)
        JOB_ORDER.append(new_id)
        src_job.superseded = True
        src_job.note = "Sent to re-render — do not upload this version."
    _save_registry()
    _executor.submit(_process_rerender, new_id,
                     os.path.join(new_dir, "input.mp4"), new_precleaned)
    return {"id": new_id, "reused_cleaned": bool(precleaned)}


def _process_merge(job_id: str, parts: List[str], titles: List[str]) -> None:
    job = JOBS[job_id]
    job.status = "running"
    out_dir = os.path.join(DATA, job_id, "out")
    os.makedirs(out_dir, exist_ok=True)
    safe = "".join(c for c in job.title if c.isalnum() or c in " -_").strip() or "merged"
    out = os.path.join(out_dir, f"{safe}.mp4")

    def progress(msg: str, frac: float) -> None:
        job.message, job.progress = msg, frac

    try:
        from .. import merge as _merge
        res = _merge.merge_videos(parts, out, progress=progress)
        job.output = out
        job.status = "done"
        how = "stream copy" if res["copied"] else "re-encoded"
        job.message = (f"Done · merged {len(parts)} parts ({how}, "
                       f"{res['duration'] / 60:.1f} min)")
        job.progress = 1.0
        try:                                   # same QA gate as every render
            from .. import qa as _qa
            _rep = _qa.check(out)
            with open(os.path.join(DATA, job_id, "qa.json"), "w") as fh:
                json.dump(_rep, fh)
            _vis = [f for f in _rep["findings"] if f["kind"] != "audio-dip"]
            job.message += (" · QA: clean" if not _vis else " · ⚠ QA: " + "; ".join(
                f"{f['kind']}@{int(f['t'] // 60)}:{f['t'] % 60:04.1f}"
                if f.get("t") is not None else f["kind"] for f in _vis[:6]))
        except Exception as exc:  # noqa: BLE001
            job.message += f" · QA error: {exc}"
    except Exception as exc:  # noqa: BLE001
        job.status = "error"
        job.error = f"{exc}"
        job.message = "Failed."
        traceback.print_exc()
    _save_registry()


@app.post("/api/jobs/merge")
async def merge_jobs(request: Request) -> dict:
    """Join two or more finished jobs into one video, in the given order.
    For lessons recorded in two takes that the client wants as a single file
    (2026-09-21, "House: Part Two"). The parts are left in place but marked
    superseded so they are not uploaded by mistake; the merged job inherits
    the first part's Drive folder."""
    body = await request.json()
    ids = [str(i) for i in (body.get("ids") or [])]
    if len(ids) < 2:
        raise HTTPException(400, "pick at least two finished jobs")
    parts, titles, srcs = [], [], []
    for jid in ids:
        j = JOBS.get(jid)
        if not j or j.status != "done" or not j.output or not os.path.exists(j.output):
            raise HTTPException(400, f"job {jid} is not a finished render")
        parts.append(j.output)
        titles.append(j.title)
        srcs.append(j)
    title = (body.get("title") or "").strip() or f"{titles[0]} (merged)"
    new_id = uuid.uuid4().hex[:12]
    os.makedirs(os.path.join(DATA, new_id), exist_ok=True)
    with open(os.path.join(DATA, new_id, "merge_spec.json"), "w",
              encoding="utf-8") as f:
        json.dump({"parts": ids, "titles": titles, "outputs": parts,
                   "title": title}, f, indent=1)
    with _lock:
        JOBS[new_id] = Job(id=new_id, title=title, kind="render",
                           drive_parent=srcs[0].drive_parent)
        JOB_ORDER.append(new_id)
        for j in srcs:
            j.superseded = True
            j.note = f"Merged into \"{title}\" — upload the merged version."
    _save_registry()
    _executor.submit(_process_merge, new_id, parts, titles)
    return {"id": new_id, "title": title}


@app.post("/api/jobs/{job_id}/upload")
async def upload_job(job_id: str) -> dict:
    """Upload this job's finished video to its Drive folder's 'final' subfolder."""
    job = JOBS.get(job_id)
    if not job or job.status != "done" or not job.output:
        raise HTTPException(404, "job not ready")
    if not job.drive_parent:
        raise HTTPException(400, "this video isn't linked to a Drive folder")
    if getattr(job, "_uploading", False):
        return {"ok": True, "already": True}      # a second click would run a
        # parallel upload of the same file — duplicates and TLS corruption
    threading.Thread(target=_maybe_upload_to_drive, args=(job,), daemon=True).start()
    return {"ok": True}


def _repeats_path(job_id: str) -> str:
    return os.path.join(DATA, job_id, "repeats.json")


def _save_repeats(job_id: str, candidates: list, transcript: list = None) -> None:
    """Persist candidates + slim transcript next to the job so they survive a
    restart and re-search needs no re-transcription."""
    try:
        with open(_repeats_path(job_id), "w", encoding="utf-8") as f:
            json.dump({"candidates": candidates or [],
                       "transcript": transcript or []}, f)
    except Exception:  # noqa: BLE001
        pass


def _load_repeats(job_id: str) -> Optional[dict]:
    p = _repeats_path(job_id)
    if os.path.exists(p):
        try:
            with open(p, encoding="utf-8") as f:
                return json.load(f)
        except Exception:  # noqa: BLE001
            return None
    return None


def _process_repeats(task_id: str, src: str, src_job_id: str) -> None:
    job = JOBS[task_id]
    job.status = "running"

    def progress(msg: str, frac: float) -> None:
        job.message, job.progress = msg, frac

    try:
        saved = _load_repeats(src_job_id)
        tr = saved.get("transcript") if saved else None
        has_words = tr and any(s.get("words") for s in tr)
        if has_words:
            # re-search the saved word-level transcript (fast — no transcription)
            progress("Searching saved transcript for repeats…", 0.3)
            cands = repeats_mod.find_in_segments(tr)
            transcript = tr
        else:
            # no saved transcript, or an old one without word timestamps → (re)transcribe
            progress(f"Transcribing ({C.WHISPER_MODEL}) + analysing repeats…", 0.1)
            cands, transcript = repeats_mod.analyze(src)
        job.repeats = cands
        _save_repeats(src_job_id, cands, transcript)   # cache for next open
        if src_job_id in JOBS:
            JOBS[src_job_id].repeats = cands
        job.status = "done"
        job.progress = 1.0
        job.message = f"{len(cands)} repeated-line candidate(s)"
    except Exception as exc:  # noqa: BLE001
        job.status = "error"
        job.error = f"{exc}"
        job.message = "Repeat detection failed."
        traceback.print_exc()


@app.post("/api/jobs/{job_id}/repeats")
async def find_repeats_ep(job_id: str) -> dict:
    src_job = JOBS.get(job_id)
    if not src_job or src_job.status != "done" or not src_job.output:
        raise HTTPException(404, "source job not ready")
    task_id = uuid.uuid4().hex[:12]
    with _lock:
        JOBS[task_id] = Job(id=task_id, title=f"{src_job.title} (repeats)",
                            kind="repeats")
        JOB_ORDER.append(task_id)
    _executor.submit(_process_repeats, task_id, src_job.output, job_id)
    return {"id": task_id}


@app.get("/api/repeats/{task_id}")
async def repeats_status(task_id: str) -> dict:
    job = JOBS.get(task_id)
    # candidates: in-memory first, else load from disk (survives restarts)
    cands = job.repeats if (job and job.repeats is not None) else None
    if cands is None:
        saved = _load_repeats(task_id)
        if saved is not None:
            cands = saved.get("candidates", [])
    if job is None and cands is None:
        raise HTTPException(404, "task not found")
    return {
        "status": job.status if job else "done",
        "progress": round(job.progress, 3) if job else 1.0,
        "message": job.message if job else f"{len(cands or [])} candidate(s)",
        "error": job.error if job else None,
        "candidates": cands or [],
    }


@app.get("/api/cleanvoice")
async def get_cleanvoice() -> dict:
    return {"enabled": C.CLEANVOICE}


@app.post("/api/cleanvoice")
async def set_cleanvoice(request: Request) -> dict:
    body = await request.json()
    C.CLEANVOICE = bool(body.get("enabled"))
    return {"enabled": C.CLEANVOICE}


@app.get("/api/encoder")
async def get_encoder() -> dict:
    return {"encoder": C.ENCODER, "gpu_available": _GPU_OK}


@app.post("/api/encoder")
async def set_encoder(request: Request) -> dict:
    body = await request.json()
    mode = str(body.get("encoder", "")).lower()
    if mode not in ("cpu", "gpu"):
        raise HTTPException(400, "encoder must be 'cpu' or 'gpu'")
    if mode == "gpu" and not _GPU_OK:
        raise HTTPException(400, "GPU (NVENC) not available on this machine")
    C.ENCODER = mode
    return {"encoder": C.ENCODER}


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    with open(os.path.join(STATIC, "index.html"), encoding="utf-8") as f:
        return HTMLResponse(f.read(), headers={"Cache-Control": "no-store"})


app.mount("/static", StaticFiles(directory=STATIC), name="static")

_load_registry()   # restore previously finished jobs on startup
