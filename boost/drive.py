"""
Google Drive read access via the official Drive API (v3) with a plain API key.

Public files/folders ("Anyone with the link can view") are readable by ANY API
key — it doesn't matter which Google account owns the files or the key. This is
far more reliable than scraping (gdown) for listing a batch folder's subfolders
and pulling the raw video out of each.

Only READ is needed (list + download). Uploading results back would need write
access (a service account shared into the destination), which is separate.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.parse
import urllib.request

API = "https://www.googleapis.com/drive/v3"
VIDEO_EXT = (".mp4", ".mov", ".m4v", ".mkv", ".webm")
_FOLDER_MIME = "application/vnd.google-apps.folder"


def key() -> "str | None":
    """DRIVE_API_KEY from env/.env (falls back to GEMINI_API_KEY — same project
    works if Drive API is enabled there)."""
    if not os.environ.get("DRIVE_API_KEY"):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env = os.path.join(root, ".env")
        if os.path.exists(env):
            for line in open(env, encoding="utf-8"):
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())
    return os.environ.get("DRIVE_API_KEY") or os.environ.get("GEMINI_API_KEY")


def folder_id(url: str) -> "str | None":
    m = re.search(r"/folders/([A-Za-z0-9_-]+)", url) or \
        re.search(r"[?&]id=([A-Za-z0-9_-]+)", url)
    return m.group(1) if m else None


def file_id(url: str) -> "str | None":
    m = re.search(r"/file/d/([A-Za-z0-9_-]+)", url) or \
        re.search(r"[?&]id=([A-Za-z0-9_-]+)", url)
    return m.group(1) if m else None


def _get(params: dict, k: str) -> dict:
    url = f"{API}/files?{urllib.parse.urlencode({**params, 'key': k})}"
    last = None
    for attempt in range(4):                    # flaky links / SSL handshake
        try:                                    # timeouts killed whole batches
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.loads(r.read())
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2 ** attempt)
    raise last


def list_children(fid: str, k: str) -> list:
    """All non-trashed children of a folder: [{id, name, mimeType}]."""
    out, token = [], None
    while True:
        params = {
            "q": f"'{fid}' in parents and trashed=false",
            "fields": "nextPageToken,files(id,name,mimeType)",
            "pageSize": 1000,
            "supportsAllDrives": "true",
            "includeItemsFromAllDrives": "true",
        }
        if token:
            params["pageToken"] = token
        data = _get(params, k)
        out += data.get("files", [])
        token = data.get("nextPageToken")
        if not token:
            break
    return out


def find_videos(fid: str, k: str, _depth: int = 0, max_depth: int = 4,
                _pname: str = "") -> list:
    """Recursively collect video files: [{id, name, parent, parent_name}] where
    `parent` is the folder directly containing the video (used to find its
    'final' subfolder) and `parent_name` its display name (used for job
    titles — the folder names carry the human-readable lesson title)."""
    vids = []
    for c in list_children(fid, k):
        if c.get("mimeType") == _FOLDER_MIME:
            if _depth < max_depth:
                vids += find_videos(c["id"], k, _depth + 1, max_depth,
                                    _pname=c.get("name", ""))
        elif (c.get("name", "").lower().endswith(VIDEO_EXT)
              or c.get("mimeType", "").startswith("video/")):
            vids.append({"id": c["id"], "name": c["name"], "parent": fid,
                         "parent_name": _pname})
    return vids


def find_subfolder(parent_id: str, names: list, k: str) -> "str | None":
    """Id of a subfolder of `parent_id` whose name matches one of `names`
    (case-insensitive), or the first one containing 'final'."""
    wanted = [n.lower() for n in names]
    subs = [c for c in list_children(parent_id, k)
            if c.get("mimeType") == _FOLDER_MIME]
    for c in subs:
        if c.get("name", "").strip().lower() in wanted:
            return c["id"]
    for c in subs:                                   # looser: contains "final"
        if "final" in c.get("name", "").lower():
            return c["id"]
    return None


def download(file_id_: str, dst: str, k: "str | None" = None) -> str:
    """Download a Drive file. Prefers OAuth (the authorized user), because the
    anonymous API key can only fetch files shared "Anyone with the link" and 403s
    on everything else — folders shared only with the user list fine but their
    files refuse to download. Falls back to the API key for public files."""
    if has_oauth():
        global _svc
        last = None
        for attempt in range(3):                # retry with a fresh connection
            try:                                # — SSL blips killed downloads
                from googleapiclient.http import MediaIoBaseDownload
                req = _service().files().get_media(fileId=file_id_,
                                                   supportsAllDrives=True)
                with open(dst, "wb") as f:
                    dl = MediaIoBaseDownload(f, req, chunksize=8 << 20)
                    done = False
                    while not done:
                        _, done = dl.next_chunk(num_retries=5)
                return dst
            except Exception as e:  # noqa: BLE001
                last = e
                _svc = None                     # rebuild the TLS session
                time.sleep(2 ** attempt)
        if not k:
            raise last
    url = (f"{API}/files/{urllib.parse.quote(file_id_)}"
           f"?alt=media&key={k}&supportsAllDrives=true")
    with urllib.request.urlopen(url, timeout=1800) as r, open(dst, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    return dst


# ---------------------------------------------------------------------------
# Write access (upload the finished video). Two options:
#   • OAuth (token.json): uploads AS A REAL USER (you) — works for normal "My
#     Drive" folders. PREFERRED. Create it once with authorize_drive.py.
#   • Service account (service_account.json): NO storage quota of its own, so it
#     can only write into SHARED DRIVES, not a user's My Drive. Fallback.
# ---------------------------------------------------------------------------
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_svc = None
_SCOPES = ["https://www.googleapis.com/auth/drive"]


def _path(env: str, default_name: str) -> "str | None":
    p = os.environ.get(env)
    if p and os.path.exists(p):
        return p
    d = os.path.join(_ROOT, default_name)
    return d if os.path.exists(d) else None


def token_path() -> "str | None":
    return _path("DRIVE_OAUTH_TOKEN", "token.json")


def service_account_path() -> "str | None":
    return _path("SERVICE_ACCOUNT_JSON", "service_account.json")


def has_oauth() -> bool:
    return token_path() is not None


def has_service_account() -> bool:
    return service_account_path() is not None


def has_write() -> bool:
    return has_oauth() or has_service_account()


def service_account_email() -> "str | None":
    p = service_account_path()
    if not p:
        return None
    try:
        return json.load(open(p)).get("client_email")
    except Exception:  # noqa: BLE001
        return None


def _authed_http(creds, timeout: int = 120):
    """Authorized HTTP transport WITH a socket timeout. Without it a stalled
    Drive connection blocks the upload thread forever: the file lands on
    Drive but the job sits at "Uploading to Google Drive…" for good (client
    saw this as uploads that never finish)."""
    import httplib2
    import google_auth_httplib2
    h = httplib2.Http(timeout=timeout)
    # Resumable uploads answer 308 "Resume Incomplete" WITHOUT a Location
    # header; httplib2 treats 308 as a redirect and dies with "Redirected but
    # the response is missing a Location: header". Google's own build_http()
    # drops 308 from the redirect set for exactly this reason.
    if hasattr(h, "redirect_codes"):
        h.redirect_codes = set(h.redirect_codes) - {308}
    return google_auth_httplib2.AuthorizedHttp(creds, http=h)



def _service(fresh: bool = False):
    """Cached Drive client. `fresh=True` builds a SEPARATE client with its own
    TLS connection: httplib2 is NOT thread-safe, and two uploads sharing one
    connection corrupt the stream ("[SSL: DECRYPTION_FAILED_OR_BAD_RECORD_MAC]"
    — seen when a UI upload ran alongside a scripted one)."""
    global _svc
    if fresh:
        return _build_service()
    if _svc is not None:
        return _svc
    _svc = _build_service()
    return _svc


def _build_service():
    from googleapiclient.discovery import build
    tp = token_path()
    if tp:                                        # OAuth as a real user (preferred)
        from google.oauth2.credentials import Credentials
        from google.auth.transport.requests import Request
        creds = Credentials.from_authorized_user_file(tp, _SCOPES)
        if not creds.valid and creds.refresh_token:
            creds.refresh(Request())
            with open(tp, "w") as f:
                f.write(creds.to_json())
        return build("drive", "v3", http=_authed_http(creds),
                     cache_discovery=False)
    elif service_account_path():                  # service account (shared drives)
        from google.oauth2 import service_account
        creds = service_account.Credentials.from_service_account_file(
            service_account_path(), scopes=_SCOPES)
        return build("drive", "v3", http=_authed_http(creds),
                     cache_discovery=False)
    raise RuntimeError("no Drive write credentials (token.json / service_account.json)")


def find_in_folder(folder_id: str, name: str) -> "str | None":
    """Id of a file called `name` already sitting in `folder_id` (so a retry
    after a stalled connection replaces it instead of adding a duplicate)."""
    try:
        q = (f"'{folder_id}' in parents and name = "
             f"'{name.replace(chr(39), chr(92) + chr(39))}' and trashed=false")
        res = _service(fresh=True).files().list(
            q=q, fields="files(id,size)", pageSize=5,
            supportsAllDrives=True, includeItemsFromAllDrives=True).execute()
        fs = res.get("files", [])
        return fs[0]["id"] if fs else None
    except Exception:  # noqa: BLE001
        return None


def upload(file_path: str, folder_id: str, name: str,
           progress=None) -> str:
    """Upload a local file into a Drive folder. Returns the new file id.

    Uploads in explicit resumable CHUNKS rather than one .execute(): a single-shot
    upload of a multi-hundred-MB file cannot recover from a mid-transfer network
    blip — the client retries the write on the same TLS socket and dies with
    "[SSL: BAD_WRITE_RETRY] bad write retry". Each chunk is its own request (so a
    blip only costs that chunk), and on hard failure the cached service is dropped
    so the next attempt builds a fresh connection instead of reusing a broken one.
    """
    from googleapiclient.http import MediaFileUpload
    global _svc
    last = None
    for attempt in range(3):
        try:
            media = MediaFileUpload(file_path, mimetype="video/mp4",
                                    chunksize=8 << 20, resumable=True)
            req = _service(fresh=True).files().create(
                body={"name": name, "parents": [folder_id]},
                media_body=media, fields="id", supportsAllDrives=True)
            resp = None
            while resp is None:
                status, resp = req.next_chunk(num_retries=5)
                if progress and status:
                    progress(status.progress())
            return resp.get("id")
        except Exception as e:  # noqa: BLE001 — retry with a fresh connection
            last = e
            _svc = None
            time.sleep(2 ** attempt)
    raise last
