"""
Cleanvoice API client for the content-cleanup pass.

Cleanvoice detects non-lexical and lexical disfluencies (filler words, coughs /
mouth sounds, stutters, hesitations, breaths) that a waveform/VAD pass can't —
they sit at speech level and need an acoustic/transcript model. We send the
audio, get back per-segment timestamps (start, end, type), and decide per type
whether to CUT or leave it.

Flow (REST v2):
    POST /v2/upload?filename=…     -> signed storage URL
    PUT  <signed url>              -> the audio bytes
    POST /v2/edits {input,config}  -> edit id
    GET  /v2/edits/{id}            -> poll until SUCCESS; result has
                                      timestamps_markers_urls.timeline_audacity
                                      (tab-separated: start<TAB>end<TAB>label)

Silence/dead air is handled locally by VAD (boost/speech.py), so long_silences
is left ON here only as a complement — its DEADAIR hits overlap with VAD and are
deduped at apply time.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request

# Cache detected segments by audio content + config so re-cleaning the SAME audio
# (e.g. a visual-only re-render) never re-uploads and re-charges Cleanvoice
# credits. Keyed by md5(audio bytes) + md5(config), stored as JSON.
_CACHE_DIR = os.path.join(tempfile.gettempdir(), "boost_cleanvoice_cache")


def _cache_key(audio_path: str, cfg: dict) -> str:
    h = hashlib.md5()
    with open(audio_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    h.update(json.dumps(cfg, sort_keys=True).encode())
    return h.hexdigest()

API = "https://api.cleanvoice.ai/v2"

# Cleanvoice tags each removable moment; a "MUTE_" prefix means it wants the
# audio silenced in place (breaths), anything else means remove it. We respect
# that: MUTE_* / breaths -> mute; everything else Cleanvoice flags (fillers,
# stutters, hesitations, mouth sounds, dead air — whatever the exact label) ->
# CUT. (Matching an explicit whitelist missed real labels like "FILLER_SOUND"/
# "STUTTERING", so fillers were silently ignored.)
def _action_for(label: str) -> str:
    L = (label or "").upper()
    if L.startswith("MUTE_") or "BREATH" in L:
        return "mute"
    return "cut"

# Features requested from Cleanvoice (long_silences complements local VAD).
CONFIG = {
    "long_silences": True,
    "fillers": True,
    "stutters": True,
    "mouth_sounds": True,
    "hesitations": True,
    "breath": True,
    "remove_noise": False,
    "keep_music": True,
    "normalize": False,
    "video": False,
    "send_email": False,
    "export_timestamps": True,
    "export_format": "mp3",
}


def _load_env() -> None:
    """Populate os.environ from .env if the key isn't already set."""
    if os.environ.get("CLEANVOICE_API_KEY"):
        return
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, ".env")
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def _key() -> str:
    _load_env()
    k = os.environ.get("CLEANVOICE_API_KEY")
    if not k:
        raise RuntimeError("CLEANVOICE_API_KEY not set (.env or environment)")
    return k


def _req(method: str, url: str, data=None, headers=None, raw=False, timeout=180,
         retries=4):
    h = {"X-API-Key": _key()}
    if headers:
        h.update(headers)
    body = None
    if raw:
        body = data
    elif data is not None:
        body = json.dumps(data).encode()
        h["Content-Type"] = "application/json"

    last = None
    for attempt in range(retries):
        try:
            r = urllib.request.Request(url, data=body, headers=h, method=method)
            with urllib.request.urlopen(r, timeout=timeout) as resp:
                c = resp.read()
            try:
                return json.loads(c)
            except json.JSONDecodeError:
                return c
        except urllib.error.HTTPError as e:
            last = e
            if e.code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                time.sleep(2 ** attempt)        # 1, 2, 4s
                continue
            raise
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            # transient DNS / connection / network blip → back off and retry
            last = e
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise
    raise last


def detect(audio_path: str, config: dict = None, poll_s: int = 5,
           max_wait_s: int = 1200) -> list:
    """Upload `audio_path`, run a Cleanvoice edit, and return detected segments as
    a list of dicts: {start, end, type, action}. Times are seconds relative to the
    given audio."""
    cfg = {**CONFIG, **(config or {})}
    fname = os.path.basename(audio_path)

    cache_f = os.path.join(_CACHE_DIR, _cache_key(audio_path, cfg) + ".json")
    if os.path.exists(cache_f):
        try:
            with open(cache_f, encoding="utf-8") as f:
                segs = json.load(f)
            print(f"[cleanvoice] cache hit ({len(segs)} segs) — no API call, "
                  "no credits used", flush=True)
            return segs
        except Exception:  # noqa: BLE001 — corrupt cache → fall through to API
            pass

    signed = _req("POST", f"{API}/upload?filename={urllib.parse.quote(fname)}")["signedUrl"]
    _req("PUT", signed, data=open(audio_path, "rb").read(),
         headers={"Content-Type": "audio/mpeg"}, raw=True)
    eid = _req("POST", f"{API}/edits",
               {"input": {"files": [signed], "config": cfg}})["id"]

    waited = 0
    while waited < max_wait_s:
        r = _req("GET", f"{API}/edits/{eid}")
        if r.get("status") == "SUCCESS":
            break
        if r.get("status") == "FAILURE":
            raise RuntimeError(f"Cleanvoice edit failed: {r}")
        time.sleep(poll_s)
        waited += poll_s
    else:
        raise TimeoutError("Cleanvoice edit timed out")

    tl = r["result"].get("timestamps_markers_urls", {}).get("timeline_audacity")
    if not tl:
        return []
    txt = urllib.request.urlopen(tl, timeout=60).read().decode()

    segs = []
    for line in txt.splitlines():
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        start, end, label = float(parts[0]), float(parts[1]), parts[2].strip()
        base = label.split("_", 1)[1] if label.startswith(("MUTE_", "CUT_")) else label
        segs.append({"start": start, "end": end, "type": base,
                     "label": label, "action": _action_for(label)})
    if segs:
        from collections import Counter
        counts = Counter((s["type"], s["action"]) for s in segs)
        print("[cleanvoice] detected:",
              ", ".join(f"{t}:{a}×{n}" for (t, a), n in sorted(counts.items())),
              flush=True)
    try:
        os.makedirs(_CACHE_DIR, exist_ok=True)
        with open(cache_f, "w", encoding="utf-8") as f:
            json.dump(segs, f)
    except Exception:  # noqa: BLE001 — caching is best-effort
        pass
    return segs


def _merge(segs: list) -> list:
    segs = sorted(segs)
    merged = []
    for s, e in segs:
        if merged and s <= merged[-1][1] + 0.02:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(a, b) for a, b in merged]


def cut_ranges(audio_path: str, config: dict = None) -> list:
    """Just the (start, end) ranges whose action is 'cut', merged & sorted."""
    return _merge([(s["start"], s["end"])
                   for s in detect(audio_path, config) if s["action"] == "cut"])


def cut_and_mute_ranges(audio_path: str, config: dict = None) -> tuple:
    """One API call → (cut_ranges, mute_ranges). Cut = fillers/coughs/dead air
    (removed); mute = breaths (silenced in place)."""
    segs = detect(audio_path, config)
    cut = _merge([(s["start"], s["end"]) for s in segs if s["action"] == "cut"])
    mute = _merge([(s["start"], s["end"]) for s in segs if s["action"] == "mute"])
    return cut, mute
