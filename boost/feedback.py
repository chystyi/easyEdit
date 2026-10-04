"""
Learning from the editor's own cut decisions.

Separating a genuine stumble from a teaching restatement is not solvable from
the transcript (Whisper writes French speech as phonetic English) nor from the
audio (the teacher speaks English with French phrases embedded, so every span
detects as English). The signal that DOES exist is the editor: they approve
real mistakes and reject pedagogical repeats, video after video, and the same
phrases recur across a subject.

So we remember every decision and use it to rank future candidates:
  * a candidate that looks like phrases the editor kept rejecting is dropped
  * one that looks like phrases they kept cutting is raised in confidence
Nothing is ever auto-cut — this only reorders and prunes what gets proposed.
"""
from __future__ import annotations

import json
import os
import re
import time

_TOK = re.compile(r"[a-zà-ÿ0-9']+")
STORE = os.environ.get("BOOST_FEEDBACK") or os.path.join(
    os.environ.get("BOOST_DATA_DIR")
    or os.path.join(os.path.expanduser("~"), "boost_jobs"), "feedback.json")

_MAX = 4000                      # keep the store bounded


def _toks(t: str) -> set:
    return {w for w in _TOK.findall((t or "").lower()) if len(w) > 1}


def load() -> list:
    try:
        with open(STORE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return []


def record(entries: list) -> int:
    """Append decisions: [{text, type, confidence, approved}]. Returns total."""
    if not entries:
        return 0
    rows = load()
    now = time.time()
    for e in entries:
        txt = (e.get("text") or "").strip()
        if not txt:
            continue
        rows.append({"text": txt, "type": e.get("type", ""),
                     "confidence": float(e.get("confidence") or 0),
                     "approved": bool(e.get("approved")), "ts": now})
    rows = rows[-_MAX:]
    try:
        os.makedirs(os.path.dirname(STORE), exist_ok=True)
        with open(STORE, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False)
    except Exception:  # noqa: BLE001
        pass
    return len(rows)


def _sim(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def apply(cands: list, *, rows: list = None, thresh: float = 0.6) -> list:
    """Re-rank/prune candidates using past decisions. Never invents new ones."""
    rows = load() if rows is None else rows
    if not rows:
        return cands
    hist = [(_toks(r["text"]), bool(r["approved"])) for r in rows if r.get("text")]
    out = []
    for c in cands:
        ct = _toks(c.get("text", ""))
        sims = [(s, ok) for s, ok in ((_sim(ct, h), ok) for h, ok in hist)
                if s >= thresh]
        if sims:
            appr = sum(1 for s, ok in sims if ok)
            rej = len(sims) - appr
            if rej >= 2 and rej > appr * 2:
                continue                              # editor keeps keeping it
            if appr >= 2 and appr > rej * 2:
                c = {**c, "confidence": round(min(0.99, c["confidence"] + 0.15), 2),
                     "reason": c.get("reason", "") + " · matches cuts you made before"}
        out.append(c)
    return out
