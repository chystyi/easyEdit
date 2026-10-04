"""
Repeated-line / false-start detection (stage 3).

Pipeline: Whisper transcript -> DETERMINISTIC text-similarity search flags
near-identical / repeated lines and clean false starts -> CUT CANDIDATES for
human review. (An LLM was tried here and proved unreliable: it missed obvious
repeats, flagged unrelated lines, and varied run-to-run. Text similarity is
deterministic, instant, free, and only flags real overlap.) Nothing is removed
without explicit approval, so `find_repeats` only proposes; `apply` cuts an
approved subset. The Gemini helpers below are kept but unused by default.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from difflib import SequenceMatcher

from . import config as C
from . import edit, transcribe

PROMPT = """You are an editor cleaning a teacher's lesson recording. Each line is a spoken
segment with an id and time range:

{segments}

The teacher CONSTANTLY restates ideas on purpose — reads a slide then explains it
in their own words, says a definition then rephrases it "so basically…", recaps,
emphasises. THIS IS THE LESSON, NOT AN ERROR. You must NOT flag any of it.

Flag ONLY genuine recording mistakes an editor would physically cut out:
- false_start / restart: the speaker breaks off mid-thought and starts the SAME
  sentence over — stumbles, fumbles a word and repeats it, "sorry", "let me…",
  garbled/duplicated words while navigating or scrolling.
- verbatim_repeat: the SAME words said again almost identically back-to-back
  (e.g. a stutter or a scroll-back), where one copy is plainly an accident.

Hard rules:
- NEVER flag rephrasing, paraphrase, explanation, "so…", "in other words",
  recaps, or saying a term then defining it. Different words for the same idea =
  KEEP. If the segment carries ANY teaching content the next take doesn't, KEEP.
- A short segment that grammatically continues into the next (transcriber split
  one sentence) is NOT a false start.
- Only flag when the words are near-identical or the speaker audibly stumbles.
- Most lessons have 0–3 real cuts. If you are flagging many, you are wrong —
  re-check and keep only blatant stumbles/duplicates. Precision over recall.

Return ONLY JSON:
{{"cuts": [{{"id": <segment id to remove>, "keep_id": <id of the take to keep, or null>,
  "type": "false_start|restart|verbatim_repeat",
  "reason": "<short why this is a mistake, not teaching>", "confidence": <0.0-1.0>}}]}}
"""


def _load_key() -> str:
    if not os.environ.get("GEMINI_API_KEY"):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env = os.path.join(root, ".env")
        if os.path.exists(env):
            for line in open(env, encoding="utf-8"):
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())
    k = os.environ.get("GEMINI_API_KEY")
    if not k:
        raise RuntimeError("GEMINI_API_KEY not set (.env or environment)")
    return k


# Tried in order; if one model is overloaded (503) we fall back to the next.
_FALLBACK_MODELS = ["gemini-2.0-flash", "gemini-flash-latest",
                    "gemini-2.5-flash-lite"]


def _gemini(segments: list, model: str = None) -> list:
    lines = "\n".join(
        f"[{s['id']}] ({s['start']:.1f}-{s['end']:.1f}) {s['text']}"
        for s in segments)
    body = {
        "contents": [{"parts": [{"text": PROMPT.format(segments=lines)}]}],
        "generationConfig": {"response_mime_type": "application/json",
                             "temperature": 0.1},
    }
    payload = json.dumps(body).encode()
    key = _load_key()
    models = [model] if model else [C.GEMINI_MODEL, *_FALLBACK_MODELS]
    seen, ordered = set(), []
    for m in models:                       # dedupe, keep order
        if m not in seen:
            seen.add(m); ordered.append(m)

    last = None
    for m in ordered:
        url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
               f"{m}:generateContent?key={key}")
        for attempt in range(4):           # retry transient errors per model
            try:
                r = urllib.request.Request(
                    url, data=payload,
                    headers={"Content-Type": "application/json"}, method="POST")
                with urllib.request.urlopen(r, timeout=180) as resp:
                    data = json.loads(resp.read())
                text = data["candidates"][0]["content"]["parts"][0]["text"]
                return json.loads(text).get("cuts", [])
            except urllib.error.HTTPError as e:
                last = e
                if e.code in (429, 500, 503) and attempt < 3:
                    time.sleep(1.5 * (attempt + 1))   # 1.5, 3, 4.5s
                    continue
                break                      # persistent error → try next model
            except Exception as e:         # noqa: BLE001
                last = e
                break
    raise last


def slim_segments(segments: list) -> list:
    """Persisted transcript: keep word-level timestamps too — the detector cuts
    at the WORD level (just the repeated phrase, not the whole sentence)."""
    return [{"id": s["id"], "start": s["start"], "end": s["end"],
             "text": s["text"],
             "words": [{"word": w["word"], "start": w["start"], "end": w["end"]}
                       for w in s.get("words", [])]}
            for s in segments]


# ---------------------------------------------------------------------------
# Deterministic repeat detection — compares the actual transcript text.
# An LLM proved unreliable here (missed obvious repeats, flagged unrelated lines,
# varied run-to-run). Text-similarity is deterministic: it WILL catch verbatim /
# near-verbatim repeats and false starts, and only flags real textual overlap.
# A human still confirms each cut, so intentional repetition is never lost.
# ---------------------------------------------------------------------------
_TOK = re.compile(r"[a-z0-9']+")
_STOP = set("the a an and or but of to in on at for as is are was were be been it "
            "that this these those so you your we our i my they them he she his her "
            "with by from up out if then than there here what which who".split())


def _toks(t: str) -> list:
    return _TOK.findall((t or "").lower())


def _similarity(a: list, b: list) -> float:
    """0..1 repeat score. Order-aware ratio catches verbatim/near-verbatim
    repeats; a long contiguous run that covers most of the shorter line catches
    clean false starts/restarts. Plain word-set overlap is deliberately NOT used
    — it over-scores template sentences ("Push factors are…" vs "Pull factors…")."""
    if not a or not b:
        return 0.0
    ratio = SequenceMatcher(None, a, b).ratio()
    short, lng = (a, b) if len(a) <= len(b) else (b, a)
    m = SequenceMatcher(None, short, lng).find_longest_match(0, len(short),
                                                             0, len(lng))
    block_cov = m.size / len(short)              # biggest shared run vs shorter line
    return max(ratio, block_cov if block_cov >= 0.7 else 0.0)


def _word_stream(segments: list) -> list:
    """Flatten segment words into [(norm, start, end, raw), …]."""
    W = []
    for s in segments:
        for w in s.get("words", []):
            toks = _TOK.findall((w.get("word") or "").lower())
            W.append((toks[0] if toks else "", float(w["start"]),
                      float(w["end"]), (w.get("word") or "").strip()))
    return W


# Self-correction markers: the speaker misspeaks then corrects (often with a
# DIFFERENT value, e.g. "…30 degrees, sorry, 40 degrees") — no repeated phrase,
# so the repeat detector can't see it; the marker word is the signal.
_CORR1 = {"sorry", "oops"}
_CORR2 = {("i", "mean"), ("i", "meant"), ("scratch", "that"), ("no", "wait")}
# Explicit retake announcements — the speaker SAYS they are redoing the take
# ("I've got to do that again"). Strongest possible signal that what came
# just before is a discarded attempt.
_CORR3 = {("do", "that", "again"), ("say", "that", "again"),
          ("start", "that", "again"), ("try", "that", "again"),
          ("do", "it", "again"), ("go", "again"),
          ("one", "more", "time"), ("take", "two")}


# French function words / spellings. A "repeat" whose two takes sit on
# opposite sides of the language line is a TRANSLATION — the teacher says it
# in French then in English — which is the lesson itself, never a mistake.
_FR = {"je", "tu", "il", "elle", "nous", "vous", "ils", "elles", "est", "sont",
       "suis", "es", "et", "le", "la", "les", "un", "une", "des", "du", "de",
       "mon", "ma", "mes", "ton", "ta", "son", "sa", "ses", "ce", "cette",
       "qui", "que", "quoi", "pour", "avec", "dans", "sur", "pas", "ne",
       "plus", "tres", "trs", "beaucoup", "aime", "aimes", "adore", "vais",
       "vas", "va", "allons", "fait", "faire", "avoir", "etre", "cest",
       "jai", "jaime", "sportif", "sportive", "quartier", "argent", "poche",
       "ecole", "semaine", "dernier", "prochain", "week"}


def _fr_score(words: list) -> float:
    """Fraction of French-looking tokens in a run."""
    ws = [w for w in words if w]
    if not ws:
        return 0.0
    return sum(1 for w in ws if w in _FR) / len(ws)


def _is_translation(a: list, b: list, gap_fr: float = 0.25) -> bool:
    """True if one take is markedly more French than the other."""
    return abs(_fr_score(a) - _fr_score(b)) >= gap_fr


def _snap_start(W: list, idx: int, *, min_pause: float = 0.45,
                max_back: float = 5.0) -> int:
    """Walk the cut's start back to the natural pause before it, so the cut
    removes the WHOLE failed take instead of clipping in mid-phrase (the
    editor kept having to extend these by hand)."""
    if idx <= 0:
        return idx
    t_end = W[idx][1]
    k = idx
    while k > 0:
        if W[k][1] - W[k - 1][2] >= min_pause:
            break
        if t_end - W[k - 1][1] > max_back:
            break
        k -= 1
    return k


def _find_corrections(W: list, norm: list, *, lookback: int = 2,
                      long_pause: float = 0.9, retake_lookback: int = 10) -> list:
    """Flag self-corrections: cut the mistaken attempt + the marker ('sorry' /
    'I mean'), keeping the corrected version that follows."""
    n = len(W)
    cands, i = [], 0
    while i < n:
        mk = None
        if norm[i] in _CORR1:
            mk = (i, i)
        elif i + 1 < n and (norm[i], norm[i + 1]) in _CORR2:
            mk = (i, i + 1)
        elif i + 2 < n and (norm[i], norm[i + 1], norm[i + 2]) in _CORR3:
            mk = (i, i + 2)
        elif i + 1 < n and (norm[i], norm[i + 1]) in _CORR3:
            mk = (i, i + 1)
        if not mk:
            i += 1
            continue
        m0, m1 = mk
        is_retake = (m1 - m0 >= 1
                     and tuple(norm[m0:m1 + 1]) in _CORR3)
        back = retake_lookback if is_retake else lookback
        s = m0                                  # walk back over the mistaken attempt
        for k in range(m0 - 1, max(-1, m0 - 1 - back), -1):
            if k < 0 or (W[k + 1][1] - W[k][2]) > long_pause:
                break
            s = k
        cut_txt = " ".join(W[x][3] for x in range(s, m1 + 1)).strip()
        keep_txt = " ".join(W[x][3] for x in range(m1 + 1, min(m1 + 8, n))).strip()
        cands.append({
            "start": round(W[s][1], 2), "end": round(W[m1][2], 2),
            "text": cut_txt, "keep_text": keep_txt, "type": "correction",
            "reason": "self-correction (“sorry / I mean”)", "confidence": 0.7,
        })
        i = m1 + 1
    return cands


def _find_word_level(W: list, norm: list, *, min_run: int = 3, max_run: int = 14,
                     max_gap: int = 4, min_content: int = 2) -> list:
    """Cut just the REPEATED PHRASE, not the whole sentence. Scans the word stream
    for an immediately-repeated run (a stumble/false start where the speaker says
    a phrase, then says it again) and cuts the first occurrence up to where the
    repeat restarts — using word timestamps for a tight cut."""
    n = len(W)
    cands, a = [], 0
    while a < n:
        if not norm[a]:
            a += 1
            continue
        found = None
        for L in range(min(max_run, n - a), min_run - 1, -1):
            first = norm[a:a + L]
            if sum(1 for w in first if w and w not in _STOP) < min_content:
                continue
            for g in range(0, max_gap + 1):       # small stumble gap between takes
                b = a + L + g
                if b + L > n:
                    continue
                if norm[b:b + L] == first:
                    found = (L, g, b)
                    break
            if found:
                break
        if found:
            L, g, b = found
            if _is_translation(norm[a:a + L], norm[b:b + L]):
                a += 1                       # French take vs English take
                continue
            a = _snap_start(W, a)
            cut_txt = " ".join(W[x][3] for x in range(a, b)).strip()
            keep_txt = " ".join(W[x][3] for x in range(b, min(b + L + 5, n))).strip()
            cands.append({
                "start": round(W[a][1], 2), "end": round(W[b][1], 2),
                "text": cut_txt, "keep_text": keep_txt,
                "type": "verbatim_repeat",
                "reason": f"repeated phrase “{' '.join(norm[a:a + L])}”",
                "confidence": round(min(1.0, 0.55 + 0.08 * L), 2),
            })
            a = b                                  # resume from the kept repeat
        else:
            a += 1
    return cands


def _fuzzy_eq(a: list, b: list, max_edits: int = 1) -> bool:
    """Two word runs are 'the same take' if they differ by at most `max_edits`
    words. A real false start is rarely word-perfect ("the Gaza Strip is 41" /
    "the Gaza Strip is 41 kilometres") — exact matching missed those."""
    if abs(len(a) - len(b)) > max_edits:
        return False
    sm = SequenceMatcher(None, a, b)
    edits = sum(max(i2 - i1, j2 - j1)
                for tag, i1, i2, j1, j2 in sm.get_opcodes() if tag != "equal")
    return edits <= max_edits


def _find_restarts(W: list, norm: list, *, head: int = 3, min_pause: float = 0.35,
                   max_attempt: int = 12) -> list:
    """False START: the speaker begins a sentence, breaks off, pauses, then
    begins the SAME sentence again. Signature: a short attempt, a pause, then
    a run whose first `head` content words repeat the attempt's opening.

    This is the pattern the client keeps flagging ("he starts a sentence and
    then restarts it"); the verbatim-run detector misses it because the two
    takes diverge after a few words."""
    n = len(W)
    out, i = [], 0
    while i < n - head * 2:
        if not norm[i] or norm[i] in _STOP:
            i += 1
            continue
        opening = [w for w in norm[i:i + head + 2] if w and w not in _STOP][:head]
        if len(opening) < head:
            i += 1
            continue
        # find the pause that ends this attempt
        j = i + 1
        brk = None
        while j < min(i + max_attempt, n):
            if W[j][1] - W[j - 1][2] >= min_pause:
                brk = j
                break
            j += 1
        if brk is None:
            i += 1
            continue
        nxt = [w for w in norm[brk:brk + head + 4] if w and w not in _STOP][:head]
        if len(nxt) == head and _fuzzy_eq(opening, nxt, max_edits=1):
            if _is_translation(norm[i:brk], norm[brk:brk + (brk - i)]):
                i += 1
                continue
            i = _snap_start(W, i)
            attempt = " ".join(W[x][3] for x in range(i, brk)).strip()
            keep = " ".join(W[x][3] for x in range(brk, min(brk + 12, n))).strip()
            if len(attempt.split()) >= 2:
                out.append({
                    "start": round(W[i][1], 2), "end": round(W[brk][1], 2),
                    "text": attempt, "keep_text": keep, "type": "false_start",
                    "reason": f"restarted after a pause (“{' '.join(opening)}…”)",
                    "confidence": 0.72,
                })
                i = brk
                continue
        i += 1
    return out


def _find_fuzzy_repeats(W: list, norm: list, *, min_run: int = 4,
                        max_run: int = 12, max_gap: int = 5) -> list:
    """Near-verbatim repeated run (one word changed/added between takes)."""
    n = len(W)
    out, a = [], 0
    while a < n:
        if not norm[a] or norm[a] in _STOP:
            a += 1
            continue
        found = None
        for L in range(min(max_run, n - a), min_run - 1, -1):
            first = norm[a:a + L]
            if sum(1 for w in first if w and w not in _STOP) < 3:
                continue
            for g in range(1, max_gap + 1):
                b = a + L + g
                if b + L > n:
                    continue
                if norm[b:b + L] != first and _fuzzy_eq(first, norm[b:b + L], 1):
                    found = (L, b)
                    break
            if found:
                break
        if found:
            L, b = found
            if _is_translation(norm[a:a + L], norm[b:b + L]):
                a += 1
                continue
            a = _snap_start(W, a)
            out.append({
                "start": round(W[a][1], 2), "end": round(W[b][1], 2),
                "text": " ".join(W[x][3] for x in range(a, b)).strip(),
                "keep_text": " ".join(W[x][3] for x in range(b, min(b + L + 5, n))).strip(),
                "type": "near_repeat",
                "reason": f"near-identical retake (“{' '.join(norm[a:a + min(L, 5)])}…”)",
                "confidence": round(min(0.9, 0.5 + 0.06 * L), 2),
            })
            a = b
        else:
            a += 1
    return out


def _find_segment_level(segments: list, *, threshold: float = None) -> list:
    """Fallback when no word timestamps: compare whole-segment text (coarser —
    cuts whole sentences)."""
    threshold = C.REPEAT_SIMILARITY if threshold is None else threshold
    toks = [_toks(s.get("text", "")) for s in segments]
    cands = []
    for i in range(len(segments)):
        if len(toks[i]) < 4:
            continue
        best_j, best_sim = None, 0.0
        for j in range(i + 1, min(i + 7, len(segments))):
            if len(toks[j]) < 4:
                continue
            sim = _similarity(toks[i], toks[j])
            if sim > best_sim:
                best_sim, best_j = sim, j
        if best_j is None or best_sim < threshold:
            continue
        si, sj = segments[i], segments[best_j]
        shorter_is_i = len(toks[i]) <= len(toks[best_j])
        sh, lg = (toks[i], toks[best_j]) if shorter_is_i else (toks[best_j], toks[i])
        mm = SequenceMatcher(None, sh, lg).find_longest_match(0, len(sh), 0, len(lg))
        if mm.a == 0 and mm.size >= len(sh) * 0.8:
            cut, keep = (si, sj) if shorter_is_i else (sj, si)   # cut the fragment
        else:
            cut, keep = si, sj                                   # cut the FIRST (stumble)
        cands.append({
            "start": cut["start"], "end": cut["end"], "text": cut["text"],
            "type": "verbatim_repeat" if best_sim >= 0.8 else "near_repeat",
            "reason": f"~{int(best_sim * 100)}% same wording as another line",
            "confidence": round(best_sim, 2), "keep_text": keep["text"],
        })
    return cands


def find_in_segments(segments: list, *, min_confidence: float = 0.0) -> list:
    """Detect repeats. Word-level (precise, cuts just the phrase) when word
    timestamps are present; otherwise falls back to segment-level."""
    has_words = any(s.get("words") for s in segments)
    if has_words:
        W = _word_stream(segments)
        norm = [x[0] for x in W]
        cands = (_find_word_level(W, norm) + _find_corrections(W, norm)
                 + _find_restarts(W, norm) + _find_fuzzy_repeats(W, norm))
    else:
        cands = _find_segment_level(segments)
    cands = [c for c in cands if c["confidence"] >= min_confidence]
    try:                                        # learn from past cut decisions
        from . import feedback as _fb
        cands = _fb.apply(cands)
    except Exception:  # noqa: BLE001 — never let this break detection
        pass
    cands.sort(key=lambda x: x["start"])
    deduped = []
    for c in cands:
        if deduped and c["start"] < deduped[-1]["end"] - 0.05:
            if c["confidence"] > deduped[-1]["confidence"]:
                deduped[-1] = c
            continue
        deduped.append(c)
    return deduped


def analyze(src: str, *, model_name: str = None):
    """Transcribe `src` then find repeats. Returns (candidates, slim_transcript)
    so both can be persisted next to the video."""
    segments = transcribe.transcribe(src, model_name=model_name)
    return find_in_segments(segments), slim_segments(segments)


def find_repeats(src: str, *, model_name: str = None,
                 min_confidence: float = 0.0) -> list:
    """Transcribe + find repeats, returning just the candidates (CLI helper)."""
    segments = transcribe.transcribe(src, model_name=model_name)
    return find_in_segments(segments, min_confidence=min_confidence)


def write_review(cands: list, path: str) -> None:
    """Write a human-readable review of the cut candidates."""
    with open(path, "w", encoding="utf-8") as f:
        if not cands:
            f.write("No repeated-line candidates found.\n")
            return
        f.write(f"{len(cands)} repeated-line candidate(s) — review before cutting.\n")
        f.write("Set \"approve\": true on the ones to remove.\n\n")
        f.write("[\n")
        for i, c in enumerate(cands):
            f.write(json.dumps({
                "approve": False,
                "start": round(c["start"], 2), "end": round(c["end"], 2),
                "type": c["type"], "confidence": round(c["confidence"], 2),
                "cut_text": c["text"], "keep_text": c["keep_text"],
                "reason": c["reason"],
            }, ensure_ascii=False) + ("," if i < len(cands) - 1 else "") + "\n")
        f.write("]\n")


def apply(src: str, dst: str, approved: list) -> bool:
    """Cut the approved (start, end) ranges from `src`. Returns True if cut."""
    ranges = [(a, b) for a, b in approved if b - a > 0.05]
    if not ranges:
        return False
    dur = edit.probe(src)["duration"]
    keep = edit._ranges_to_keep(ranges, dur)
    if not keep:
        return False
    edit._render_keep(src, dst, keep)
    return True


def _approved_from_review(path: str) -> list:
    """Read (start, end) ranges marked "approve": true in a review JSON file."""
    data = json.load(open(path, encoding="utf-8"))
    return [(e["start"], e["end"]) for e in data if e.get("approve")]


def _main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Repeated-line detector (review + apply)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    pf = sub.add_parser("find", help="detect repeats -> review JSON")
    pf.add_argument("input")
    pf.add_argument("--review", default="repeats_review.json")
    pf.add_argument("--model", default=None, help="whisper model (default: config)")
    pa = sub.add_parser("apply", help="cut approved entries from a review JSON")
    pa.add_argument("input")
    pa.add_argument("--review", required=True)
    pa.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.cmd == "find":
        cands = find_repeats(args.input, model_name=args.model)
        write_review(cands, args.review)
        print(f"{len(cands)} candidate(s) -> {args.review}")
        print('Edit the file, set "approve": true on cuts to make, then: '
              'python -m boost.repeats apply <video> --review '
              f'{args.review} --out result.mp4')
    else:
        approved = _approved_from_review(args.review)
        ok = apply(args.input, args.out, approved)
        print(f"applied {len(approved)} cut(s): {ok} -> {args.out}" if ok
              else "nothing approved / cut")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
