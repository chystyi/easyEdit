"""
Recover the cut/keep timeline of an already-delivered edited render by aligning
its audio back to the original raw recording — WITHOUT re-running Cleanvoice.

Both files are the same recording; the edited one is the raw with segments cut
(fillers/silences) and some breaths muted. Kept segments are byte-for-byte the
same audio, so a normalised cross-correlation of short windows locks on with
peak ~1.0. We build a frame-accurate edited->raw time map, then read off the
contiguous raw intervals that survived (the "keep ranges").

Used to re-render a video (e.g. to fix a top-crop or restore an over-cut line)
reusing the exact cleaned timeline, so no Cleanvoice credits are spent.
"""
from __future__ import annotations

import subprocess

import numpy as np
from scipy.signal import correlate

FS = 8000


def _decode(path: str) -> np.ndarray:
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-ac", "1", "-ar", str(FS),
         "-f", "f32le", "pipe:1"], capture_output=True).stdout
    return np.frombuffer(out, dtype=np.float32).astype(np.float64)


def _match(ew: np.ndarray, R: np.ndarray, lo: int, hi: int) -> tuple[int, float]:
    """Best normalised-xcorr position of window `ew` inside R[lo:hi]. Returns
    (absolute raw sample index, peak in 0..1)."""
    lo = max(0, lo)
    hi = min(len(R), hi)
    seg = R[lo:hi]
    n = len(ew)
    if len(seg) < n:
        return -1, 0.0
    e = ew - ew.mean()
    es = np.sqrt((e ** 2).sum())
    if es < 1e-6:
        return -1, 0.0
    c = correlate(seg, e, mode="valid", method="fft")
    csum = np.concatenate([[0.0], np.cumsum(seg ** 2)])
    energy = np.sqrt(np.maximum(csum[n:] - csum[:-n], 1e-9))
    cc = c / (energy * es + 1e-9)
    k = int(np.argmax(cc))
    return lo + k, float(cc[k])


def recover_keep_ranges(raw_path: str, edited_path: str, *,
                        win: float = 0.75, hop: float = 0.5,
                        jump_tol: float = 0.12, min_peak: float = 0.80,
                        ) -> list[tuple[float, float]]:
    """Return the list of (start, end) raw-time intervals that the edited render
    kept, in order. Times are seconds in the RAW file."""
    R = _decode(raw_path)
    E = _decode(edited_path)
    W = int(win * FS)
    H = int(hop * FS)

    # dense map: edited sample -> raw sample (offset off = raw - edited).
    # The offset only ever GROWS (cuts remove raw), so we search a tight forward
    # window around the last offset and take the NEAREST strong match. A wide
    # window would let a repeated phrase lock onto a much later occurrence and
    # skip a chunk of the video, so we only widen (with a stricter peak) when the
    # near window truly fails, and never accept an offset that moves backwards.
    anchors: list[tuple[int, int]] = []          # (edited_i, raw_i)
    off = 0                                       # expected raw-edited offset (samples)
    ei = 0
    while ei + W <= len(E):
        ew = E[ei:ei + W]
        if ew.std() > 1e-4:                       # skip muted/silent windows
            exp = ei + off
            # (back, fwd, min_peak): near window first, then a cautious wide one
            for back, fwd, mp in ((0.4, 8.0, min_peak), (1.0, 25.0, 0.93)):
                ri, peak = _match(ew, R, exp - int(back * FS), exp + W + int(fwd * FS))
                if peak >= mp and ri - ei >= off - int(0.3 * FS):   # no backward jump
                    off = max(off, ri - ei)       # monotonic non-decreasing
                    anchors.append((ei, ri))
                    break
        ei += H

    if not anchors:
        return []

    # group anchors into segments of constant offset; a change of offset = a cut.
    tol = jump_tol * FS
    segs: list[dict] = [{"e0": anchors[0][0], "e1": anchors[0][0],
                         "off": anchors[0][1] - anchors[0][0]}]
    for ei, ri in anchors[1:]:
        off_i = ri - ei
        if abs(off_i - segs[-1]["off"]) > tol:     # offset jumped -> new segment
            segs.append({"e0": ei, "e1": ei, "off": off_i})
        else:
            segs[-1]["e1"] = ei

    # refine each cut to the exact edited sample where the offset switches, so the
    # keep ranges tile the edited timeline with no slack (bisection on a short probe
    # window: does E[t] still match the OLD offset, or already the NEW one?).
    sw = int(0.15 * FS)

    def matches(t: int, off: int) -> float:
        w = E[t:t + sw]
        if len(w) < sw or w.std() < 1e-4:
            return -1.0
        _, pk = _match(w, R, t + off - int(0.05 * FS), t + off + sw + int(0.05 * FS))
        return pk

    bounds = [0]                                   # edited-sample split points
    for k in range(len(segs) - 1):
        lo, hi = segs[k]["e1"], segs[k + 1]["e0"]  # cut lies in (lo, hi]
        off_a, off_b = segs[k]["off"], segs[k + 1]["off"]
        while hi - lo > int(0.02 * FS):
            mid = (lo + hi) // 2
            pa, pb = matches(mid, off_a), matches(mid, off_b)
            if pa >= pb:                           # still the old segment
                lo = mid
            else:
                hi = mid
        bounds.append(hi)
    bounds.append(len(E))

    raw_dur = len(R) / FS
    keep: list[tuple[float, float]] = []
    for k, seg in enumerate(segs):
        e_a, e_b = bounds[k], bounds[k + 1]
        rs, re = e_a + seg["off"], e_b + seg["off"]
        s, e = max(0.0, rs / FS), min(raw_dur, re / FS)   # clamp to raw length
        if e - s > 0.05:
            keep.append((s, e))
    return keep


if __name__ == "__main__":
    import sys
    kr = recover_keep_ranges(sys.argv[1], sys.argv[2])
    tot = sum(e - s for s, e in kr)
    print(f"{len(kr)} keep-ranges, total {tot:.1f}s")
    for s, e in kr[:40]:
        print(f"  {s:8.2f} - {e:8.2f}  ({e - s:5.2f}s)")
