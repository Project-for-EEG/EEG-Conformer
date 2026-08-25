"""
Re-score cached predictions under SzCORE-style event rules.

Our numbers so far use a strict scorer: any predicted run that does not overlap
a true seizure counts as one false alarm, with no tolerance and no merging. The
2025 seizure-detection challenge (and most recent literature) scores with
SzCORE event rules instead, which are substantially more permissive:

  1. predicted events separated by < 90 s are MERGED into one event
  2. reference events are EXTENDED 30 s before onset and 60 s after offset
     before overlap matching (a near-miss just before/after a seizure is not a
     false alarm)
  3. events longer than 5 min are SPLIT into 5-min pieces
  4. sensitivity = refs with >= 1 overlapping prediction; FA = predicted events
     overlapping no (extended) ref; FA reported per 24 h

Same predictions, same thresholds -- only the ruleset changes. This anchors our
numbers against the challenge (winner: 37% sensitivity at 1.34 FA/day) instead
of leaving them scored under a stricter rule than anyone else uses.

Approximation note: this is a faithful re-implementation of the published rule
set, not the official SzCORE library (which requires 19-ch unipolar EDF input
that CHB-MIT's bipolar recordings cannot provide -- its own CHB-MIT column is
broken for exactly that reason).

Usage:
    python score_szcore.py                          # headline cache
    python score_szcore.py --cache endtoend_natural.npz
"""
import argparse
from pathlib import Path

import numpy as np

from calibrate_per_patient import detect
from evaluate_temporal import find_runs, SECONDS_PER_WINDOW

# rule constants, converted to window units (1 window = 2 s of new data)
MERGE = int(90 / SECONDS_PER_WINDOW)     # 45 windows
PRE = int(30 / SECONDS_PER_WINDOW)       # 15 windows
POST = int(60 / SECONDS_PER_WINDOW)      # 30 windows
SPLIT = int(300 / SECONDS_PER_WINDOW)    # 150 windows


def merge_events(events, gap):
    out = []
    for lo, hi in events:
        if out and lo - out[-1][1] < gap:
            out[-1] = (out[-1][0], hi)
        else:
            out.append((lo, hi))
    return out


def split_events(events, maxlen):
    out = []
    for lo, hi in events:
        while hi - lo > maxlen:
            out.append((lo, lo + maxlen))
            lo += maxlen
        out.append((lo, hi))
    return out


def overlaps(a, b):
    return a[0] < b[1] and b[0] < a[1]


def score_block_szcore(pred, truth):
    """(refs, hit_refs, false_alarms) for one contiguous recording."""
    hyp = split_events(merge_events(find_runs(pred), MERGE), SPLIT)
    refs = split_events(find_runs(truth), SPLIT)
    ext = [(max(0, lo - PRE), hi + POST) for lo, hi in refs]

    hit = sum(1 for r in ext if any(overlaps(r, h) for h in hyp))
    fa = sum(1 for h in hyp if not any(overlaps(r, h) for r in ext))
    return len(refs), hit, fa


def score_block_strict(pred, truth):
    """Our original rule: every non-overlapping predicted run is one FA."""
    refs = find_runs(truth)
    hit = sum(1 for r in refs if pred[r[0]:r[1]].any())
    fa = sum(1 for lo, hi in find_runs(pred) if not truth[lo:hi].any())
    return len(refs), hit, fa


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", type=str, default="endtoend_probs.npz")
    ap.add_argument("--smooth", type=int, default=5)
    ap.add_argument("--min-consec", type=int, default=3)
    args = ap.parse_args()
    K, M = args.smooth, args.min_consec

    data = np.load(args.cache, allow_pickle=True)["data"].item()

    # sanity: do any events exceed the 5-min split length?
    longest = 0
    for d in data.values():
        for _, labels in d["blocks"]:
            for lo, hi in find_runs(labels.astype(bool)):
                longest = max(longest, hi - lo)
    print(f"longest true event: {longest * SECONDS_PER_WINDOW:.0f} s "
          f"(split rule {'ACTIVE' if longest > SPLIT else 'inactive'})\n")

    print(f"{'patient':8s} {'model':>6s} "
          f"{'strict sens':>12s} {'strict FA/h':>12s} "
          f"{'szcore sens':>12s} {'szcore FA/h':>12s}")

    tot = {"s_ref": 0, "s_hit": 0, "s_fa": 0,
           "z_ref": 0, "z_hit": 0, "z_fa": 0, "hours": 0.0}

    for pat in sorted(data):
        d = data[pat]
        s_ref = s_hit = s_fa = z_ref = z_hit = z_fa = 0
        windows = 0
        for probs, labels in d["blocks"]:
            pred = detect(probs, d["t"], K, M)
            truth = labels.astype(bool)
            windows += len(pred)
            r, h, f = score_block_strict(pred, truth)
            s_ref += r; s_hit += h; s_fa += f
            r, h, f = score_block_szcore(pred, truth)
            z_ref += r; z_hit += h; z_fa += f

        hours = windows * SECONDS_PER_WINDOW / 3600
        for k, v in (("s_ref", s_ref), ("s_hit", s_hit), ("s_fa", s_fa),
                     ("z_ref", z_ref), ("z_hit", z_hit), ("z_fa", z_fa),
                     ("hours", hours)):
            tot[k] += v

        if s_ref == 0:
            continue
        print(f"{pat:8s} {d['model']:>6s} "
              f"{s_hit}/{s_ref} {s_hit/s_ref:5.2f} {s_fa/hours:11.2f} "
              f"{z_hit}/{z_ref} {z_hit/z_ref:5.2f} {z_fa/hours:11.2f}")

    h = tot["hours"]
    print(f"\nPOOLED over {h:.0f} h:")
    print(f"  strict (ours) : {tot['s_hit']}/{tot['s_ref']} = "
          f"{tot['s_hit']/tot['s_ref']:.3f} sensitivity at "
          f"{tot['s_fa']/h:.2f} FA/h  ({tot['s_fa']/h*24:.1f} FA/day)")
    print(f"  SzCORE-style  : {tot['z_hit']}/{tot['z_ref']} = "
          f"{tot['z_hit']/tot['z_ref']:.3f} sensitivity at "
          f"{tot['z_fa']/h:.2f} FA/h  ({tot['z_fa']/h*24:.1f} FA/day)")
    print(f"\n2025 challenge winner on private data, same ruleset: "
          f"0.37 sensitivity at 1.34 FA/day")
    print("caveat: our numbers allow 1-seizure-per-patient personalisation "
          "and CHB01/CHB21 are same-subject-tainted until the leak-free "
          "retrain lands")


if __name__ == "__main__":
    main()
