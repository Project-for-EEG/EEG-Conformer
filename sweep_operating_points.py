"""
Sweep the false-alarm levers against cached model outputs.

Sensitivity is solved (0.867 of 166 events). False alarms are not: the pipeline
targets 2 FA/hr and delivers ~9.7. Rolling calibration failed, which rules out
drift -- the threshold is mis-set from the start, not going stale.

Three levers remain, none of which touch the model:

  target      the overshoot factor has been stable near 5x across every run, so
              asking for 0.4 may actually deliver 2.
  min_consec  a detection needs 3 consecutive windows (6 s) today. Seizures run
              15-100 s, so demanding 8-10 should cut isolated false alarms while
              costing only the shortest events.
  operating   at 0.867 there is room to trade sensitivity for specificity, and
  point       0.75 at 1 FA/hr beats 0.87 at 10 for any real use.

Everything below is still honest: thresholds come from each patient's own
adaptation background, never from the test period.

Usage:
    python sweep_operating_points.py
    python sweep_operating_points.py --per-patient   # find the outliers
"""
import argparse
from pathlib import Path

import numpy as np

from evaluate_end_to_end import score
from calibrate_per_patient import pick_threshold

CACHE = Path("endtoend_probs.npz")


def evaluate(data, target_fa, k, m):
    """Recalibrate every patient at target_fa, score, pool."""
    ev = hit = 0
    fa_w = hours = 0.0
    per_patient = {}
    for pat, d in data.items():
        t = pick_threshold(d["bg"], target_fa, k, m)
        s = score(d["blocks"], t, k, m)
        ev += s["events"]
        hit += s["caught"]
        fa_w += s["fa_hr"] * s["hours"]
        hours += s["hours"]
        per_patient[pat] = s
    return {"sens": hit / ev if ev else 0.0, "caught": hit, "events": ev,
            "fa_hr": fa_w / hours if hours else 0.0,
            "per_patient": per_patient}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smooth", type=int, default=5)
    ap.add_argument("--per-patient", action="store_true")
    args = ap.parse_args()
    k = args.smooth

    if not CACHE.exists():
        print(f"{CACHE} not found -- run evaluate_end_to_end.py first.")
        return
    data = np.load(CACHE, allow_pickle=True)["data"].item()

    targets = [0.1, 0.25, 0.5, 1.0, 2.0]
    consecs = [3, 5, 8, 12]

    print("sensitivity / ACHIEVED FA-per-hour, by requested budget and "
          "minimum duration\n")
    header = "  ".join(f"{'m=' + str(m):>14s}" for m in consecs)
    print(f"{'target':>7s}  {header}")
    grid = {}
    for t in targets:
        cells = []
        for m in consecs:
            r = evaluate(data, t, k, m)
            grid[(t, m)] = r
            cells.append(f"{r['sens']:.3f} @{r['fa_hr']:6.2f}")
        print(f"{t:7.2f}  " + "  ".join(f"{c:>14s}" for c in cells))

    print()
    best = None
    for (t, m), r in grid.items():
        if r["fa_hr"] <= 1.0 and (best is None or r["sens"] > best[1]["sens"]):
            best = ((t, m), r)
    if best:
        (t, m), r = best
        print(f"BEST under 1 FA/hr : {r['sens']:.3f} sensitivity "
              f"({r['caught']}/{r['events']}) at {r['fa_hr']:.2f} FA/hr "
              f"[target {t}, min_consec {m}]")
    else:
        print("Nothing reaches 1 FA/hr.")

    best2 = None
    for (t, m), r in grid.items():
        if r["fa_hr"] <= 2.0 and (best2 is None or r["sens"] > best2[1]["sens"]):
            best2 = ((t, m), r)
    if best2:
        (t, m), r = best2
        print(f"BEST under 2 FA/hr : {r['sens']:.3f} sensitivity "
              f"({r['caught']}/{r['events']}) at {r['fa_hr']:.2f} FA/hr "
              f"[target {t}, min_consec {m}]")

    if args.per_patient and best2:
        (t, m), r = best2
        print(f"\nper-patient at that point (target {t}, min_consec {m}) --"
              f" worst false-alarm rates first\n")
        rows = sorted(r["per_patient"].items(),
                      key=lambda kv: -kv[1]["fa_hr"])
        print(f"  {'patient':8s} {'ev':>4s} {'sens':>6s} {'FA/hr':>8s} {'h':>6s}")
        for pat, s in rows:
            print(f"  {pat:8s} {s['events']:4d} {s['sens']:6.2f} "
                  f"{s['fa_hr']:8.2f} {s['hours']:6.1f}")
        fa_sorted = [s["fa_hr"] * s["hours"] for _, s in rows]
        total = sum(fa_sorted)
        print(f"\n  top 3 patients account for "
              f"{100*sum(fa_sorted[:3])/total:.0f}% of all false alarms")


if __name__ == "__main__":
    main()
