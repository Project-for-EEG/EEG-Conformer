"""
Base vs fine-tuned, compared at MATCHED false-alarm rates.

The raw run reports each arm at whatever FA rate its own threshold produced
(11.05 vs 18.12 FA/hr), so part of the apparent sensitivity gain is simply the
tuned model raising more alarms. This re-reads the cached test-period
probabilities and, for each patient, re-thresholds the tuned model to the FA
rate the base model actually achieved on that same patient -- then compares
sensitivity. Any difference left is the model, not the operating point.

Also reports the reverse (base re-thresholded up to the tuned model's FA rate)
so the conclusion cannot hinge on which direction the matching goes.

Requires finetune_probs.npz, written by finetune_per_patient.py.
"""
import argparse
from pathlib import Path

import numpy as np

from calibrate_per_patient import score_blocks

CACHE = Path("finetune_probs.npz")


def sweep(blocks, k, m, n=400):
    """(threshold, sensitivity, FA/hr, flagged fraction) across the range."""
    out = []
    for t in np.linspace(0.0005, 0.9999, n):
        s = score_blocks(blocks, t, k, m)
        out.append((t, s["sens"], s["fa_hr"], s["flagged"]))
    return out


def at_fa(curve, target_fa, max_flagged=0.10):
    """Best sensitivity achievable without exceeding target_fa.

    The flagged-fraction cap is essential, not cosmetic. False alarms are
    counted as alarm *runs*, so a threshold low enough to flag everything
    collapses each recording into a single run that overlaps a real seizure --
    scoring ~0 false alarms at 100% sensitivity while being useless. Without
    this bound the search picks that degenerate point every time.
    """
    ok = [(sens, t, fa) for t, sens, fa, flag in curve
          if fa <= target_fa and flag <= max_flagged]
    if not ok:
        return 0.0, 1.0, 0.0
    sens, t, fa = max(ok)
    return sens, t, fa


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smooth", type=int, default=5)
    ap.add_argument("--min-consec", type=int, default=3)
    ap.add_argument("--max-flagged", type=float, default=0.10,
                    help="reject operating points that flag more than this "
                         "fraction of time (guards against saturation)")
    args = ap.parse_args()
    k, m = args.smooth, args.min_consec

    if not CACHE.exists():
        print(f"{CACHE} not found -- run finetune_per_patient.py first.")
        return

    data = np.load(CACHE, allow_pickle=True)["data"].item()

    print("Tuned model re-thresholded to each patient's BASE false-alarm rate\n")
    print(f"{'patient':8s} {'events':>6s} {'baseFA':>7s} {'base':>6s} "
          f"{'tuned@sameFA':>13s} {'delta':>7s}")

    tot_ev = base_hit = tuned_hit = 0
    base_hi = tuned_hi = 0          # reverse direction: base raised to tuned FA
    for pat in sorted(data):
        d = data[pat]
        sb = score_blocks(d["base"], d["t_base"], k, m)
        st = score_blocks(d["tuned"], d["t_tuned"], k, m)
        if sb["events"] == 0:
            continue

        tuned_curve = sweep(d["tuned"], k, m)
        base_curve = sweep(d["base"], k, m)

        sens_matched, _, _ = at_fa(tuned_curve, sb["fa_hr"], args.max_flagged)
        sens_base_up, _, _ = at_fa(base_curve, st["fa_hr"], args.max_flagged)

        ev = sb["events"]
        tot_ev += ev
        base_hit += sb["caught"]
        tuned_hit += round(sens_matched * ev)
        base_hi += round(sens_base_up * ev)
        tuned_hi += st["caught"]

        mark = "  <-- up" if sens_matched > sb["sens"] + 1e-9 else (
            "  <-- DOWN" if sens_matched < sb["sens"] - 1e-9 else "")
        print(f"{pat:8s} {ev:6d} {sb['fa_hr']:7.2f} {sb['sens']:6.2f} "
              f"{sens_matched:13.2f} {sens_matched - sb['sens']:+7.2f}{mark}")

    print()
    print(f"Matched at the BASE model's FA rate ({tot_ev} events):")
    print(f"  base  : {base_hit}/{tot_ev} = {base_hit/tot_ev:.3f}")
    print(f"  tuned : {tuned_hit}/{tot_ev} = {tuned_hit/tot_ev:.3f}")
    print(f"  gain  : {(tuned_hit - base_hit)/tot_ev:+.3f}")
    print()
    print("Matched the other way, at the TUNED model's FA rate:")
    print(f"  base  : {base_hi}/{tot_ev} = {base_hi/tot_ev:.3f}")
    print(f"  tuned : {tuned_hi}/{tot_ev} = {tuned_hi/tot_ev:.3f}")
    print(f"  gain  : {(tuned_hi - base_hi)/tot_ev:+.3f}")


if __name__ == "__main__":
    main()
