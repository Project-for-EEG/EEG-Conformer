"""
Per-patient threshold calibration.

A single global threshold cannot serve every patient: on the same architecture,
fold 3 reached 100% event sensitivity while fold 1 reached 55%, and the
threshold that suits one starves the other. But the false-alarm rate depends
only on how the model behaves on a patient's *background* EEG -- which needs no
seizure labels at all.

So: take the first hour of each patient's recording, find the threshold that
would hold false alarms to a target rate over that hour, and apply it to the
rest of that patient's data. Calibration is fully unsupervised -- the labels in
the calibration window are never read.

Reported against the best single global threshold, tuned on the same data, so
the comparison is fair to the baseline.

Usage:
    python calibrate_per_patient.py --target-fa 1.0
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from config import ModelConfig
from model import create_model
from evaluate_temporal import find_runs, smooth, predict_file, SECONDS_PER_WINDOW

PREP_DIR = Path("preprocessed_data")
CKPT_DIR = Path("checkpoints")
WINDOWS_PER_HOUR = int(3600 / SECONDS_PER_WINDOW)   # 1800


def detect(probs, thresh, k, min_consec):
    """Smoothed, thresholded, blip-filtered detections for one contiguous block."""
    pred = smooth(probs, k) >= thresh
    if min_consec > 1:
        cleaned = np.zeros_like(pred)
        for lo, hi in find_runs(pred):
            if hi - lo >= min_consec:
                cleaned[lo:hi] = True
        pred = cleaned
    return pred


def fa_per_hour_on_background(probs, thresh, k, min_consec):
    """(FA/hr, flagged fraction) assuming the block is seizure-free.

    Both are needed. Counting alarm *runs* alone is gameable: a threshold low
    enough to flag everything merges into a single run, scoring ~1 FA/hr while
    the alarm never actually stops. The flagged fraction catches that.
    """
    pred = detect(probs, thresh, k, min_consec)
    hours = len(probs) * SECONDS_PER_WINDOW / 3600
    fa = len(find_runs(pred)) / hours if hours else 0.0
    return fa, float(pred.mean()) if len(pred) else 0.0


def pick_threshold(cal_probs, target_fa, k, min_consec, max_flagged=0.02):
    """Lowest threshold whose FA rate on the calibration block meets the budget.

    Lower threshold means more sensitivity, so we want the smallest one that
    still fits the budget. FA is monotone decreasing in threshold.
    """
    grid = np.unique(np.concatenate([
        np.linspace(0.001, 0.999, 400),
        np.quantile(cal_probs, np.linspace(0.90, 0.99999, 200)),
    ]))
    grid.sort()
    for t in grid:
        fa, flagged = fa_per_hour_on_background(cal_probs, t, k, min_consec)
        if fa <= target_fa and flagged <= max_flagged:
            return float(t)
    return 1.0


def score_blocks(blocks, thresh, k, min_consec):
    """Event sensitivity / FA-hr / precision over a list of contiguous blocks."""
    ev_tot = ev_hit = fa = tp = fp = 0
    windows = 0
    for probs, labels in blocks:
        pred = detect(probs, thresh, k, min_consec)
        truth = labels.astype(bool)
        windows += len(pred)
        tp += int(np.sum(pred & truth))
        fp += int(np.sum(pred & ~truth))
        events = find_runs(truth)
        ev_tot += len(events)
        ev_hit += sum(1 for lo, hi in events if pred[lo:hi].any())
        for lo, hi in find_runs(pred):
            if not truth[lo:hi].any():
                fa += 1
    hours = windows * SECONDS_PER_WINDOW / 3600
    return {
        "events": ev_tot, "caught": ev_hit,
        "sens": ev_hit / ev_tot if ev_tot else float("nan"),
        "fa_hr": fa / hours if hours else 0.0,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        # Same guard as calibration: a saturated detector merges into one run
        # per recording, most overlapping a real seizure, so it books almost no
        # false alarms while flagging everything.
        "flagged": (tp + fp) / windows if windows else 0.0,
        "hours": hours,
    }


def split_calibration(per_file, cal_windows):
    """First cal_windows windows -> calibration; remainder -> test blocks.

    Blocks stay contiguous and never span a recording boundary, so detected
    runs are physically meaningful.
    """
    cal, test, taken = [], [], 0
    for probs, labels in per_file:
        if taken >= cal_windows:
            test.append((probs, labels)); continue
        need = cal_windows - taken
        if len(probs) <= need:
            cal.append(probs); taken += len(probs)
        else:
            cal.append(probs[:need])
            test.append((probs[need:], labels[need:]))
            taken += need
    test = [(pr, lb) for pr, lb in test if len(pr) >= 10]
    return (np.concatenate(cal) if cal else np.array([])), test


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target-fa", type=float, default=1.0, help="false alarms/hour budget")
    ap.add_argument("--cal-hours", type=float, default=1.0, help="calibration period per patient")
    ap.add_argument("--max-flagged", type=float, default=0.02,
                    help="max fraction of calibration time allowed in alarm")
    ap.add_argument("--smooth", type=int, default=5)
    ap.add_argument("--min-consec", type=int, default=3)
    args = ap.parse_args()

    cal_windows = int(args.cal_hours * WINDOWS_PER_HOUR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    k, m = args.smooth, args.min_consec

    print(f"target {args.target_fa} FA/hr | {args.cal_hours} h calibration "
          f"({cal_windows} windows) | smooth k={k}, min_consec={m}\n")

    per_patient_blocks, per_patient_thresh, all_blocks = {}, {}, []

    for ckpt_path in sorted(CKPT_DIR.glob("fold*.pt")):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = create_model(ModelConfig(**ckpt["model_config"])).to(device)
        model.load_state_dict(ckpt["model_state_dict"]); model.eval()

        for patient in ckpt["val_patients"]:
            per_file = [predict_file(model, device, f)
                        for f in sorted((PREP_DIR / patient).glob("*.npz"))]
            cal, test = split_calibration(per_file, cal_windows)
            if len(cal) < cal_windows // 2 or not test:
                print(f"  {patient}: too short to calibrate, skipped"); continue
            t = pick_threshold(cal, args.target_fa, k, m, args.max_flagged)
            per_patient_thresh[patient] = t
            per_patient_blocks[patient] = test
            all_blocks.extend(test)
        del model; torch.cuda.empty_cache()

    print(f"{'patient':8s} {'thresh':>7s} {'events':>8s} {'sens':>7s} {'FA/hr':>7s} {'prec':>7s}")
    tot_ev = tot_hit = 0
    for p in sorted(per_patient_blocks):
        s = score_blocks(per_patient_blocks[p], per_patient_thresh[p], k, m)
        tot_ev += s["events"]; tot_hit += s["caught"]
        print(f"{p:8s} {per_patient_thresh[p]:7.4f} {s['caught']:3d}/{s['events']:<4d} "
              f"{s['sens']:7.3f} {s['fa_hr']:7.2f} {s['precision']:7.3f}")

    print(f"\nPER-PATIENT CALIBRATION: {tot_hit}/{tot_ev} events = "
          f"{tot_hit/tot_ev:.3f} event sensitivity")

    # Fair baseline: the single global threshold that hits the same FA budget
    # on exactly the same test data.
    print(f"\nBest single global threshold at the same {args.target_fa} FA/hr budget:")
    best = None
    for t in np.linspace(0.005, 0.999, 300):
        s = score_blocks(all_blocks, t, k, m)
        if s["fa_hr"] <= args.target_fa and s["flagged"] <= args.max_flagged:
            best = (t, s); break

    stats = {p_: score_blocks(per_patient_blocks[p_], per_patient_thresh[p_], k, m)
             for p_ in per_patient_blocks}
    hrs = sum(v["hours"] for v in stats.values())
    fa_pooled = sum(v["fa_hr"] * v["hours"] for v in stats.values()) / hrs
    print(f"  per-patient achieved : {tot_hit/tot_ev:.3f} sensitivity at "
          f"{fa_pooled:.2f} FA/hr pooled over {hrs:.0f} h")
    if best:
        t, s = best
        print(f"  best single global   : {s['sens']:.3f} sensitivity at "
              f"{s['fa_hr']:.2f} FA/hr (threshold {t:.4f}, "
              f"{100*s['flagged']:.2f}% flagged)")
        print()
        print(f"  per-patient {tot_hit/tot_ev:.3f}  vs  global "
              f"{s['sens']:.3f}  ->  {(tot_hit/tot_ev) - s['sens']:+.3f}")
    else:
        print(f"  best single global   : NONE meets {args.target_fa} FA/hr "
              f"within {100*args.max_flagged:.0f}% flagged time")


if __name__ == "__main__":
    main()
