"""
Re-tune the operating point for event F1, which is what SzCORE actually scores.

Every threshold in this project so far was chosen to maximise event sensitivity
at a tolerated false-alarm rate. That is the right objective for a clinical
alarm and the wrong one for the SzCORE leaderboard, which reports per-subject
event F1 and therefore prices a false alarm against a caught seizure directly.

The gap is not small. Of the 17 algorithms with published Dianalund results,
the two with the HIGHEST sensitivity placed last and second to last:

    f1 0.0218   sensitivity 1.000   290 FP/day     <- caught every seizure
    f1 0.0210   sensitivity 0.993   282 FP/day
    f1 0.3043   sensitivity 0.584    13 FP/day     <- best of the 17

So the leaderboard is won on precision, and this project has never optimised
for it. This script re-scores the cached leave-one-patient-out predictions
under the real scorer and reports the F1-optimal operating point.

WHAT IS DIFFERENT FROM score_szcore.py

    score_szcore.py re-implements the SzCORE event rules locally. This uses
    the actual `timescoring` package with default Parameters(), so there is no
    chance of the re-implementation drifting from the library that will score
    the submission. The defaults are confirmed to be tolerance 30 s before and
    60 s after, events merged when closer than 90 s, events split at 300 s.

    Two things the local scorer does not match, and this one does:
      - fpRate is false alarms per DAY, not per hour.
      - Subjects are averaged with nanmean, not pooled. A patient with one
        hour of recording counts as much as a patient with twenty.

WHY TUNE ON CHB-MIT AND CHECK ON SIENA

    A setting tuned on one cohort can simply be overfitted to it. Siena has 33
    events, and the README already warns that two seizures flipping moves its
    row by 6 points. So each cohort is tuned independently and then evaluated
    on the other. The setting that survives both is the one to submit, because
    Dianalund is held out and cannot be tuned on at all.

Usage:
    python retune_f1.py
    python retune_f1.py --cache endtoend_lopo.npz --check endtoend_siena_only.npz

Needs `pip install timescoring`.
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from timescoring import scoring
from timescoring.annotations import Annotation

from evaluate_temporal import find_runs, smooth, SECONDS_PER_WINDOW

# One prediction every SECONDS_PER_WINDOW of recording, so the mask handed to
# timescoring is sampled at that rate. The library resamples internally; what
# matters is that the rate it is told matches the rate the mask was built at.
MASK_FS = 1.0 / SECONDS_PER_WINDOW

THRESHOLDS = np.round(np.concatenate([np.linspace(0.70, 0.98, 15),
                                      np.linspace(0.982, 0.999, 8)]), 4)
SMOOTHING = [1, 2, 3, 5, 7, 9]
MIN_CONSEC = [2, 3, 4, 5, 6, 8, 11]

# The setting the project has been running, kept so the comparison is concrete
# rather than remembered.
BASELINE = (0.45, 1, 1)


def load_blocks(path):
    """{patient: [(probs, labels), ...]} from an evaluate_end_to_end cache."""
    data = np.load(path, allow_pickle=True)["data"].item()
    return {p: [(np.asarray(pr, np.float64), np.asarray(la).astype(bool))
                for pr, la in data[p]["blocks"]]
            for p in sorted(data)}


def detect(probs, thresh, k, min_consec):
    """Same rule as calibrate_per_patient.detect, on already-loaded probs."""
    pred = smooth(probs, k) >= thresh
    if min_consec > 1:
        cleaned = np.zeros_like(pred)
        for lo, hi in find_runs(pred):
            if hi - lo >= min_consec:
                cleaned[lo:hi] = True
        pred = cleaned
    return pred


def evaluate(blocks, thresh, k, min_consec):
    """(mean per-subject F1, pooled FP/day, mean sensitivity, mean precision).

    Counts are accumulated per patient across that patient's recordings and the
    rates computed once, rather than averaging rates over recordings. A patient
    is scored as a patient, which is what the leaderboard does.

    Recordings are scored separately rather than concatenated, because joining
    two recordings end to end invents a boundary that is neither seizure nor
    background and would book a false event there.
    """
    f1s, sens, precs = [], [], []
    false_alarms, seconds = 0, 0.0
    for patient, recordings in blocks.items():
        tp = fp = ref = 0
        for probs, labels in recordings:
            pred = detect(probs, thresh, k, min_consec)
            scored = scoring.EventScoring(Annotation(labels, MASK_FS),
                                          Annotation(pred, MASK_FS))
            tp += scored.tp
            fp += scored.fp
            ref += scored.refTrue
            seconds += len(labels) * SECONDS_PER_WINDOW
        false_alarms += fp
        if ref == 0:          # nothing to be sensitive to; would be a nan
            continue
        recall = tp / ref
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        denom = recall + precision
        f1s.append(2 * recall * precision / denom if denom else 0.0)
        sens.append(recall)
        precs.append(precision)
    return (float(np.mean(f1s)), false_alarms / (seconds / 86400),
            float(np.mean(sens)), float(np.mean(precs)))


def sweep(blocks):
    """Every (threshold, k, min_consec) scored, best first."""
    rows = []
    for k in SMOOTHING:
        for min_consec in MIN_CONSEC:
            for thresh in THRESHOLDS:
                f1, fa_day, se, pr = evaluate(blocks, thresh, k, min_consec)
                rows.append({"f1": f1, "fa_per_day": fa_day, "sensitivity": se,
                             "precision": pr, "threshold": float(thresh),
                             "k": k, "min_consec": min_consec})
    rows.sort(key=lambda r: -r["f1"])
    return rows


def show(label, r):
    print("%-14s F1 %.4f | %6.1f FA/day | sensitivity %.3f precision %.3f"
          " | threshold %.4f k %d min_consec %d"
          % (label, r["f1"], r["fa_per_day"], r["sensitivity"], r["precision"],
             r["threshold"], r["k"], r["min_consec"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default="endtoend_lopo.npz",
                    help="predictions to tune on")
    ap.add_argument("--check", default="endtoend_siena_only.npz",
                    help="second cohort, to test that the setting transfers")
    args = ap.parse_args()

    tune = load_blocks(args.cache)
    print("tuning on  %s: %d patients" % (args.cache, len(tune)))
    rows = sweep(tune)
    best = rows[0]

    print()
    show("tuned best", best)
    b = dict(zip(("threshold", "k", "min_consec"), BASELINE))
    f1, fa, se, pr = evaluate(tune, *BASELINE)
    show("baseline", {**b, "f1": f1, "fa_per_day": fa,
                      "sensitivity": se, "precision": pr})
    print("               -> %.2fx on event F1, at %.0f%% of the false alarms"
          % (best["f1"] / f1, 100 * best["fa_per_day"] / fa))

    print("\nbest F1 within a false-alarm budget, since Dianalund's leaders all")
    print("sit between 6 and 23 FA/day and this project has run at ~200:")
    for cap in (5, 13, 23, 50, 100):
        under = [r for r in rows if r["fa_per_day"] <= cap]
        if under:
            show("  <= %d/day" % cap, under[0])
        else:
            print("  <= %d/day     nothing reaches this" % cap)

    out = {"tuned_on": args.cache, "best": best,
           "baseline": {"threshold": BASELINE[0], "k": BASELINE[1],
                        "min_consec": BASELINE[2], "f1": f1, "fa_per_day": fa},
           "top": rows[:25]}

    if args.check and Path(args.check).exists():
        other = load_blocks(args.check)
        print("\ndoes the setting transfer? %s has %d patients"
              % (args.check, len(other)))
        other_best = sweep(other)[0]
        for label, r in (("tuned here", best), ("tuned there", other_best)):
            a = evaluate(tune, r["threshold"], r["k"], r["min_consec"])
            c = evaluate(other, r["threshold"], r["k"], r["min_consec"])
            print("  %-12s (threshold %.4f k %d min_consec %d)"
                  "  on %s F1 %.4f  |  on %s F1 %.4f"
                  % (label, r["threshold"], r["k"], r["min_consec"],
                     Path(args.cache).stem, a[0], Path(args.check).stem, c[0]))
        print("\nthe setting to submit is whichever holds up on BOTH, not the")
        print("one with the single highest number. A setting that is good on")
        print("its own cohort and worse on the other is fitted to that cohort.")
        out["transfer"] = {"check_cache": args.check, "check_best": other_best}

    path = Path("cv_results") / ("retune_f1_%s.json"
                                 % datetime.now().strftime("%Y%m%d_%H%M%S"))
    path.write_text(json.dumps(out, indent=2))
    print("\nsaved %s" % path)


if __name__ == "__main__":
    main()
