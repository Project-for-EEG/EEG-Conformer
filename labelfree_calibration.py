"""
Is the per-patient calibration gap real, or is it a score-scale artifact?

Per-patient thresholds are worth 16.1 points of pooled event sensitivity:
78.0% with a threshold from the training fold, 94.0% with one from the held-out
patient's own earlier recording. The obvious reading is that the pipeline needs
to know the patient.

There is a second reading. uncalibrated_lopo.py found that across the 23 fold
checkpoints the background probability medians run from 0.025 to 0.998. The
fold models are nowhere near a common scale, so one absolute threshold cannot
suit all of them, and a threshold taken from the training fold is being applied
to a model whose output distribution has moved. On that reading, part of the
gap is calibration rather than patient knowledge.

The two readings differ in what they imply. If the gap is patient knowledge,
nothing can be done without that patient's data. If it is scale, a threshold
expressed as a PERCENTILE of the recording being processed fixes it for free:
no labels, no earlier session, no patient identity, nothing but the file
already in hand.

SCHEMES COMPARED, all on the same base-model predictions

    fixed        one absolute threshold for every patient. This is the
                 uncalibrated baseline and the thing to beat.
    per_file     the pct-th percentile of this recording's own scores. Uses
                 the whole recording, so it is not causal -- an upper bound on
                 what percentile transfer can do.
    causal       the pct-th percentile of the preceding chunk only, which is
                 deployable: hour k is thresholded by hour k-1. The first chunk
                 has no history, so it falls back to the training-fold
                 threshold rather than to anything of the patient's.
    expanding    the pct-th percentile of everything seen so far in the
                 recording. Also causal, and steadier than a one-hour window.

HOW THEY ARE COMPARED

    Not at a shared percentile, which would compare different alarm rates.
    Each scheme is swept and read at the SAME false-alarm rate the calibrated
    pipeline achieves, 5.16 per hour, so the only thing differing is where the
    threshold came from. Reading two systems at two alarm rates and calling the
    difference a result is the error this project has already made once, in
    the Helsinki scaling curve.

Usage:
    python labelfree_calibration.py
    python labelfree_calibration.py --target-fa 5.16 --out labelfree.txt

Needs `pip install timescoring`.
"""
import argparse
from pathlib import Path

import numpy as np
from timescoring import scoring
from timescoring.annotations import Annotation

from evaluate_temporal import find_runs, smooth, SECONDS_PER_WINDOW

MASK_FS = 1.0 / SECONDS_PER_WINDOW
WINDOWS_PER_HOUR = int(3600 / SECONDS_PER_WINDOW)

# Matches score_szcore.py and the committed szcore_lopo.txt.
SMOOTH_K = 5
MIN_CONSEC = 3


def detect_with(probs, thresh, k, min_consec):
    """thresh may be a scalar or one value per window."""
    pred = smooth(probs, k) >= thresh
    if min_consec > 1:
        cleaned = np.zeros_like(pred)
        for lo, hi in find_runs(pred):
            if hi - lo >= min_consec:
                cleaned[lo:hi] = True
        pred = cleaned
    return pred


def thresholds_for(scheme, probs, pct, fallback, chunk):
    """Per-window thresholds under one scheme. Never reads labels."""
    n = len(probs)
    if scheme == "fixed":
        return np.full(n, fallback, dtype=float)
    if scheme == "per_file":
        return np.full(n, float(np.percentile(probs, pct)), dtype=float)
    if scheme == "causal":
        th = np.full(n, fallback, dtype=float)
        for start in range(chunk, n, chunk):
            prev = probs[start - chunk:start]
            if len(prev):
                th[start:start + chunk] = np.percentile(prev, pct)
        return th
    if scheme == "expanding":
        th = np.full(n, fallback, dtype=float)
        for start in range(chunk, n, chunk):
            th[start:start + chunk] = np.percentile(probs[:start], pct)
        return th
    raise ValueError(scheme)


def score_scheme(data, scheme, pct, fallbacks, chunk):
    """(pooled, per-subject) event metrics for one scheme at one percentile."""
    per = {}
    tp_t = fp_t = ref_t = 0
    sec_t = 0.0
    for patient in sorted(data):
        tp = fp = ref = 0
        sec = 0.0
        for probs, labels in data[patient]["blocks"]:
            probs = np.asarray(probs, dtype=np.float64)
            th = thresholds_for(scheme, probs, pct, fallbacks[patient], chunk)
            pred = detect_with(probs, th, SMOOTH_K, MIN_CONSEC)
            truth = np.asarray(labels).astype(bool)
            s = scoring.EventScoring(Annotation(truth, MASK_FS),
                                     Annotation(pred, MASK_FS))
            tp += s.tp
            fp += s.fp
            ref += s.refTrue
            sec += len(truth) * SECONDS_PER_WINDOW
        tp_t += tp
        fp_t += fp
        ref_t += ref
        sec_t += sec
        if ref:
            se = tp / ref
            pr = tp / (tp + fp) if (tp + fp) else 0.0
            per[patient] = (se, pr, fp / (sec / 3600))
    pooled = {"sens": tp_t / ref_t if ref_t else float("nan"),
              "prec": tp_t / (tp_t + fp_t) if (tp_t + fp_t) else 0.0,
              "fa_hr": fp_t / (sec_t / 3600), "fa_day": fp_t / (sec_t / 86400),
              "caught": tp_t, "events": ref_t}
    pooled["f1"] = (2 * pooled["sens"] * pooled["prec"] /
                    (pooled["sens"] + pooled["prec"])
                    if pooled["sens"] and pooled["prec"] else 0.0)
    subject = {"sens": float(np.mean([v[0] for v in per.values()])),
               "fa_hr": float(np.mean([v[2] for v in per.values()]))}
    return pooled, subject


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--uncalibrated", default="endtoend_uncalibrated.npz",
                    help="base-model predictions, training-fold thresholds")
    ap.add_argument("--calibrated", default="endtoend_lopo.npz",
                    help="the headline run, for the target to match")
    ap.add_argument("--target-fa", type=float, default=None,
                    help="alarm rate to read every scheme at; default is "
                         "whatever the calibrated run achieves")
    ap.add_argument("--roll-hours", type=float, default=1.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    unc = np.load(args.uncalibrated, allow_pickle=True)["data"].item()
    cal = np.load(args.calibrated, allow_pickle=True)["data"].item()
    chunk = int(args.roll_hours * WINDOWS_PER_HOUR)
    fallbacks = {p: float(unc[p]["t"]) for p in unc}

    out = []

    def say(line=""):
        print(line)
        out.append(line)

    say("label-free calibration: is the 16-point gap scale or patient knowledge?")
    say("predictions %s, base models only, no personalisation" % args.uncalibrated)
    say()

    # The two ends of the gap, both scored the same way.
    cal_pooled, cal_sub = score_scheme(
        cal, "fixed", 0, {p: float(cal[p]["t"]) for p in cal}, chunk)
    target = args.target_fa if args.target_fa else cal_pooled["fa_hr"]
    say("CALIBRATED   per-patient threshold from the patient's own recording")
    say("  sensitivity %.4f (%d of %d)  %.2f FA/h  %.1f FA/day  F1 %.4f"
        % (cal_pooled["sens"], cal_pooled["caught"], cal_pooled["events"],
           cal_pooled["fa_hr"], cal_pooled["fa_day"], cal_pooled["f1"]))
    unc_pooled, unc_sub = score_scheme(unc, "fixed", 0, fallbacks, chunk)
    say("UNCALIBRATED one threshold from the training fold, nothing of theirs")
    say("  sensitivity %.4f (%d of %d)  %.2f FA/h  %.1f FA/day  F1 %.4f"
        % (unc_pooled["sens"], unc_pooled["caught"], unc_pooled["events"],
           unc_pooled["fa_hr"], unc_pooled["fa_day"], unc_pooled["f1"]))
    gap = cal_pooled["sens"] - unc_pooled["sens"]
    say()
    say("the gap to close: %+.4f pooled event sensitivity" % gap)
    say("every scheme below is read at %.2f FA/h, the calibrated rate, so the"
        % target)
    say("only difference is where the threshold came from")
    say()

    pcts = np.concatenate([np.linspace(90.0, 99.0, 37),
                           np.linspace(99.1, 99.99, 40)])
    say("%-11s %9s %9s %9s %9s %9s %10s"
        % ("scheme", "sens", "FA/h", "F1", "prec", "closed", "percentile"))
    rows = []
    for scheme in ("per_file", "causal", "expanding"):
        best = None
        for pct in pcts:
            pooled, sub = score_scheme(unc, scheme, float(pct), fallbacks, chunk)
            if pooled["fa_hr"] <= target:
                if best is None or pooled["sens"] > best[0]["sens"]:
                    best = (pooled, sub, float(pct))
        if best is None:
            say("%-11s   no setting stays inside the alarm budget" % scheme)
            continue
        pooled, sub, pct = best
        closed = (pooled["sens"] - unc_pooled["sens"]) / gap if gap else 0.0
        rows.append((scheme, pooled, sub, pct, closed))
        say("%-11s %9.4f %9.2f %9.4f %9.3f %8.0f%% %10.2f"
            % (scheme, pooled["sens"], pooled["fa_hr"], pooled["f1"],
               pooled["prec"], 100 * closed, pct))

    say()
    if rows:
        scheme, pooled, sub, pct, closed = max(rows, key=lambda r: r[4])
        say("BEST LABEL-FREE SCHEME: %s at the %.2f-th percentile" % (scheme, pct))
        say("  closes %.0f%% of the %.3f gap, reaching %.4f against the"
            % (100 * closed, gap, pooled["sens"]))
        say("  calibrated %.4f and the uncalibrated %.4f, all at <= %.2f FA/h"
            % (cal_pooled["sens"], unc_pooled["sens"], target))
        say()
        if closed >= 0.5:
            say("READ: most of what looked like per-patient knowledge was the fold")
            say("models not sharing a score scale. A percentile of the recording")
            say("already in hand recovers it, with no labels and no patient history.")
        elif closed >= 0.2:
            say("READ: part of the gap is scale and part is not. A percentile")
            say("recovers some of it for free; the rest needs the patient.")
        else:
            say("READ: the gap is not mainly a scale artifact. Per-patient")
            say("calibration is doing something a percentile cannot imitate, so")
            say("the 16 points are the price of not having the patient's data.")
        say()
        say("'per_file' uses the whole recording and is therefore an upper bound,")
        say("not deployable. 'causal' and 'expanding' only look backwards and are.")

    if args.out:
        Path(args.out).write_text("\n".join(out) + "\n", encoding="ascii")
        print("\nsaved %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
