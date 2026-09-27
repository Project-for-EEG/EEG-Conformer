"""
Score cached predictions with the official SzCORE library rather than ours.

score_szcore.py re-implements the SzCORE event rules. This runs the same
predictions through `timescoring`, the package the benchmark itself uses, so
the numbers in the README do not depend on our re-implementation being right.

They agree. On endtoend_lopo.npz the two produce identical refTrue, true
positives and false positives, patient by patient and pooled. That is worth
having on record, because it means the re-implementation is sound and any
difference between our numbers and a published one is a difference of
protocol, not of arithmetic. The two implementations do differ in one rule
that happens not to fire here: timescoring merges reference events less than
90 s apart before splitting them, and score_szcore.py does not. The smallest
gap between two true seizures in this cache is 92 s, one window clear of it.

WHAT THIS ADDS THAT score_szcore.py DOES NOT REPORT

    false alarms per 24 h   The field quotes alarms per day. 5.16 per hour and
                            124 per day are the same number and read very
                            differently.
    precision and F1        Sensitivity alone says nothing about how often the
                            detector is wrong when it fires.
    per-subject averages    SzCORE averages metrics across subjects; our
                            headline pools every event into one ratio. Pooling
                            lets the patients with the most seizures dominate.
                            Both are reported, labelled, because they differ.
    fixed alarm budgets     Sensitivity reachable at 1, 2, 5 and 10 alarms per
                            day, which is the range a clinical detector would
                            have to work in.

THE BUDGET ROWS ARE AN ORACLE AND ARE LABELLED AS SUCH

    Each budget row sweeps one global threshold over the test predictions and
    keeps the best sensitivity that stays inside the budget. The threshold is
    therefore chosen knowing the test result. No deployable system can do that.
    The rows show what the model's ranking could support at those alarm rates,
    not what this pipeline would achieve.

Usage:
    python score_official.py --cache endtoend_lopo.npz
    python score_official.py --cache endtoend_lopo.npz --out szcore_official.txt
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from timescoring import scoring
from timescoring.annotations import Annotation

from evaluate_temporal import find_runs, smooth, SECONDS_PER_WINDOW

MASK_FS = 1.0 / SECONDS_PER_WINDOW
BUDGETS = (1, 2, 5, 10)


def detect(probs, thresh, k, min_consec):
    pred = smooth(probs, k) >= thresh
    if min_consec > 1:
        cleaned = np.zeros_like(pred)
        for lo, hi in find_runs(pred):
            if hi - lo >= min_consec:
                cleaned[lo:hi] = True
        pred = cleaned
    return pred


def count(blocks, thresh, k, min_consec):
    """(tp, fp, refTrue, seconds) for one patient, via the official scorer.

    Recordings are scored separately. Concatenating them would invent a
    boundary that belongs to neither recording and book an event across it.
    """
    tp = fp = ref = 0
    seconds = 0.0
    for probs, labels in blocks:
        pred = detect(np.asarray(probs, dtype=np.float64), thresh, k, min_consec)
        truth = np.asarray(labels).astype(bool)
        scored = scoring.EventScoring(Annotation(truth, MASK_FS),
                                      Annotation(pred, MASK_FS))
        tp += scored.tp
        fp += scored.fp
        ref += scored.refTrue
        seconds += len(truth) * SECONDS_PER_WINDOW
    return tp, fp, ref, seconds


def rates(tp, fp, ref, seconds):
    sens = tp / ref if ref else float("nan")
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * sens * prec / (sens + prec) if sens and prec and (sens + prec) else 0.0
    hours = seconds / 3600
    return {"sens": sens, "prec": prec, "f1": f1, "events": ref, "caught": tp,
            "fa": fp, "fa_hr": fp / hours if hours else 0.0,
            "fa_day": fp / (seconds / 86400) if seconds else 0.0, "hours": hours}


def score_all(data, thresholds, k, min_consec):
    """(pooled, per_subject) at the given per-patient thresholds."""
    per, tot_tp = {}, 0
    tot_fp = tot_ref = 0
    tot_sec = 0.0
    for p in sorted(data):
        tp, fp, ref, sec = count(data[p]["blocks"], thresholds[p], k, min_consec)
        per[p] = rates(tp, fp, ref, sec)
        tot_tp += tp
        tot_fp += fp
        tot_ref += ref
        tot_sec += sec
    pooled = rates(tot_tp, tot_fp, tot_ref, tot_sec)
    scored = [v for v in per.values() if v["events"]]
    subject = {m: float(np.mean([v[m] for v in scored]))
               for m in ("sens", "prec", "f1", "fa_hr", "fa_day")}
    return pooled, subject, per


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default="endtoend_lopo.npz")
    ap.add_argument("--smooth", type=int, default=5)
    ap.add_argument("--min-consec", type=int, default=3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    data = np.load(args.cache, allow_pickle=True)["data"].item()
    thresholds = {p: float(data[p]["t"]) for p in data}
    out = []

    def say(line=""):
        print(line)
        out.append(line)

    say("official SzCORE scoring (timescoring) of %s" % args.cache)
    say("smoothing %d windows, min_consec %d, thresholds as stored in the cache"
        % (args.smooth, args.min_consec))
    ts = sorted(set(round(v, 6) for v in thresholds.values()))
    say("%d cases, %d distinct thresholds (%.4f to %.4f)"
        % (len(data), len(ts), min(ts), max(ts)))
    say()

    pooled, subject, per = score_all(data, thresholds, args.smooth, args.min_consec)

    say("%-10s %6s %7s %9s %9s %8s %9s" %
        ("case", "events", "caught", "sens", "precision", "FA/h", "FA/day"))
    for p in sorted(per):
        v = per[p]
        say("%-10s %6d %7d %9.3f %9.3f %8.2f %9.1f"
            % (p, v["events"], v["caught"], v["sens"], v["prec"],
               v["fa_hr"], v["fa_day"]))
    say()
    say("POOLED -- every event in one ratio. This is the README headline.")
    say("  sensitivity %.4f (%d of %d)   precision %.4f   F1 %.4f"
        % (pooled["sens"], pooled["caught"], pooled["events"],
           pooled["prec"], pooled["f1"]))
    say("  false alarms %d over %.1f h = %.2f per hour = %.1f per 24 h"
        % (pooled["fa"], pooled["hours"], pooled["fa_hr"], pooled["fa_day"]))
    say()
    say("PER SUBJECT -- each case weighted equally. This is SzCORE's own")
    say("convention, and it is the higher number, so it is not the one quoted.")
    say("  sensitivity %.4f   precision %.4f   F1 %.4f"
        % (subject["sens"], subject["prec"], subject["f1"]))
    say("  false alarms %.2f per hour = %.1f per 24 h"
        % (subject["fa_hr"], subject["fa_day"]))
    say()

    say("SENSITIVITY AT A FIXED ALARM BUDGET")
    say("ORACLE. One global threshold swept over the test predictions, keeping")
    say("the best sensitivity inside each budget. The threshold is chosen with")
    say("the test result in hand, so no deployable system reaches these. They")
    say("bound what the model's ranking supports, nothing more.")
    # A fixed linear grid is too coarse where it matters. The alarm rate is a
    # step function of the threshold, and every step sits at some observed
    # probability, so the grid is drawn from the data: quantiles of the actual
    # test probabilities, which concentrate points exactly where the rate
    # changes. A linear grid alone reported 6.2 alarms/day for a 10/day budget
    # because the next threshold down jumped straight past it.
    all_probs = np.concatenate([np.asarray(b[0], dtype=np.float64)
                                for p in data for b in data[p]["blocks"]])
    grid = np.unique(np.concatenate([
        np.linspace(0.50, 0.999, 200),
        np.quantile(all_probs, np.linspace(0.900, 0.99999, 600)),
    ]))
    swept = []
    for t in grid:
        pl, sb, _ = score_all(data, {p: float(t) for p in data},
                              args.smooth, args.min_consec)
        swept.append((pl, sb, float(t)))
    say("%-12s %9s %9s %9s %9s %10s" %
        ("budget", "sens", "achieved", "precision", "F1", "threshold"))
    for b in BUDGETS:
        ok = [s for s in swept if s[0]["fa_day"] <= b]
        if not ok:
            say("%-12s   nothing reaches this budget" % ("<= %d/day" % b))
            continue
        pl, sb, t = max(ok, key=lambda s: s[0]["sens"])
        say("%-12s %9.4f %9.2f %9.3f %9.4f %10.6f"
            % ("<= %d/day" % b, pl["sens"], pl["fa_day"], pl["prec"],
               pl["f1"], t))
    say("  (rows above are pooled; per-subject budgets below)")
    for b in BUDGETS:
        ok = [s for s in swept if s[1]["fa_day"] <= b]
        if not ok:
            say("%-12s   nothing reaches this budget" % ("<= %d/day" % b))
            continue
        pl, sb, t = max(ok, key=lambda s: s[1]["sens"])
        say("%-12s %9.4f %9.2f %9s %9.4f %10.6f"
            % ("<= %d/day" % b, sb["sens"], sb["fa_day"], "-", sb["f1"], t))

    if args.out:
        Path(args.out).write_text("\n".join(out) + "\n", encoding="ascii")
        print("\nsaved %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
