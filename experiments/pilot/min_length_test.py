"""
Item 3: does a longer minimum alarm length cut false alarms, and what does it cost?

Re-scores saved predictions only. No model is run, nothing in the repo is
written. Uses the repo's own detect / score_all (official timescoring) and
pick_threshold unchanged, so the N=3 rows must reproduce the README.

Rule N = an alarm needs N consecutive positive windows after smoothing.
Windows are 4 s long at a 2-s step, so N windows cover 2N+2 s of signal
(N=3 -> 8 s, today's rule; N=9 -> 20 s, close to their 20-s rule).

Three ways of setting the threshold, kept separate:
  stored    the thresholds saved in the cache, picked for N=3. Test-free.
  repicked  calibrated caches only: each patient's threshold re-picked from
            that patient's own stored background at the same 2 FA/h target,
            with rule N. This is what the pipeline itself would do. The
            base-versus-tuned model choice is held as in the original run.
  oracle    one global threshold swept over the test predictions. Uses the
            test result, so it is an upper bound and labelled as such.

Usage (from C:\\EEG):
    python <this file> --out <scratchpad>/min_length_results.txt
"""
import argparse
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

REPO = Path(r"C:\EEG")
sys.path.insert(0, str(REPO))

from score_official import score_all          # noqa: E402
from calibrate_per_patient import pick_threshold  # noqa: E402
from evaluate_temporal import find_runs       # noqa: E402

K = 5                 # smoothing, as in every headline run
TARGET_FA = 2.0       # per hour, the calibration target of both headline runs
RULES = (3, 4, 5, 6, 7, 9, 11)
BUDGETS = (1, 2, 5, 10, 25, 50, 100)
CACHES = [
    ("CHB-MIT, no test-patient data", "endtoend_uncalibrated.npz", False),
    ("CHB-MIT, patient-calibrated", "endtoend_lopo.npz", True),
    ("Siena, patient-calibrated", "endtoend_siena_only.npz", True),
]


def seconds(n):
    return 2 * n + 2


def load(name):
    return np.load(REPO / name, allow_pickle=True)["data"].item()


def oracle_grid(data):
    probs = np.concatenate([np.asarray(b[0], dtype=np.float64)
                            for p in data for b in data[p]["blocks"]])
    return np.unique(np.concatenate([
        np.linspace(0.50, 0.999, 100),
        np.quantile(probs, np.linspace(0.900, 0.99999, 300)),
    ]))


def sweep(job):
    """Oracle curve for one (cache, rule): list of (fa_day, sens) pooled and per subject."""
    name, n = job
    data = load(name)
    out = []
    for t in oracle_grid(data):
        pl, sb, _ = score_all(data, {p: float(t) for p in data}, K, n)
        out.append((float(t), pl["fa_day"], pl["sens"], sb["fa_day"], sb["sens"]))
    return name, n, out


def fmt_row(label, pl, sb):
    return ("%-9s %4d/%-4d %7.3f %8.1f %8.1f   %7.3f %8.1f"
            % (label, pl["caught"], pl["events"], pl["sens"], pl["fa_day"],
               pl["fa"], sb["sens"], sb["fa_day"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    lines = []

    def say(s=""):
        print(s, flush=True)
        lines.append(s)

    t0 = time.time()
    say("Minimum alarm length test. Saved predictions only; no model run.")
    say("smoothing %d windows; rule N = N consecutive windows = 2N+2 s of signal" % K)
    say()

    # ---- seizure lengths, so the cost side is visible -----------------------
    say("SEIZURE LENGTHS (reference events, in positive windows at a 2-s step)")
    for label, name, _ in CACHES[1:]:
        data = load(name)
        lens = [hi - lo for p in data for _, lab in data[p]["blocks"]
                for lo, hi in find_runs(np.asarray(lab).astype(bool))]
        lens = np.array(lens)
        say("  %s: %d labelled runs, median %d windows (~%d s)"
            % (label.split(",")[0], len(lens), np.median(lens), 2 * np.median(lens) + 2))
        say("    shorter than rule N:  " + "  ".join(
            "N=%d:%d" % (n, int((lens < n).sum())) for n in RULES))
    say("  (a labelled run shorter than N windows can still be caught: smoothing")
    say("   and the 30 s / 60 s scoring tolerance let a detection overlap it)")
    say()

    # ---- stored and re-picked thresholds ------------------------------------
    for label, name, calibrated in CACHES:
        data = load(name)
        stored = {p: float(data[p]["t"]) for p in data}
        say("=" * 78)
        say("%s  (%s, %d cases)" % (label, name, len(data)))
        if calibrated:
            rep3 = {p: pick_threshold(np.asarray(data[p]["bg"], dtype=np.float64),
                                      TARGET_FA, K, 3) for p in data}
            diff = max(abs(rep3[p] - stored[p]) for p in data)
            say("check: re-picking at N=3 reproduces the stored thresholds, "
                "max difference %.2e" % diff)
        say("%-9s %9s %7s %8s %8s   %7s %8s" % ("rule", "caught", "sens", "FA/day",
                                                 "FAs", "subj.sens", "subj.FA/d"))
        say("  thresholds as stored (picked for N=3, test-free):")
        for n in RULES:
            pl, sb, _ = score_all(data, stored, K, n)
            say(fmt_row("N=%d %2ds" % (n, seconds(n)), pl, sb))
        if calibrated:
            say("  thresholds re-picked per patient for each rule (deployable):")
            for n in RULES:
                th = {p: pick_threshold(np.asarray(data[p]["bg"], dtype=np.float64),
                                        TARGET_FA, K, n) for p in data}
                pl, sb, _ = score_all(data, th, K, n)
                say(fmt_row("N=%d %2ds" % (n, seconds(n)), pl, sb))
        say()

    # ---- oracle curves --------------------------------------------------------
    jobs = [(name, n) for _, name, _ in CACHES for n in RULES]
    with Pool(args.workers) as pool:
        res = pool.map(sweep, jobs)
    curves = {(name, n): out for name, n, out in res}
    say("=" * 78)
    say("ORACLE: best pooled sensitivity inside each alarm budget, one global")
    say("threshold swept over the test predictions. Upper bounds, not results.")
    for label, name, _ in CACHES:
        say()
        say(label)
        say("%-9s" % "rule" + "".join("%9s" % ("<=%d/d" % b) for b in BUDGETS))
        for n in RULES:
            row = "%-9s" % ("N=%d %2ds" % (n, seconds(n)))
            for b in BUDGETS:
                ok = [s for s in curves[(name, n)] if s[1] <= b]
                row += "%9.3f" % max(s[2] for s in ok) if ok else "%9s" % "-"
            say(row)
    say()
    say("done in %.0f min" % ((time.time() - t0) / 60))
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="ascii")


if __name__ == "__main__":
    main()
