"""
Does longer temporal context reduce false alarms?

Every model here decides on a single 4-second window in isolation. A 4-second
view cannot separate a seizure onset from a chewing artifact, because what
distinguishes them is what happens over the following minute. Systems that
report false alarms per *day* rather than per hour all use 60-80 s of context.

This tests the idea cheaply. Rather than retraining on raw EEG, it trains a
small model on the *probability sequence* the existing detector already
produces: for each window, the surrounding +/- N windows of probabilities are
the input, and the label is that window's. Nothing about the EEG model changes.

That makes it CPU-only and quick, and it isolates the question: given the same
per-window scores, does seeing the neighbourhood help?

Compared against the current approach, which is a moving average plus a
minimum-run-length rule -- itself a fixed, hand-designed context filter. This
just learns the filter instead.

Splits are by patient, matching the rest of the project.

Usage:
    python temporal_context.py
    python temporal_context.py --context 30      # +/- 30 windows = 2 minutes
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier

from evaluate_temporal import find_runs, SECONDS_PER_WINDOW
from calibrate_per_patient import detect

CACHE = "endtoend_leakfree.npz"
SAME_SUBJECT = [("CHB01", "CHB21")]


def windows_with_context(probs, radius):
    """(n, 2*radius+1) matrix: each row is a window and its neighbours.

    Edges are padded by repeating the boundary value, so every window keeps a
    row and the sequence stays aligned with its labels.
    """
    padded = np.pad(probs, radius, mode="edge")
    idx = np.arange(len(probs))[:, None] + np.arange(2 * radius + 1)[None, :]
    return padded[idx]


def build_features(blocks, radius):
    """Context matrix and labels for one patient, blocks kept separate so the
    neighbourhood never spans a recording boundary."""
    X, y = [], []
    for probs, labels in blocks:
        if len(probs) < 2 * radius + 1:
            continue
        X.append(windows_with_context(probs, radius))
        y.append(labels)
    if not X:
        return None, None
    return np.concatenate(X), np.concatenate(y)


def score_sequence(blocks, predict_fn, radius, min_consec=3, thresh=0.5):
    """Event sensitivity and FA/hour from a per-window scoring function.

    thresh is swept by the caller: the baseline uses each patient's calibrated
    threshold, so scoring the learned model at a flat 0.5 would compare two
    different operating points rather than two methods.
    """
    ev_tot = ev_hit = fa = windows = 0
    for probs, labels in blocks:
        if len(probs) < 2 * radius + 1:
            continue
        s = predict_fn(windows_with_context(probs, radius))
        pred = s >= thresh
        if min_consec > 1:
            cleaned = np.zeros_like(pred)
            for lo, hi in find_runs(pred):
                if hi - lo >= min_consec:
                    cleaned[lo:hi] = True
            pred = cleaned
        truth = labels.astype(bool)
        windows += len(pred)
        events = find_runs(truth)
        ev_tot += len(events)
        ev_hit += sum(1 for lo, hi in events if pred[lo:hi].any())
        for lo, hi in find_runs(pred):
            if not truth[lo:hi].any():
                fa += 1
    hours = windows * SECONDS_PER_WINDOW / 3600
    return {"events": ev_tot, "caught": ev_hit,
            "sens": ev_hit / ev_tot if ev_tot else 0.0,
            "fa_hr": fa / hours if hours else 0.0, "hours": hours}


def baseline_score(blocks, thresh, min_consec=3, k=5):
    """The current approach: moving average, threshold, minimum run length."""
    ev_tot = ev_hit = fa = windows = 0
    for probs, labels in blocks:
        pred = detect(probs, thresh, k, min_consec)
        truth = labels.astype(bool)
        windows += len(pred)
        events = find_runs(truth)
        ev_tot += len(events)
        ev_hit += sum(1 for lo, hi in events if pred[lo:hi].any())
        for lo, hi in find_runs(pred):
            if not truth[lo:hi].any():
                fa += 1
    hours = windows * SECONDS_PER_WINDOW / 3600
    return {"events": ev_tot, "caught": ev_hit,
            "sens": ev_hit / ev_tot if ev_tot else 0.0,
            "fa_hr": fa / hours if hours else 0.0, "hours": hours}


def patient_folds(patients, n_folds=3, seed=42):
    order = list(patients)
    np.random.RandomState(seed).shuffle(order)
    size = len(order) // n_folds
    folds = [order[i * size:(i + 1) * size if i < n_folds - 1 else len(order)]
             for i in range(n_folds)]
    for a, b in SAME_SUBJECT:
        fa = next((i for i, f in enumerate(folds) if a in f), None)
        fb = next((i for i, f in enumerate(folds) if b in f), None)
        if fa is None or fb is None or fa == fb:
            continue
        partner = next(p for p in folds[fa] if p != a)
        folds[fa][folds[fa].index(partner)] = b
        folds[fb][folds[fb].index(b)] = partner
    return folds


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default=CACHE)
    ap.add_argument("--context", type=int, nargs="*", default=[5, 15, 30],
                    help="radius in windows; 15 = +/- 30 s, 30 = +/- 60 s")
    ap.add_argument("--model", choices=["logistic", "gbm"], default="gbm")
    ap.add_argument("--min-consec", type=int, default=3)
    args = ap.parse_args()

    data = np.load(args.cache, allow_pickle=True)["data"].item()
    patients = sorted(data)
    folds = patient_folds(patients)

    print(f"temporal context from {args.cache}: {len(patients)} patients, "
          f"{args.model}")
    print("splits are by patient, chb01/chb21 kept together\n")

    # baseline: the current smoothing rule, at each patient's own threshold
    b = {"caught": 0, "events": 0, "fa": 0.0, "hours": 0.0}
    for p in patients:
        s = baseline_score(data[p]["blocks"], data[p]["t"], args.min_consec)
        b["caught"] += s["caught"]; b["events"] += s["events"]
        b["fa"] += s["fa_hr"] * s["hours"]; b["hours"] += s["hours"]
    print(f"  current smoothing rule : {b['caught']}/{b['events']} = "
          f"{b['caught']/b['events']:.3f} sensitivity at "
          f"{b['fa']/b['hours']:.2f} FA/h\n")

    results = []
    for radius in args.context:
        secs = radius * SECONDS_PER_WINDOW
        per_fold = {}
        for fi, val_p in enumerate(folds):
            train_p = [p for j, f in enumerate(folds) if j != fi for p in f]
            Xtr, ytr = [], []
            for p in train_p:
                X, y = build_features(data[p]["blocks"], radius)
                if X is not None:
                    Xtr.append(X); ytr.append(y)
            X_train = np.concatenate(Xtr); y_train = np.concatenate(ytr)

            if args.model == "gbm":
                clf = HistGradientBoostingClassifier(
                    max_iter=150, learning_rate=0.1, max_depth=4,
                    class_weight="balanced", random_state=42)
            else:
                clf = LogisticRegression(max_iter=2000, class_weight="balanced")
            clf.fit(X_train, y_train)
            predict = lambda M: clf.predict_proba(M)[:, 1]

            for p in val_p:
                per_fold.setdefault(p, (data[p]["blocks"], predict))

        # sweep the threshold so this can be read at the SAME false-alarm rate
        # as the baseline, instead of at an arbitrary 0.5
        curve = []
        for t in np.linspace(0.05, 0.99, 40):
            a = {"caught": 0, "events": 0, "fa": 0.0, "hours": 0.0}
            for p, (blocks, predict) in per_fold.items():
                s = score_sequence(blocks, predict, radius, args.min_consec, t)
                a["caught"] += s["caught"]; a["events"] += s["events"]
                a["fa"] += s["fa_hr"] * s["hours"]; a["hours"] += s["hours"]
            if a["events"]:
                curve.append((a["fa"] / a["hours"], a["caught"] / a["events"], t))

        target_fa = b["fa"] / b["hours"]
        ok = [c for c in curve if c[0] <= target_fa]
        fa, sens, t = max(ok, key=lambda c: c[1]) if ok else min(curve)
        results.append({"radius": radius, "seconds_each_side": secs,
                        "sens": sens, "fa_hr": fa, "threshold": t})
        print(f"  context +/-{secs:5.0f} s : {sens:.3f} sensitivity at "
              f"{fa:.2f} FA/h  (threshold {t:.2f}, matched to baseline)")

    print()
    base_sens, base_fa = b["caught"] / b["events"], b["fa"] / b["hours"]
    best = max(results, key=lambda r: r["sens"] - 0.02 * r["fa_hr"])
    print(f"baseline  {base_sens:.3f} sens @ {base_fa:.2f} FA/h")
    print(f"best      {best['sens']:.3f} sens @ {best['fa_hr']:.2f} FA/h "
          f"(+/-{best['seconds_each_side']:.0f} s)")
    print(f"delta     {best['sens']-base_sens:+.3f} sensitivity, "
          f"{best['fa_hr']-base_fa:+.2f} FA/h")

    out = Path("cv_results") / f"temporal_context_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps(
        {"baseline": {"sens": base_sens, "fa_hr": base_fa}, "context": results},
        indent=2))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
