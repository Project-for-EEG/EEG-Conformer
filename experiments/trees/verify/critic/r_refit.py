"""critic spot-check refit (CPU, 3 threads). One fold, one arm:
  Tamp_s0  exact replica of trees_train.py's Tamp_s0 for that fold (same rows, same params)
  Trel_s0  same rows and params, but the 8 size features of every window are replaced by
           log(feature) minus that recording's per-channel median of log(feature)
           (a per-recording gain normalisation; uses the whole recording, so not causal)
Predicts EVERY window of every case and saves per-file probabilities plus, per training
patient, a per-file mask of the windows that were in the training sample.
Run with cwd C:\\EEG. Writes only into verify/critic."""
import argparse
import sys
import time
sys.dont_write_bytecode = True
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
sys.path.insert(0, r"C:\EEG")
from finetune_per_patient import patient_files  # noqa: E402

TREES = Path(r"C:\Users\user\AppData\Local\Temp\claude\36c1f29744f93fe6"
             r"\b3889703-629c-4c5a-b583-69f722743ff4\scratchpad\trees")
FE, ME = TREES / "feats", TREES / "verify" / "critic"
REPO = Path(r"C:\EEG")
PARAMS = dict(learning_rate=0.05, max_iter=300, max_leaf_nodes=31, min_samples_leaf=20,
              l2_regularization=1.0, class_weight="balanced", early_stopping=False,
              random_state=0)
CAP = 2000


def load(c, f, arm):
    with np.load(FE / ("%s__%s.npz" % (c, f.stem))) as d:
        X, y = d["feats"].astype(np.float32), d["labels"].astype(np.int8)
    if arm.startswith("Trel"):
        with np.errstate(divide="ignore", invalid="ignore"):
            lg = np.log(X[:, :, :8])
        lg[~np.isfinite(lg)] = np.nan
        med = np.nanmedian(lg, axis=0, keepdims=True) if np.isfinite(lg).any() else 0.0
        X = X.copy()
        X[:, :, :8] = lg - med
    return X.reshape(len(y), -1), y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", required=True)
    ap.add_argument("--arm", required=True, choices=["Tamp_s0", "Trel_s0"])
    ap.add_argument("--threads", type=int, default=3)
    a = ap.parse_args()
    t0 = time.time()
    folds = [p.stem[len("lopo_"):] for p in sorted((REPO / "checkpoints" / "lopo").glob("lopo_*.pt"))]
    fi_ = folds.index(a.fold)
    held = a.fold.split("_")
    cases = sorted(d.name for d in (REPO / "preprocessed_data").iterdir() if d.is_dir())
    train_cases = [c for c in cases if c not in held]
    files = {c: patient_files(c) for c in cases}
    rows, labs, intrain = [], [], {}
    for j, q in enumerate(train_cases):
        parts = [load(q, f, a.arm) for f in files[q]]
        y = np.concatenate([p[1] for p in parts])
        r = np.random.default_rng([0, fi_, j])
        pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
        n_neg = max(0, CAP - len(pos))
        if n_neg < len(neg):
            neg = r.choice(neg, size=n_neg, replace=False)
        keep = np.sort(np.concatenate([pos, neg]))
        X = np.concatenate([p[0] for p in parts])
        rows.append(X[keep])
        labs.append(y[keep])
        m = np.zeros(len(y), bool)
        m[keep] = True
        off = np.cumsum([0] + [len(p[1]) for p in parts])
        intrain[q] = [m[off[i]:off[i + 1]] for i in range(len(parts))]
        del X, parts
    Xtr, ytr = np.concatenate(rows), np.concatenate(labs)
    del rows
    print("fold %s arm %s: %d rows (%d seizure), %d features, load %.0f s"
          % (a.fold, a.arm, len(ytr), ytr.sum(), Xtr.shape[1], time.time() - t0), flush=True)
    with threadpool_limits(a.threads):
        clf = HistGradientBoostingClassifier(**PARAMS).fit(Xtr, ytr)
        del Xtr
        print("fit %.0f s" % (time.time() - t0), flush=True)
        probs = {}
        for c in cases:
            probs[c] = [clf.predict_proba(load(c, f, a.arm)[0])[:, 1].astype(np.float32)
                        for f in files[c]]
    np.save(ME / ("refit_%s_%s.npy" % (a.arm, a.fold)),
            np.array({"probs": probs, "intrain": intrain, "fold": a.fold, "arm": a.arm},
                     dtype=object), allow_pickle=True)
    print("done %.0f s" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
