"""
Audit task 2/4: one independent refit of Trel_s0 for one fold, built so that the held-out
patient's features are NOT LOADED until the model is fitted and every threshold is fixed.
Then the held-out post-split windows are predicted and compared with the stored
NEW\\trees_out\\preds_rel0.npy / thresholds_rel0.json. A match means the stored test
predictions and thresholds are what a test-patient-blind pipeline produces.

rel is recomputed here from OLD\\feats with rel_feats.py's production functions (proved
bitwise equal to NEW\\rel and to an explicit nanmedian loop by a1_causal.py); the stored
NEW\\rel files are not read. Thresholds use calibrate_per_patient.pick_threshold directly
(the original routine), not trees_train.thresholds_for.
Extra: overlap between calibration windows and training rows, and the 2 FA/h threshold
recomputed on calibration windows the trees never trained on.
"""
import json
import os
import sys
import time

sys.dont_write_bytecode = True
import numpy as np  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

REPO = r"C:\EEG"
CLEAN = r"C:\Users\user\AppData\Local\Temp\claude\36c1f29744f93fe6\c7848ca6-bfef-414b-a321-d6686c1a43f8\scratchpad\clean"
OLDF = r"C:\Users\user\AppData\Local\Temp\claude\36c1f29744f93fe6\b3889703-629c-4c5a-b583-69f722743ff4\scratchpad\trees\feats"
sys.path.insert(0, REPO)
sys.path.insert(1, CLEAN)
from finetune_per_patient import patient_files, find_split, adapt_background_mask  # noqa: E402
from calibrate_per_patient import pick_threshold  # noqa: E402
import rel_feats  # noqa: E402

HOLD = sys.argv[1] if len(sys.argv) > 1 else "CHB15"
SEED = 0
TARGETS = (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0)
PARAMS = dict(learning_rate=0.05, max_iter=300, max_leaf_nodes=31, min_samples_leaf=20,
              l2_regularization=1.0, class_weight="balanced", early_stopping=False,
              random_state=0)
CAP = 2000


def case_matrix(c, files):
    xs, ys = [], []
    for f in files:
        with np.load(os.path.join(OLDF, "%s__%s.npz" % (c, f.stem))) as d:
            feats, y = d["feats"], d["labels"].astype(np.int8)
        lg = rel_feats.log_size(feats)
        x = feats.copy()
        x[:, :, :8] = (lg - rel_feats.running_median(lg)).astype(np.float32)
        xs.append(x.reshape(len(y), -1))
        ys.append(y)
    return np.concatenate(xs), ys


def main():
    t0 = time.time()
    lopo = sorted(f for f in os.listdir(os.path.join(REPO, "checkpoints", "lopo"))
                  if f.startswith("lopo_") and f.endswith(".pt"))
    folds = [f[len("lopo_"):-3] for f in lopo]
    fold_idx = folds.index(HOLD)
    held = HOLD.split("_")
    cases = sorted(d for d in os.listdir(os.path.join(REPO, "preprocessed_data"))
                   if os.path.isdir(os.path.join(REPO, "preprocessed_data", d)))
    train_cases = [c for c in cases if c not in held]
    print("fold %s (index %d of %d), held %s, %d training cases"
          % (HOLD, fold_idx, len(folds), held, len(train_cases)))

    files = {c: patient_files(c) for c in cases}
    splits = {c: find_split(files[c], 1) for c in cases}

    # ---- training patients only
    X, Y, off = {}, {}, {}
    for q in train_cases:
        X[q], Y[q] = case_matrix(q, files[q])
        off[q] = np.cumsum([0] + [len(y) for y in Y[q]])
    assert not any(h in X for h in held)
    rows, labs, rowid = [], [], {}
    for j, q in enumerate(train_cases):
        y = np.concatenate(Y[q])
        r = np.random.default_rng([SEED, fold_idx, j])
        pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
        n_neg = max(0, CAP - len(pos))
        if n_neg < len(neg):
            neg = r.choice(neg, size=n_neg, replace=False)
        keep = np.sort(np.concatenate([pos, neg]))
        rowid[q] = keep
        rows.append(X[q][keep])
        labs.append(y[keep])
    Xtr, ytr = np.concatenate(rows), np.concatenate(labs)
    print("training rows %d (%d seizure), %d features" % (len(ytr), ytr.sum(), Xtr.shape[1]))
    with threadpool_limits(3):
        clf = HistGradientBoostingClassifier(**PARAMS).fit(Xtr, ytr)
        print("fitted in %.0f s" % (time.time() - t0))

        # ---- thresholds from training patients' pre-split background only
        rng = np.random.default_rng(42)
        masks = {q: adapt_background_mask(files[q], splits[q], 0.3, rng) for q in train_cases}
        pooled, pooled_unseen = [], []
        n_cal = n_cal_seen = 0
        for q in train_cases:
            fi, wi = splits[q]
            pre = np.arange(off[q][fi] + wi)
            p_pre = clf.predict_proba(X[q][pre])[:, 1]
            seen = np.zeros(off[q][-1], bool)
            seen[rowid[q]] = True
            reserved, everything, unseen = [], [], []
            for i in range(fi + 1):
                lo, hi = off[q][i], off[q][i + 1]
                p, y, s = p_pre[lo:min(hi, len(pre))], Y[q][i], seen[lo:hi]
                if i == fi:
                    y, s = y[:wi], s[:wi]
                bg = y == 0
                m = masks[q][i] & bg
                reserved.append(p[m])
                everything.append(p[bg])
                unseen.append(p[m & ~s])
                n_cal += int(m.sum())
                n_cal_seen += int((m & s).sum())
            held_p = np.concatenate(reserved)
            pooled.append(np.asarray(held_p if len(held_p) >= 200 else np.concatenate(everything),
                                     np.float32))
            u = np.concatenate(unseen)
            pooled_unseen.append(np.asarray(u if len(u) >= 200 else np.concatenate(everything),
                                            np.float32))
        cal = np.concatenate(pooled)
        thr = {t: pick_threshold(cal, t, 5, 3) for t in TARGETS}
        thr_unseen2 = pick_threshold(np.concatenate(pooled_unseen), 2.0, 5, 3)
        thr_unseen005 = pick_threshold(np.concatenate(pooled_unseen), 0.05, 5, 3)
        print("calibration windows %d, of which also training rows %d (%.1f%%)"
              % (n_cal, n_cal_seen, 100 * n_cal_seen / n_cal))
        print("thresholds fixed BEFORE any %s data is loaded  [%.0f s]" % (held, time.time() - t0))

        # ---- only now: the held-out patient, post-split windows only
        stored = np.load(os.path.join(CLEAN, "trees_out", "preds_rel0.npy"),
                         allow_pickle=True).item()["Trel_s0"]
        sthr = json.load(open(os.path.join(CLEAN, "trees_out", "thresholds_rel0.json")))["Trel_s0"]
        for c in held:
            Xc, Yc = case_matrix(c, files[c])
            offc = np.cumsum([0] + [len(y) for y in Yc])
            fi, wi = splits[c]
            post = np.arange(offc[fi] + wi, offc[-1])
            pp = clf.predict_proba(Xc[post])[:, 1]
            full = np.full(offc[-1], np.nan)
            full[post] = pp
            blocks = []
            for i in range(fi, len(files[c])):
                p, y = full[offc[i]:offc[i + 1]], Yc[i]
                if i == fi:
                    p, y = p[wi:], y[wi:]
                if len(y) >= 10:
                    blocks.append((p, y))
            assert len(blocks) == len(stored[c])
            dmax = max(float(np.max(np.abs(p - np.asarray(sp)))) for (p, _), (sp, _) in
                       zip(blocks, stored[c]))
            leq = all(np.array_equal(y, np.asarray(sy)) for (_, y), (_, sy) in zip(blocks, stored[c]))
            tmax = max(abs(thr[t] - sthr[c][str(t)]) for t in TARGETS)
            print("%s: %d test blocks, %d post-split windows; labels equal %s; "
                  "max|refit prob - stored prob| %.3e; max|refit thr - stored thr| over 14 targets %.3e"
                  % (c, len(blocks), len(post), leq, dmax, tmax))
            print("   stored thr 2.0 %.6f, refit %.6f; from calibration windows never trained on: %.6f"
                  % (sthr[c]["2.0"], thr[2.0], thr_unseen2))
            print("   stored thr 0.05 %.6f, refit %.6f; from never-trained-on windows: %.6f"
                  % (sthr[c]["0.05"], thr[0.05], thr_unseen005))
            # how many of the held-out test windows have a rel history reaching into pre-split
            hist_pre = int(np.sum((post - offc[fi] < wi + 299) & (post < offc[fi + 1])))
            print("   test windows whose 300-window rel history includes pre-split windows of %s: %d"
                  " (split file %s, split at window %d)" % (c, hist_pre, files[c][fi].stem, wi))
    print("done in %.0f s" % (time.time() - t0))


if __name__ == "__main__":
    main()
