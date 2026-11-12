"""Refit ONE fold's Tamp_s0 and Tfree_s0 with my own code (tree_rule.txt + the sampling
recipe of train_memory_efficient.load_patient_data, seeded as trees_train.py documents:
default_rng([seed, fold_index, train_patient_index])), rebuild that fold's calibration
background (rng 42, 30% reserved, min 200 fallback), run the repo's pick_threshold at the 14
targets and compare thresholds and test predictions with out/thresholds_a.json and
out/preds_a.npy. Run with cwd C:\\EEG, python -B. Usage: rs_refit.py FOLD"""
import sys
sys.dont_write_bytecode = True
import time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, r"C:\EEG")
from rs_common import load_all, runs, TARGETS, HERE, TREES
from calibrate_per_patient import pick_threshold

FOLD = sys.argv[1] if len(sys.argv) > 1 else "CHB06"
PREP = Path(r"C:\EEG\preprocessed_data")
FEATS = TREES / "feats"
PARAMS = dict(learning_rate=0.05, max_iter=300, max_leaf_nodes=31, min_samples_leaf=20,
              l2_regularization=1.0, class_weight="balanced", early_stopping=False,
              random_state=0)
SHAPE_NAMES = ["nll", "rel_A5", "rel_D5", "rel_D4", "rel_D3", "rel_D2", "rel_D1", "zc", "zcd",
               "kurt", "skew"]
lines = []


def say(s=""):
    print(s, flush=True)
    lines.append(s)


t0 = time.time()
folds = [p.stem[len("lopo_"):] for p in sorted(Path(r"C:\EEG\checkpoints\lopo").glob("lopo_*.pt"))]
fold_index = folds.index(FOLD)
held = FOLD.split("_")
cases = sorted(d.name for d in PREP.iterdir() if d.is_dir())
train = [c for c in cases if c not in held]
say("fold %s (index %d of %d), held %s, %d training cases" % (FOLD, fold_index, len(folds),
                                                              held, len(train)))

files, labs = {}, {}
for c in cases:
    files[c] = sorted((PREP / c).glob("*.npz"))
    labs[c] = []
    for f in files[c]:
        with np.load(f) as d:
            labs[c].append(d["labels"].astype(np.int64))


def my_split(c):
    for fi, y in enumerate(labs[c]):
        lo, hi = runs(y.astype(bool))
        if len(lo):
            h = int(hi[0])
            rem = int(y[h:].sum()) + sum(int(z.sum()) for z in labs[c][fi + 1:])
            return (fi, h) if rem > 0 else None


split = {c: my_split(c) for c in cases}


def load_feats(c):
    xs, names = [], None
    for f, y in zip(files[c], labs[c]):
        with np.load(FEATS / ("%s__%s.npz" % (c, f.stem))) as d:
            assert np.array_equal(d["labels"].astype(np.int64), y), (c, f.stem)
            xs.append(d["feats"])
            names = list(d["names"])
    return np.concatenate(xs), names


# training rows, calibration (pre-split) rows, test (post-split) rows
Xtr, ytr, PRE, POST = [], [], {}, {}
names = None
for c in cases:
    X, names = load_feats(c)
    y = np.concatenate(labs[c])
    off = np.cumsum([0] + [len(z) for z in labs[c]])
    fi, wi = split[c]
    cut = off[fi] + wi
    if c in train:
        j = train.index(c)
        r = np.random.default_rng([0, fold_index, j])
        pos = np.flatnonzero(y == 1)
        neg = np.flatnonzero(y == 0)
        n_neg = max(0, 2000 - len(pos))
        if n_neg < len(neg):
            neg = r.choice(neg, size=n_neg, replace=False)
        keep = np.sort(np.concatenate([pos, neg]))
        Xtr.append(X[keep])
        ytr.append(y[keep])
        PRE[c] = X[:cut]
    else:
        POST[c] = (X[cut:], off, cut)
    del X
Xtr = np.concatenate(Xtr)
ytr = np.concatenate(ytr)
shape_cols = [names.index(n) for n in SHAPE_NAMES]
say("training rows %d (%d seizure); feature names %s; shape cols %s; load %.0f s"
    % (len(ytr), ytr.sum(), names, shape_cols, time.time() - t0))

P, T = load_all()


def cal_background(prob_pre):
    rng = np.random.default_rng(42)
    pooled = []
    for q in train:
        fi, wi = split[q]
        off = np.cumsum([0] + [len(z) for z in labs[q]])
        res, allbg = [], []
        for i in range(fi + 1):
            y = labs[q][i][:wi] if i == fi else labs[q][i]
            m = np.zeros(len(y), bool)
            neg = np.where(y == 0)[0]
            if len(neg):
                m[rng.choice(neg, size=int(len(neg) * 0.3), replace=False)] = True
            p = prob_pre[q][off[i]:off[i] + len(y)]
            res.append(p[m & (y == 0)])
            allbg.append(p[y == 0])
        r_ = np.concatenate(res)
        pooled.append(r_ if len(r_) >= 200 else np.concatenate(allbg))
    return np.concatenate(pooled).astype(np.float32)


with threadpool_limits(3):
    for arm, cols in (("Tamp_s0", None), ("Tfree_s0", shape_cols)):
        ta = time.time()

        def sel(x):
            x = x if cols is None else x[:, :, cols]
            return x.reshape(len(x), -1)

        clf = HistGradientBoostingClassifier(**PARAMS).fit(sel(Xtr), ytr)
        prob_pre = {q: clf.predict_proba(sel(PRE[q]))[:, 1] for q in train}
        bg = cal_background(prob_pre)
        mine = {tg: pick_threshold(bg, float(tg), 5, 3) for tg in TARGETS}
        c0 = held[0]
        dt = {tg: mine[tg] - T[arm][c0][tg] for tg in TARGETS}
        say("%s: %d features, fit+predict %.0f s, calibration background %d windows"
            % (arm, sel(Xtr[:1]).shape[1], time.time() - ta, len(bg)))
        say("  thresholds target: mine / saved")
        for tg in TARGETS:
            say("    %-5s %.6f / %.6f  (d %+.2g)" % (tg, mine[tg], T[arm][c0][tg], dt[tg]))
        say("  max|d threshold| %.3g" % max(abs(v) for v in dt.values()))
        for c in held:
            Xp, off, cut = POST[c]
            p = clf.predict_proba(sel(Xp))[:, 1]
            fi, wi = split[c]
            blocks = []
            pos = 0
            for i in range(fi, len(files[c])):
                n = len(labs[c][i]) - (wi if i == fi else 0)
                if n >= 10:
                    blocks.append((p[pos:pos + n], labs[c][i][-n:]))
                pos += n
            assert len(blocks) == len(P[arm][c])
            worst = 0.0
            for (pm, ym), (ps, ys) in zip(blocks, P[arm][c]):
                assert np.array_equal(ym, np.asarray(ys, np.int64))
                worst = max(worst, float(np.abs(pm - ps).max()))
            say("  test predictions %s: %d blocks, %d windows, max|p_mine - p_saved| %.3g"
                % (c, len(blocks), sum(len(b[1]) for b in blocks), worst))
say("total %.0f s" % (time.time() - t0))
(HERE / ("rs_refit_%s_out.txt" % FOLD)).write_text("\n".join(lines) + "\n", encoding="ascii")
