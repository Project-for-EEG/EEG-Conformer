"""Single-fold refit (fold CHB01_CHB21, Trel_s0, 3 threads) to check that the stored Trel_s0
test predictions and thresholds for CHB01/CHB21 come from the training patients' background via
the repo's own calibrate_per_patient.pick_threshold (not trees_train.thresholds_for).
Also: A0 thresholds for 3 folds from cached logits, same check, no training."""
import os, sys, json, time
import numpy as np
sys.path.insert(0, r"C:\EEG")
from finetune_per_patient import patient_files, find_split, adapt_background_mask
from calibrate_per_patient import pick_threshold
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

T = r"C:\Users\user\AppData\Local\Temp\claude\36c1f29744f93fe6"
FEATS = os.path.join(T, "b3889703-629c-4c5a-b583-69f722743ff4", "scratchpad", "trees", "feats")
OLDOUT = os.path.join(T, "b3889703-629c-4c5a-b583-69f722743ff4", "scratchpad", "trees", "out")
REL = os.path.join(T, "c7848ca6-bfef-414b-a321-d6686c1a43f8", "scratchpad", "clean", "rel")
NEWOUT = os.path.join(T, "c7848ca6-bfef-414b-a321-d6686c1a43f8", "scratchpad", "clean", "trees_out")
PIL = os.path.join(T, "f241281f-add4-41ac-9210-f47dab0b83b0", "scratchpad")
PILOT_FOLDS = ["CHB12", "CHB15", "CHB24", "CHB13", "CHB06", "CHB16"]
TARGETS = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
PARAMS = dict(learning_rate=0.05, max_iter=300, max_leaf_nodes=31, min_samples_leaf=20,
              l2_regularization=1.0, class_weight="balanced", early_stopping=False, random_state=0)
CAP = 2000
mode = sys.argv[1]

folds = sorted(p[5:-3] for p in os.listdir(r"C:\EEG\checkpoints\lopo")
               if p.startswith("lopo_") and p.endswith(".pt"))
cases = sorted(d for d in os.listdir(r"C:\EEG\preprocessed_data")
               if os.path.isdir(os.path.join(r"C:\EEG\preprocessed_data", d)))
files = {c: patient_files(c) for c in cases}
split = {c: find_split(files[c], 1) for c in cases}

def background(train_cases, per_file_pre):
    """per_file_pre[q] = list of prob arrays for files[:fi+1] (last one truncated at wi)."""
    rng = np.random.default_rng(42)
    pooled = []
    for q in train_cases:
        fi, wi = split[q]
        masks = adapt_background_mask(files[q], split[q], 0.3, rng)
        res, ev = [], []
        for i in range(fi + 1):
            with np.load(files[q][i]) as d:
                lab = d["labels"]
            if i == fi:
                lab = lab[:wi]
            p = per_file_pre[q][i]
            assert len(p) == len(lab)
            bg = lab == 0
            res.append(p[masks[i] & bg]); ev.append(p[bg])
        h = np.concatenate(res)
        pooled.append(np.asarray(h if len(h) >= 200 else np.concatenate(ev), dtype=np.float32))
    return np.concatenate(pooled)

def compare(arm, tag, fold, thr_mine):
    TH = json.load(open(os.path.join(NEWOUT if arm.startswith("Trel") else OLDOUT,
                                     "thresholds_%s.json" % tag)))[arm]
    worst = 0.0
    for c in fold.split("_"):
        for tg in TARGETS:
            worst = max(worst, abs(TH[c][str(tg)] - thr_mine[tg]))
    print("  %s fold %s: max |stored - pick_threshold| over 14 targets = %.3g" % (arm, fold, worst))
    print("     stored  ", [round(TH[fold.split('_')[0]][str(t)], 5) for t in TARGETS])
    print("     mine    ", [round(thr_mine[t], 5) for t in TARGETS])

sig = lambda x: 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))

if mode == "a0":
    for fold in folds[:3]:
        held = fold.split("_")
        train_cases = [c for c in cases if c not in held]
        pre = {}
        for q in train_cases:
            fi, wi = split[q]
            pre[q] = []
            for i in range(fi + 1):
                stem = os.path.splitext(os.path.basename(files[q][i]))[0]
                src = os.path.join(PIL, "pilot_cache" if fold in PILOT_FOLDS else "pilot_cache2",
                                   "%s__%s.npz" % (q, stem))
                with np.load(src) as d:
                    p = sig(d["logit_" + fold])
                pre[q].append(p[:wi] if i == fi else p)
        bg = background(train_cases, pre)
        thr = {tg: pick_threshold(bg, tg, 5, 3) for tg in TARGETS}
        print("A0 fold %s: background %d windows" % (fold, len(bg)))
        compare("A0", "a", fold, thr)
    sys.exit()

# ---- Trel_s0 refit of fold 0
t0 = time.time()
fold = folds[0]; fi_ = 0; seed = 0
held = fold.split("_")
assert fold == "CHB01_CHB21"
X, Y = {}, {}
for c in cases:
    xs, ys = [], []
    for f in files[c]:
        stem = os.path.splitext(os.path.basename(f))[0]
        with np.load(os.path.join(FEATS, "%s__%s.npz" % (c, stem))) as d:
            x = d["feats"].copy(); y = d["labels"].astype(np.int8)
        with np.load(os.path.join(REL, "%s__%s.npz" % (c, stem))) as r:
            assert np.array_equal(r["labels"], y)
            x[:, :, :8] = r["rel"]
        xs.append(x.reshape(len(y), -1)); ys.append(y)
    X[c] = xs; Y[c] = ys
print("loaded %.0f s" % (time.time() - t0), flush=True)
train_cases = [c for c in cases if c not in held]
rows, labs = [], []
for j, q in enumerate(train_cases):
    xq = np.concatenate(X[q]); y = np.concatenate(Y[q])
    r = np.random.default_rng([seed, fi_, j])
    pos = np.flatnonzero(y == 1); neg = np.flatnonzero(y == 0)
    n_neg = max(0, CAP - len(pos))
    if n_neg < len(neg):
        neg = r.choice(neg, size=n_neg, replace=False)
    keep = np.sort(np.concatenate([pos, neg]))
    rows.append(xq[keep]); labs.append(y[keep])
Xtr = np.concatenate(rows); ytr = np.concatenate(labs)
print("train rows %d (%d seizure)" % (len(ytr), ytr.sum()), flush=True)
with threadpool_limits(3):
    clf = HistGradientBoostingClassifier(**PARAMS).fit(Xtr, ytr)
    print("fit %.0f s" % (time.time() - t0), flush=True)
    pre = {}
    for q in train_cases:
        fi, wi = split[q]
        pre[q] = []
        for i in range(fi + 1):
            x = X[q][i][:wi] if i == fi else X[q][i]
            pre[q].append(clf.predict_proba(x)[:, 1] if len(x) else np.zeros(0))
    # test blocks of held-out cases
    mine_test = {}
    for c in held:
        fi, wi = split[c]
        blocks = []
        for i in range(fi, len(files[c])):
            x, y = X[c][i], Y[c][i]
            if i == fi:
                x, y = x[wi:], y[wi:]
            if len(y) >= 10:
                blocks.append((clf.predict_proba(x)[:, 1], y))
        mine_test[c] = blocks
bg = background(train_cases, pre)
thr = {tg: pick_threshold(bg, tg, 5, 3) for tg in TARGETS}
print("background %d windows, %.0f s" % (len(bg), time.time() - t0))
P = np.load(os.path.join(NEWOUT, "preds_rel0.npy"), allow_pickle=True).item()["Trel_s0"]
for c in held:
    assert len(P[c]) == len(mine_test[c]), (c, len(P[c]), len(mine_test[c]))
    md = max(float(np.max(np.abs(a[0] - b[0]))) for a, b in zip(P[c], mine_test[c]))
    lab_ok = all(np.array_equal(a[1], b[1]) for a, b in zip(P[c], mine_test[c]))
    print("  Trel_s0 %s test preds: %d blocks, max |stored - refit| = %.3g, labels equal %s" % (
        c, len(P[c]), md, lab_ok))
compare("Trel_s0", "rel0", fold, thr)
