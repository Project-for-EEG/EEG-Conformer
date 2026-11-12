"""
Tree test, step 2: train the trees per fold, set deployable thresholds for every arm
(including today's model, A0) with uncalibrated_lopo.py's procedure, and save the test
predictions. Rule: tree_rule.txt. CPU only. Run with cwd C:\\EEG.
Copied to clean/ for clean_rule.txt v2: adds Trel_s{0,1,2} and Trelf_s0 (size features
replaced by NEW\\rel relative features: running 300-window median / whole-file median).
A process runs either Trel arms, or Trelf arms, or neither. Existing arms are unchanged.

Arms: A0 (cached lopo_ fold-model logits, no retraining), Tamp_s{0,1,2}, Tfree_s{0,1,2},
Tall (Tamp on every training window, one fit). --arms picks a subset so several
processes can share the work; each writes its own files tagged --tag:
<out>/preds_<tag>.npy (arm -> case -> [(prob, labels)] test blocks),
<out>/thresholds_<tag>.json (arm -> case -> {target: threshold}), train_log_<tag>.json.
Only the windows that are used are predicted: training patients' pre-split windows
(calibration) and the held-out cases' post-split windows (test).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

sys.dont_write_bytecode = True
REPO = Path(r"C:\EEG")
sys.path.insert(0, str(REPO))
from finetune_per_patient import patient_files, find_split, adapt_background_mask  # noqa: E402
from calibrate_per_patient import fa_per_hour_on_background                         # noqa: E402

OLD = Path(r"C:\Users\user\AppData\Local\Temp\claude\36c1f29744f93fe6"
           r"\f241281f-add4-41ac-9210-f47dab0b83b0\scratchpad")
CACHE1, CACHE2 = OLD / "pilot_cache", OLD / "pilot_cache2"
PILOT_FOLDS = ["CHB12", "CHB15", "CHB24", "CHB13", "CHB06", "CHB16"]
TARGETS = (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0)
K, N_RUN, HOLDOUT, MIN_SAMPLE, CAP = 5, 3, 0.3, 200, 2000
N_FEAT = 19
SHAPE = np.arange(8, 19)          # nll, 6 rel, zc, zcd, kurt, skew
PARAMS = dict(learning_rate=0.05, max_iter=300, max_leaf_nodes=31, min_samples_leaf=20,
              l2_regularization=1.0, class_weight="balanced", early_stopping=False,
              random_state=0)
ALL_ARMS = ["A0", "Tamp_s0", "Tamp_s1", "Tamp_s2", "Tfree_s0", "Tfree_s1", "Tfree_s2", "Tall",
            "Trel_s0", "Trel_s1", "Trel_s2", "Trelf_s0", "N1"]


def sig(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


def thresholds_for(bg, targets=TARGETS):
    """pick_threshold(bg, target, K, N_RUN) for every target, from one pass over its grid.
    Same grid, same order, same acceptance test as calibrate_per_patient.pick_threshold."""
    grid = np.unique(np.concatenate([np.linspace(0.001, 0.999, 400),
                                     np.quantile(bg, np.linspace(0.90, 0.99999, 200))]))
    grid.sort()
    out, todo = {}, sorted(targets)
    for t in grid:
        if not todo:
            break
        fa, flagged = fa_per_hour_on_background(bg, t, K, N_RUN)
        for tg in list(todo):
            if fa <= tg and flagged <= 0.02:
                out[tg] = float(t)
                todo.remove(tg)
    for tg in todo:
        out[tg] = 1.0
    return out


def arm_spec(arm):
    """(feature columns or None for all, sampling seed or -1 for every window)."""
    if arm == "Tall":
        return None, -1
    kind, s = arm.split("_s")
    return (SHAPE if kind == "Tfree" else None), int(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--rel", default=None, help="NEW\\rel, needed for Trel/Trelf arms")
    ap.add_argument("--n1-cache", default=None, help="NEW\\n1_cache, needed for the N1 arm")
    args = ap.parse_args()
    arms = args.arms.split(",")
    assert all(a in ALL_ARMS for a in arms), arms
    rel_key = None
    if any(a.startswith("Trel") for a in arms):
        if all(a.startswith("Trel_") for a in arms):
            rel_key = "rel"
        elif all(a.startswith("Trelf_") for a in arms):
            rel_key = "relf"
        else:
            raise SystemExit("a process runs only Trel arms, only Trelf arms, or neither")
        assert args.rel, "--rel is required for Trel/Trelf arms"
    if "N1" in arms:
        assert args.n1_cache, "--n1-cache is required for the N1 arm"
    feats_dir, out = Path(args.feats), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    folds = {}
    for p in sorted((REPO / "checkpoints" / "lopo").glob("lopo_*.pt")):
        name = p.stem[len("lopo_"):]
        folds[name] = name.split("_")
    cases = sorted(d.name for d in (REPO / "preprocessed_data").iterdir() if d.is_dir())
    assert len(folds) == 23 and sorted(c for h in folds.values() for c in h) == cases

    # ---- per case: features (all files concatenated), labels, file offsets, logits ---
    files, X, Y, off, splits, LG, LG1 = {}, {}, {}, {}, {}, {}, {}
    for c in cases:
        files[c] = patient_files(c)
        splits[c] = find_split(files[c], 1)
        assert splits[c] is not None
        xs, ys, lgs, lgs1 = [], [], [], []
        for f in files[c]:
            with np.load(feats_dir / ("%s__%s.npz" % (c, f.stem))) as d:
                y = d["labels"].astype(np.int8)
                if rel_key is None:
                    xs.append(d["feats"].reshape(len(y), -1))
                else:
                    x = d["feats"].copy()
                    with np.load(Path(args.rel) / ("%s__%s.npz" % (c, f.stem))) as r:
                        assert np.array_equal(r["labels"], d["labels"]), (c, f.stem)
                        x[:, :, :8] = r[rel_key]
                    xs.append(x.reshape(len(y), -1))
            ys.append(y)
            if "A0" in arms:
                lg = {}
                with np.load(CACHE1 / ("%s__%s.npz" % (c, f.stem))) as d:
                    assert np.array_equal(d["labels"], y), (c, f.stem)
                    for p in PILOT_FOLDS:
                        lg[p] = d["logit_" + p]
                with np.load(CACHE2 / ("%s__%s.npz" % (c, f.stem))) as d:
                    assert np.array_equal(d["labels"], y), (c, f.stem)
                    for p in folds:
                        if p not in PILOT_FOLDS:
                            lg[p] = d["logit_" + p]
                lgs.append(lg)
            if "N1" in arms:
                with np.load(Path(args.n1_cache) / ("%s__%s.npz" % (c, f.stem))) as d:
                    assert np.array_equal(d["labels"], y), (c, f.stem)
                    lgs1.append({p: d["logit_" + p] for p in folds})
        X[c] = np.concatenate(xs)
        Y[c] = ys
        off[c] = np.cumsum([0] + [len(y) for y in ys])
        LG[c] = lgs
        LG1[c] = lgs1
        del xs
    print("[%s] loaded %d cases, %d windows, %d features, arms %s, %d threads, "
          "size features %s, %.0f s"
          % (args.tag, len(cases), sum(len(X[c]) for c in cases), X[cases[0]].shape[1],
             ",".join(arms), args.threads, rel_key or "raw", time.time() - t0), flush=True)

    def pre_rows(c):
        fi, wi = splits[c]
        return np.arange(off[c][fi] + wi)               # every window before the split

    def post_rows(c):
        fi, wi = splits[c]
        return np.arange(off[c][fi] + wi, off[c][-1])

    def to_files(c, rows, p):
        """Scatter predictions for `rows` back into per-file arrays (NaN elsewhere)."""
        full = np.full(off[c][-1], np.nan)
        full[rows] = p
        return [full[off[c][i]:off[c][i + 1]] for i in range(len(files[c]))]

    def test_blocks(c, per_file):
        fi, wi = splits[c]
        res = []
        for i, (p, y) in enumerate(zip(per_file, Y[c])):
            if i < fi:
                continue
            if i == fi:
                p, y = p[wi:], y[wi:]
            if len(y) >= 10:
                assert not np.isnan(p).any()
                res.append((np.asarray(p, np.float64), y))
        return res

    def calib_bg(train_cases, masks, per_file):
        """training_fold_background + adapt_background_probs, on given per-file probs."""
        pooled = []
        for q in train_cases:
            fi, wi = splits[q]
            reserved, everything = [], []
            for i in range(fi + 1):
                p, y = per_file[q][i], Y[q][i]
                if i == fi:
                    p, y = p[:wi], y[:wi]
                assert not np.isnan(p).any()
                bg = y == 0
                reserved.append(p[masks[q][i] & bg])
                everything.append(p[bg])
            held = np.concatenate(reserved)
            pooled.append(np.asarray(held if len(held) >= MIN_SAMPLE
                                     else np.concatenate(everything), dtype=np.float32))
        return np.concatenate(pooled)

    preds = {a: {} for a in arms}
    thr = {a: {} for a in arms}
    log = []
    with threadpool_limits(args.threads):
        for fi_, (name, held) in enumerate(folds.items()):
            train_cases = [c for c in cases if c not in held]
            rng = np.random.default_rng(42)                  # as training_fold_background
            masks = {q: adapt_background_mask(files[q], splits[q], HOLDOUT, rng)
                     for q in train_cases}
            for arm in arms:
                tf = time.time()
                if arm in ("A0", "N1"):
                    src_lg = LG if arm == "A0" else LG1
                    per_file = {c: [sig(lg[name]) for lg in src_lg[c]] for c in cases}
                    n_tr = n_pos = n_ft = 0
                    train_auc = float("nan")
                else:
                    cols, seed = arm_spec(arm)
                    rows, labs = [], []
                    for j, q in enumerate(train_cases):
                        y = np.concatenate(Y[q])
                        if seed < 0:
                            keep = np.arange(len(y))
                        else:
                            r = np.random.default_rng([seed, fi_, j])
                            pos = np.flatnonzero(y == 1)
                            neg = np.flatnonzero(y == 0)
                            n_neg = max(0, CAP - len(pos))
                            if n_neg < len(neg):
                                neg = r.choice(neg, size=n_neg, replace=False)
                            keep = np.sort(np.concatenate([pos, neg]))
                        rows.append(X[q][keep])
                        labs.append(y[keep])
                    Xtr = np.concatenate(rows)
                    ytr = np.concatenate(labs)
                    del rows

                    def sel(x):
                        if cols is None:
                            return x
                        return x.reshape(len(x), 23, N_FEAT)[:, :, cols].reshape(len(x), -1)
                    Xtr = sel(Xtr)
                    clf = HistGradientBoostingClassifier(**PARAMS).fit(Xtr, ytr)
                    sub = np.random.default_rng(0).choice(len(ytr), min(len(ytr), 50000),
                                                          replace=False)
                    train_auc = float(roc_auc_score(ytr[sub], clf.predict_proba(Xtr[sub])[:, 1]))
                    n_tr, n_pos, n_ft = len(ytr), int(ytr.sum()), Xtr.shape[1]
                    del Xtr
                    per_file = {}
                    for c in cases:
                        rws = post_rows(c) if c in held else pre_rows(c)
                        p = clf.predict_proba(sel(X[c][rws]))[:, 1] if len(rws) else np.zeros(0)
                        per_file[c] = to_files(c, rws, p)
                t = thresholds_for(calib_bg(train_cases, masks, per_file))
                for c in held:
                    preds[arm][c] = test_blocks(c, per_file[c])
                    thr[arm][c] = t
                log.append({"fold": name, "arm": arm, "n_train": n_tr, "n_pos": n_pos,
                            "n_features": n_ft, "train_auc": train_auc,
                            "seconds": round(time.time() - tf, 1), "t_2fa": t[2.0]})
                print("[%s] %-12s %-9s %6d rows (%5d seizure) %3d feats  train AUC %.4f  "
                      "t@2FA/h %.4f  %5.1f s  [%.0f min]"
                      % (args.tag, name, arm, n_tr, n_pos, n_ft, train_auc, t[2.0],
                         time.time() - tf, (time.time() - t0) / 60), flush=True)

    np.save(out / ("preds_%s.npy" % args.tag), np.array(preds, dtype=object), allow_pickle=True)
    (out / ("thresholds_%s.json" % args.tag)).write_text(json.dumps(
        {a: {c: {str(k): v for k, v in d.items()} for c, d in thr[a].items()} for a in thr},
        indent=1))
    (out / ("train_log_%s.json" % args.tag)).write_text(json.dumps(log, indent=1))
    print("[%s] done in %.0f min" % (args.tag, (time.time() - t0) / 60))


if __name__ == "__main__":
    main()
