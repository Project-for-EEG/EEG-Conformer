"""Audit step 4: refit Tamp_s0 and Tfree_s0 for fold CHB01_CHB21 (2 tree fits, 3 threads).
Checks: same training windows for both arms; reproduces saved test preds and all 14 saved
thresholds; pick_threshold per target == thresholds_for on the tree's calibration
background; no held-out rows in training or calibration; and how much in-sample
calibration windows (also in the tree's training sample) shift the thresholds.
Memory-lean: only the rows that are used are loaded. Read-only outside this folder."""
import json
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np

REPO = Path(r"C:\EEG")
TR = Path(r"C:\Users\user\AppData\Local\Temp\claude\36c1f29744f93fe6"
          r"\b3889703-629c-4c5a-b583-69f722743ff4\scratchpad\trees")
OUT = TR / "verify" / "audit"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(TR))
from threadpoolctl import threadpool_limits                                 # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier                 # noqa: E402
from finetune_per_patient import patient_files, find_split, adapt_background_mask  # noqa
from calibrate_per_patient import pick_threshold                             # noqa: E402
from score_official import count                                             # noqa: E402
import trees_train as tt                                                     # noqa: E402

FEATS = TR / "feats"
NAME, FOLD_IDX, HELD = "CHB01_CHB21", 0, ["CHB01", "CHB21"]


def main():
    t0 = time.time()
    folds = sorted(p.stem[5:] for p in (REPO / "checkpoints" / "lopo").glob("lopo_*.pt"))
    assert folds[FOLD_IDX] == NAME
    cases = sorted(d.name for d in (REPO / "preprocessed_data").iterdir() if d.is_dir())
    train = [c for c in cases if c not in HELD]
    files = {c: patient_files(c) for c in cases}
    splits = {c: find_split(files[c], 1) for c in cases}
    Y = {}
    for c in cases:
        Y[c] = []
        for f in files[c]:
            with np.load(FEATS / ("%s__%s.npz" % (c, f.stem))) as d:
                Y[c].append(d["labels"].astype(np.int8))
    off = {c: np.cumsum([0] + [len(y) for y in Y[c]]) for c in cases}

    def rows_of(c, rows):
        """feature rows (global row index within case c) -> (len(rows), 437)."""
        out = np.empty((len(rows), 437), np.float32)
        for i, f in enumerate(files[c]):
            lo, hi = off[c][i], off[c][i + 1]
            m = (rows >= lo) & (rows < hi)
            if m.any():
                with np.load(FEATS / ("%s__%s.npz" % (c, f.stem))) as d:
                    F = d["feats"].reshape(hi - lo, -1)
                out[m] = F[rows[m] - lo]
        return out

    # training sample, as trees_train.py lines 195-208, done twice to show both arms agree
    def keep_rows(seed):
        ks = {}
        for j, q in enumerate(train):
            y = np.concatenate(Y[q])
            r = np.random.default_rng([seed, FOLD_IDX, j])
            pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
            n_neg = max(0, tt.CAP - len(pos))
            if n_neg < len(neg):
                neg = r.choice(neg, size=n_neg, replace=False)
            ks[q] = np.sort(np.concatenate([pos, neg]))
        return ks
    k1, k2 = keep_rows(0), keep_rows(0)
    k_s1 = keep_rows(1)
    same = all(np.array_equal(k1[q], k2[q]) for q in train)
    ov = np.mean([len(np.intersect1d(k1[q], k_s1[q])) / len(k1[q]) for q in train])
    print("same windows for Tamp_s0/Tfree_s0 construction: %s; seed0 vs seed1 overlap %.3f"
          % (same, ov))
    Xtr = np.concatenate([rows_of(q, k1[q]) for q in train])
    ytr = np.concatenate([np.concatenate(Y[q])[k1[q]] for q in train])
    print("train rows %d, seizure %d, NaN share %.4f [%.0f s]"
          % (len(ytr), ytr.sum(), np.isnan(Xtr).mean(), time.time() - t0), flush=True)
    log = json.loads((TR / "out" / "train_log_a.json").read_text())
    lg = {(e["fold"], e["arm"]): e for e in log}
    print("train_log_a n_train/n_pos: Tamp %s/%s Tfree %s/%s"
          % (lg[(NAME, "Tamp_s0")]["n_train"], lg[(NAME, "Tamp_s0")]["n_pos"],
             lg[(NAME, "Tfree_s0")]["n_train"], lg[(NAME, "Tfree_s0")]["n_pos"]))

    # calibration rows (pre-split) of training patients, test rows of held-out
    pre = {q: np.arange(off[q][splits[q][0]] + splits[q][1]) for q in train}
    post = {c: np.arange(off[c][splits[c][0]] + splits[c][1], off[c][-1]) for c in HELD}
    Xpre = {q: rows_of(q, pre[q]) for q in train}
    Xpost = {c: rows_of(c, post[c]) for c in HELD}
    print("loaded calibration and test rows [%.0f s]" % (time.time() - t0), flush=True)

    rng = np.random.default_rng(42)
    masks = {q: adapt_background_mask(files[q], splits[q], 0.3, rng) for q in train}
    preds = np.load(TR / "out" / "preds.npy", allow_pickle=True).item()
    thr = json.loads((TR / "out" / "thresholds_a.json").read_text())

    for arm, cols in (("Tamp_s0", None), ("Tfree_s0", tt.SHAPE)):
        def sel(x):
            return x if cols is None else x.reshape(len(x), 23, 19)[:, :, cols].reshape(len(x), -1)
        tf = time.time()
        clf = HistGradientBoostingClassifier(**tt.PARAMS).fit(sel(Xtr), ytr)
        print("\n%s fit %d features in %.0f s" % (arm, sel(Xtr[:1]).shape[1], time.time() - tf))
        # test preds
        worst = 0.0
        for c in HELD:
            p = clf.predict_proba(sel(Xpost[c]))[:, 1]
            fi, wi = splits[c]
            per = []
            for i in range(len(files[c])):
                lo, hi = off[c][i], off[c][i + 1]
                start = off[c][fi] + wi
                if hi <= start:
                    continue
                seg = p[max(lo, start) - start: hi - start]
                if len(seg) >= 10:
                    per.append(seg)
            sv = preds[arm][c]
            assert len(per) == len(sv), (c, len(per), len(sv))
            for a, (b, _) in zip(per, sv):
                worst = max(worst, float(np.max(np.abs(a - np.asarray(b)))))
            print("  %s test p max %.6f, n exactly 1.0: %d" % (c, p.max(), int((p == 1.0).sum())))
        print("  max |refit - saved test pred| = %.2e" % worst)
        # calibration background, plus in-sample flag
        pooled, insmp = [], []
        for q in train:
            pp = clf.predict_proba(sel(Xpre[q]))[:, 1]
            fi, wi = splits[q]
            res, allb, ins_r, ins_a = [], [], [], []
            for i in range(fi + 1):
                lo, hi = off[q][i], off[q][i + 1]
                if i == fi:
                    hi = lo + wi
                p_i = pp[lo:hi]
                y_i = Y[q][i][:hi - lo]
                bg = y_i == 0
                ins = np.isin(np.arange(lo, hi), k1[q])
                res.append(p_i[masks[q][i] & bg]); ins_r.append(ins[masks[q][i] & bg])
                allb.append(p_i[bg]); ins_a.append(ins[bg])
            h = np.concatenate(res)
            use_res = len(h) >= 200
            pooled.append(np.asarray(h if use_res else np.concatenate(allb), np.float32))
            insmp.append(np.concatenate(ins_r if use_res else ins_a))
        bg = np.concatenate(pooled)
        ins = np.concatenate(insmp)
        T = tt.thresholds_for(bg)
        pt = {tg: pick_threshold(bg, tg, 5, 3) for tg in tt.TARGETS}
        d_eq = max(abs(pt[tg] - T[tg]) for tg in tt.TARGETS)
        d_sv = max(abs(T[tg] - thr[arm][c][str(tg)]) for tg in tt.TARGETS for c in HELD)
        print("  calib bg %d windows (%.1f h); share also in tree training sample %.3f"
              % (len(bg), len(bg) * 2 / 3600, ins.mean()))
        print("  max |pick_threshold - thresholds_for| %.2e ; max |refit - saved thr| %.2e"
              % (d_eq, d_sv))
        print("  thresholds: " + " ".join("%s:%.4f" % (tg, T[tg]) for tg in tt.TARGETS))
        q_in = np.quantile(bg[ins], [0.5, 0.99, 0.999]) if ins.any() else [np.nan] * 3
        q_out = np.quantile(bg[~ins], [0.5, 0.99, 0.999])
        print("  bg prob quantiles 50/99/99.9%%: in-sample %s | out-of-sample %s"
              % (np.round(q_in, 4), np.round(q_out, 4)))
        # NB: dropping in-sample windows breaks time contiguity of the bg vector just as
        # the masks already do; it is only a sensitivity check of the threshold.
        T2 = tt.thresholds_for(bg[~ins])
        print("  thresholds w/o in-sample bg: " + " ".join("%s:%.4f" % (tg, T2[tg])
                                                          for tg in tt.TARGETS))
        blocks = [(np.asarray(p, np.float64), y) for c in HELD for p, y in preds[arm][c]]
        for tg in (0.05, 0.5, 2.0):
            a1, b1, r1, s1 = count(blocks, T[tg], 5, 3)
            a2, b2, _, _ = count(blocks, T2[tg], 5, 3)
            print("  held-out CHB01+CHB21 at target %s: %d/%d, %.1f FA/day  vs  w/o in-sample "
                  "%d/%d, %.1f FA/day" % (tg, a1, r1, b1 / (s1 / 86400), a2, r1,
                                         b2 / (s1 / 86400)))
        del clf
    print("\ndone [%.0f s]" % (time.time() - t0))


if __name__ == "__main__":
    with threadpool_limits(3):
        main()
