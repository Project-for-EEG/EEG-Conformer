"""Independent rebuild of A0: test blocks and all 14 thresholds per fold, from the cached
fold-model logits, with my own split / mask / background code and the repo's
calibrate_per_patient.pick_threshold. Compares with out/preds_a.npy, thresholds_a.json and
C:\\EEG\\endtoend_uncalibrated.npz. Run with cwd C:\\EEG, python -B."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits

sys.path.insert(0, r"C:\EEG")
from rs_common import load_all, runs, TARGETS, HERE
from calibrate_per_patient import pick_threshold

OLD = Path(r"C:\Users\user\AppData\Local\Temp\claude\36c1f29744f93fe6"
           r"\f241281f-add4-41ac-9210-f47dab0b83b0\scratchpad")
PREP = Path(r"C:\EEG\preprocessed_data")
lines = []


def say(s=""):
    print(s, flush=True)
    lines.append(s)


cases = sorted(d.name for d in PREP.iterdir() if d.is_dir())
folds = [p.stem[len("lopo_"):] for p in sorted((Path(r"C:\EEG\checkpoints\lopo")).glob("lopo_*.pt"))]
held_of = {f: f.split("_") for f in folds}
say("cases %d, folds %d" % (len(cases), len(folds)))

files, labs, logit = {}, {}, {}
for c in cases:
    files[c] = sorted((PREP / c).glob("*.npz"))
    labs[c], logit[c] = [], []
    for f in files[c]:
        with np.load(f) as d:
            y = d["labels"].astype(np.int64)
        labs[c].append(y)
        lg = {}
        for cache in ("pilot_cache", "pilot_cache2"):
            with np.load(OLD / cache / ("%s__%s.npz" % (c, f.stem))) as d:
                assert np.array_equal(d["labels"].astype(np.int64), y), (c, f.stem)
                for k in d.files:
                    if k.startswith("logit_"):
                        assert k[6:] not in lg, ("fold in both caches", k)
                        lg[k[6:]] = d[k]
        assert sorted(lg) == sorted(folds), (c, sorted(lg))
        logit[c].append(lg)


def my_split(c):
    seen = 0
    for fi, y in enumerate(labs[c]):
        lo, hi = runs(y.astype(bool))
        if len(lo):
            h = int(hi[0])
            rem = int(y[h:].sum()) + sum(int(z.sum()) for z in labs[c][fi + 1:])
            return (fi, h) if rem > 0 else None
    return None


split = {c: my_split(c) for c in cases}
assert all(split[c] is not None for c in cases)


def sig32(x):
    x = np.asarray(x, np.float32)
    return (1.0 / (1.0 + np.exp(-x.astype(np.float64)))).astype(np.float32)


P, T = load_all()
ref = np.load(r"C:\EEG\endtoend_uncalibrated.npz", allow_pickle=True)["data"].item()

# ---- test blocks -------------------------------------------------------------------
worst_saved = worst_ref = 0.0
nblk = nwin = 0
for f in folds:
    for c in held_of[f]:
        fi, wi = split[c]
        blocks = []
        for i in range(fi, len(files[c])):
            p = sig32(logit[c][i][f])
            y = labs[c][i]
            if i == fi:
                p, y = p[wi:], y[wi:]
            if len(y) >= 10:
                blocks.append((p, y))
        assert len(blocks) == len(P["A0"][c]) == len(ref[c]["blocks"]), c
        for (p, y), (ps, ys), (pr, yr) in zip(blocks, P["A0"][c], ref[c]["blocks"]):
            assert np.array_equal(y, ys) and np.array_equal(y, np.asarray(yr, np.int64)), c
            worst_saved = max(worst_saved, float(np.abs(p.astype(float) - ps).max()))
            worst_ref = max(worst_ref, float(np.abs(p.astype(float) - pr.astype(float)).max()))
            nblk += 1
            nwin += len(y)
say("A0 test blocks rebuilt from caches + own split: %d blocks, %d windows (%.2f h); "
    "labels identical to preds_a and npz; max|dp| vs preds_a %.3g, vs npz %.3g"
    % (nblk, nwin, nwin * 2 / 3600, worst_saved, worst_ref))
say("splits (file index, window): " + " ".join("%s:%d/%d" % (c, *split[c]) for c in cases))

# ---- thresholds ----------------------------------------------------------------------
worst_t_saved = worst_t_ref = 0.0
n_diff = 0
with threadpool_limits(3):
    for f in folds:
        train = [c for c in cases if c not in held_of[f]]
        rng = np.random.default_rng(42)
        pooled = []
        for q in train:
            fi, wi = split[q]
            res, allbg = [], []
            for i in range(fi + 1):
                y = labs[q][i][:wi] if i == fi else labs[q][i]
                held = np.zeros(len(y), bool)
                neg = np.where(y == 0)[0]
                if len(neg):
                    held[rng.choice(neg, size=int(len(neg) * 0.3), replace=False)] = True
                p = sig32(logit[q][i][f])[:len(y)]
                res.append(p[held & (y == 0)])
                allbg.append(p[y == 0])
            r = np.concatenate(res)
            pooled.append(r if len(r) >= 200 else np.concatenate(allbg))
        bg = np.concatenate(pooled).astype(np.float32)
        mine = {tg: pick_threshold(bg, float(tg), 5, 3) for tg in TARGETS}
        c0 = held_of[f][0]
        dsv = max(abs(mine[tg] - T["A0"][c0][tg]) for tg in TARGETS)
        drf = abs(mine["2.0"] - float(ref[c0]["t"]))
        worst_t_saved, worst_t_ref = max(worst_t_saved, dsv), max(worst_t_ref, drf)
        n_diff += sum(abs(mine[tg] - T["A0"][c0][tg]) > 1e-6 for tg in TARGETS)
        say("  fold %-12s bg %6d  t@2 mine %.6f saved %.6f npz %.6f | max|d| over 14 targets "
            "vs saved %.2g" % (f, len(bg), mine["2.0"], T["A0"][c0]["2.0"],
                               float(ref[c0]["t"]), dsv))
say("A0 thresholds: max|mine - saved| over 23 folds x 14 targets %.3g (%d cells > 1e-6); "
    "max|mine(2.0) - npz t| %.3g" % (worst_t_saved, n_diff, worst_t_ref))
(HERE / "rs_a0_out.txt").write_text("\n".join(lines) + "\n", encoding="ascii")
