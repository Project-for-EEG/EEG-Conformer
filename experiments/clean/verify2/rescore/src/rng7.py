"""rescore / mismatch explanation (written AFTER reading clean_score.py).

My own bootstrap code, but fed the same random draws clean_score.py uses
(default_rng(7): integers(0,23,(2000,23)) for subjects, then integers(0,3,(2000,3))
for tree seeds). If the intervals then equal the reported ones, every interval
difference between my run and the log is Monte Carlo noise from the rng.
Tie rule variants: 'stable' (mine: stable argsort by FA) and 'maxsens' (clean_score:
equal FA keeps the highest sensitivity).
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
GRID = np.array([50, 60, 75, 90, 100, 115, 125, 150], dtype=float)
c = np.load(HERE / "counts.npz")
arms = list(c["arms"])
A = {a: i for i, a in enumerate(arms)}
case, block = c["case"], c["block"]
TP, FP, REF, SEC = c["tp"], c["fp"], c["ref"], c["sec"]
bmap = json.loads((HERE / "block_map.json").read_text())
stems = np.array([bmap[cs][int(b)]["file"] for cs, b in zip(case, block)])
keep155 = ~np.isin(stems, ["chb12_27_labeled", "chb12_28_labeled", "chb12_29_labeled"])
subjects = sorted(set("CHB01" if x == "CHB21" else x for x in case))
S_idx = np.array([subjects.index("CHB01" if x == "CHB21" else x) for x in case])
FAM = {"Tamp": ["Tamp_s0", "Tamp_s1", "Tamp_s2"], "Trel": ["Trel_s0", "Trel_s1", "Trel_s2"],
       "N1": ["N1"], "A0": ["A0"]}
QS = [("Q1", "Tamp", "N1"), ("Q2", "Trel", "Tamp"), ("Q3", "N1", "A0"), ("Q4", "Trel", "N1")]
LV = {"Q1": [60, 75, 90, 100, 115, 125], "Q2": [60, 75, 90, 100], "Q3": [50, 60, 75, 90, 100, 115, 125],
      "Q4": [50, 60, 75, 90, 100]}
B = 2000
rng = np.random.default_rng(7)
idx = rng.integers(0, 23, size=(B, 23))
W = np.zeros((B, 23))
for r in range(B):
    np.add.at(W[r], idx[r], 1)
SI = rng.integers(0, 3, size=(B, 3))


def per_subject(mask):
    out = []
    for f in (TP, FP, REF, SEC):
        m = np.zeros(f.shape[:2] + (23,))
        for s in range(23):
            m[:, :, s] = f[:, :, mask & (S_idx == s)].sum(-1)
        out.append(m)
    return out


def interp(fa, sens, lv, clamp, tie):
    if tie == "maxsens":
        o = np.lexsort((-sens, fa))
        fa, sens = fa[o], sens[o]
        first = np.concatenate([[True], np.diff(fa) > 0])
        x, y = fa[first], sens[first]
    else:
        o = np.argsort(fa, kind="stable")
        x, y = fa[o], sens[o]
    v = np.interp(lv, x, y)
    if not clamp:
        v = np.where((lv >= x[0]) & (lv <= x[-1]), v, np.nan)
    return v


lines = []
for tie in ("stable", "maxsens"):
    for sname, mask in (("155", keep155), ("168", np.ones_like(keep155))):
        tp, fp, ref, sec = per_subject(mask)
        for q, X, Y in QS:
            lv = np.array(LV[q], float)
            bn, bc, partial = np.full(B, np.nan), np.full(B, np.nan), 0
            for r in range(B):
                w = W[r]

                def fam(F, cl):
                    ms = FAM[F]
                    if len(ms) == 3:
                        ms = [ms[k] for k in SI[r]]
                    v = []
                    for m in ms:
                        t, f, rr, s = tp[A[m]] @ w, fp[A[m]] @ w, ref[A[m]] @ w, sec[A[m]] @ w
                        v.append(interp(f / (s / 86400), t / rr, lv, cl, tie))
                    return np.mean(v, 0)
                d = 100 * (fam(X, False) - fam(Y, False))
                ok = ~np.isnan(d)
                if ok.any():
                    bn[r] = d[ok].mean()
                    partial += int(ok.sum() < len(lv))
                bc[r] = np.mean(100 * (fam(X, True) - fam(Y, True)))
            kept = ~np.isnan(bn)
            lo, hi = np.percentile(bn[kept], [2.5, 97.5])
            clo, chi = np.percentile(bc, [2.5, 97.5])
            s = ("tie=%-7s set %s %s %s-%s: 95%% [%+.1f, %+.1f] clamped [%+.1f, %+.1f] partial %d dropped %d"
                 % (tie, sname, q, X, Y, lo, hi, clo, chi, partial, int((~kept).sum())))
            print(s)
            lines.append(s)
(HERE / "rng7.txt").write_text("\n".join(lines) + "\n")
