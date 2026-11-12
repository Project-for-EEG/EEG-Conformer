"""critic: independent statistics on c1_counts.npy (own code, rule v2 section 4).
Reproduces Q1-Q4, then fills gaps: leave-one-subject-out, Tfree/Tall vs N1 (exploratory),
Trelf_s0 vs Trel_s0 as the rule words it, Q2-without on the main levels, Trel vs A0."""
import json
import sys
from pathlib import Path

import numpy as np

NEW = Path(r"C:\Users\user\AppData\Local\Temp\claude\36c1f29744f93fe6"
           r"\c7848ca6-bfef-414b-a321-d6686c1a43f8\scratchpad\clean")
OUT = NEW / "verify2" / "critic"
d = np.load(OUT / "c1_counts.npy", allow_pickle=True).item()
C, META, CASES = d["counts"], d["meta"], d["cases"]
GRID = (50, 60, 75, 90, 100, 115, 125, 150)
NB = 2000
GROUPS = {"A0": ["A0"], "N1": ["N1"], "Tamp": ["Tamp_s0", "Tamp_s1", "Tamp_s2"],
          "Tfree": ["Tfree_s0", "Tfree_s1", "Tfree_s2"], "Tall": ["Tall"],
          "Trel": ["Trel_s0", "Trel_s1", "Trel_s2"], "Trelf": ["Trelf_s0"],
          "Trel_s0": ["Trel_s0"]}
EARLY = ["CHB02", "CHB04", "CHB05", "CHB09", "CHB10", "CHB12", "CHB15", "CHB16", "CHB24"]
SAME15 = ["CHB03", "CHB06", "CHB07", "CHB08", "CHB11", "CHB13", "CHB14", "CHB17", "CHB18",
          "CHB19", "CHB20", "CHB22", "CHB23"]
EXCL = {"chb12_27_labeled", "chb12_28_labeled", "chb12_29_labeled"}

# per case x target tables, 168 and 155
TAB = {}
for k in ("168", "155"):
    TAB[k] = {}
    for a, per in C.items():
        arr = np.zeros((len(CASES), 14, 4))
        for i, c in enumerate(CASES):
            keep = np.array([not (k == "155" and s in EXCL) for s, _, _ in META[c]])
            arr[i] = per[c][keep].sum(0)
        TAB[k][a] = arr
for k in TAB:
    a0 = TAB[k]["A0"]
    print("set", k, "events", a0[:, 0, 2].sum(), "hours %.3f" % (a0[:, 0, 3].sum() / 3600))


def subj_of(c):
    return "CHB01" if c == "CHB21" else c


def curve(tab, w):
    tp, fp, ref, sec = (w @ tab[:, :, j] for j in range(4))
    return fp / (sec / 86400.0), tp / ref


def interp(fa, s, L, clamp):
    o = np.lexsort((-s, fa))
    fa, s = fa[o], s[o]
    first = np.concatenate([[True], np.diff(fa) > 0])
    fa, s = fa[first], s[first]
    L = np.asarray(L, float)
    return np.interp(L, fa, s) if clamp else np.interp(L, fa, s, left=np.nan, right=np.nan)


def gat(k, arms, w, L, clamp=False):
    return np.mean([interp(*curve(TAB[k][a], w), L, clamp) for a in arms], 0)


def weights(use):
    subj = []
    for c in use:
        s = subj_of(c)
        if s not in subj:
            subj.append(s)
    S = np.array([[1.0 if (c in use and subj_of(c) == s) else 0.0 for c in CASES] for s in subj])
    return subj, S


def reach(k, g, w):
    return [L for L in GRID if not np.isnan(gat(k, GROUPS[g], w, [L])[0])]


def compare(k, gx, gy, use=CASES, levels=None, rng_seed=7, nb=NB, indep_seeds=False):
    subj, S = weights(use)
    w1 = S.sum(0)
    if levels is None:
        levels = [L for L in reach(k, gx, w1) if L in reach(k, gy, w1)]
    ax, ay = GROUPS[gx], GROUPS[gy]
    vx, vy = gat(k, ax, w1, levels), gat(k, ay, w1, levels)
    point = float(np.nanmean(vx - vy))
    rng = np.random.default_rng(rng_seed)
    idx = rng.integers(0, len(subj), size=(nb, len(subj)))
    SI = rng.integers(0, 3, size=(nb, 3))
    SI2 = rng.integers(0, 3, size=(nb, 3)) if indep_seeds else SI
    bn, bc = np.full(nb, np.nan), np.full(nb, np.nan)
    partial = 0
    for r in range(nb):
        cnt = np.bincount(idx[r], minlength=len(subj)).astype(float)
        w = cnt @ S
        sx = [ax[i] for i in SI[r]] if len(ax) == 3 else ax
        sy = [ay[i] for i in SI2[r]] if len(ay) == 3 else ay
        dd = gat(k, sx, w, levels) - gat(k, sy, w, levels)
        ok = ~np.isnan(dd)
        if ok.any():
            bn[r] = dd[ok].mean()
            partial += int(ok.sum() < len(levels))
        bc[r] = np.mean(gat(k, sx, w, levels, True) - gat(k, sy, w, levels, True))
    kept = ~np.isnan(bn)
    lo, hi = np.percentile(bn[kept], [2.5, 97.5])
    clo, chi = np.percentile(bc, [2.5, 97.5])
    return {"levels": levels, "point": point, "per_level": (vx - vy).tolist(), "lo": lo, "hi": hi,
            "clo": clo, "chi": chi, "partial": partial, "dropped": int((~kept).sum()),
            "bn": bn, "idx": idx, "subj": subj}


def decide(p, lo, hi):
    if p >= 0.05 and lo > 0:
        return "BETTER"
    if p <= -0.05 and hi < 0:
        return "WORSE"
    if lo >= -0.05 and hi <= 0.05:
        return "SAME"
    return "UNCLEAR"


def fmt(r):
    d1, d2 = decide(r["point"], r["lo"], r["hi"]), decide(r["point"], r["clo"], r["chi"])
    return ("%+.4f over %s | 95%% [%+.4f, %+.4f] | clamped [%+.4f, %+.4f] | partial %d dropped %d "
            "-> %s" % (100 * r["point"], r["levels"], 100 * r["lo"], 100 * r["hi"], 100 * r["clo"],
                       100 * r["chi"], r["partial"], r["dropped"], d1 if d1 == d2 else "UNCLEAR"))


