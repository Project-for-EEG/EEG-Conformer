"""rescore / step 5b: how robust are the bootstrap decisions?

Same data and statistic as analyze.py (own code, not clean_score.py).
Varies: rng seed (6 seeds), number of draws, and how resampled tree seeds are
combined when one seed's curve does not reach a level:
  prop   NaN if ANY picked seed is outside its range (analyze.py's choice)
  nanm   nanmean over the picked seeds (NaN only if all are outside)
  fixed  tree seeds not resampled (all three, NaN-propagating)
Also: why draws are dropped, and what drives Q4's lower bound.
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
GRID = np.array([50, 60, 75, 90, 100, 115, 125, 150], dtype=float)
EXCLUDE = ["chb12_27_labeled", "chb12_28_labeled", "chb12_29_labeled"]
OUT = []


def say(s=""):
    print(s)
    OUT.append(s)


c = np.load(HERE / "counts.npz")
arms = list(c["arms"])
A = {a: i for i, a in enumerate(arms)}
case, block = c["case"], c["block"]
TP, FP, REF, SEC = c["tp"], c["fp"], c["ref"], c["sec"]
bmap = json.loads((HERE / "block_map.json").read_text())
stems = np.array([bmap[cs][int(b)]["file"] for cs, b in zip(case, block)])
keep155 = ~np.isin(stems, EXCLUDE)
subjects = sorted(set("CHB01" if x == "CHB21" else x for x in case))
S_idx = np.array([subjects.index("CHB01" if x == "CHB21" else x) for x in case])
FAM = {"Tamp": ["Tamp_s0", "Tamp_s1", "Tamp_s2"], "Trel": ["Trel_s0", "Trel_s1", "Trel_s2"],
       "N1": ["N1"], "A0": ["A0"]}
QS = [("Q1", "Tamp", "N1"), ("Q2", "Trel", "Tamp"), ("Q3", "N1", "A0"), ("Q4", "Trel", "N1")]
SETS = {"155": keep155, "168": np.ones_like(keep155)}


def per_subject(mask):
    out = []
    for f in (TP, FP, REF, SEC):
        m = np.zeros(f.shape[:2] + (23,))
        for s in range(23):
            m[:, :, s] = f[:, :, mask & (S_idx == s)].sum(-1)
        out.append(m)
    return out


def interp_many(fa, sens, lv, clamp):
    """fa, sens: [draws, 14]. Returns [draws, levels]."""
    o = np.argsort(fa, axis=1, kind="stable")
    x = np.take_along_axis(fa, o, 1)
    y = np.take_along_axis(sens, o, 1)
    out = np.empty((fa.shape[0], len(lv)))
    for i in range(fa.shape[0]):
        out[i] = np.interp(lv, x[i], y[i])
        if not clamp:
            out[i][(lv < x[i, 0]) | (lv > x[i, -1])] = np.nan
    return out


def decide(stat, lo, hi):
    if stat >= 5 and lo > 0:
        return "BETTER"
    if stat <= -5 and hi < 0:
        return "WORSE"
    if lo > -5 and hi < 5:
        return "SAME"
    return "UNCLEAR"


PS = {s: per_subject(m) for s, m in SETS.items()}


def levels_and_stat(sname, X, Y):
    tp, fp, ref, sec = PS[sname]
    rng_ok = {}
    cur = {}
    for m in FAM[X] + FAM[Y]:
        t, f, r, s = (a[A[m]].sum(-1) for a in (tp, fp, ref, sec))
        sens, fa = t / r, f / (s / 86400)
        cur[m] = (fa, sens)
        rng_ok[m] = (fa.min(), fa.max())
    lv = np.array([L for L in GRID if all(rng_ok[m][0] <= L <= rng_ok[m][1] for m in FAM[X] + FAM[Y])])
    fx = np.mean([interp_many(cur[m][0][None], cur[m][1][None], lv, False)[0] for m in FAM[X]], 0)
    fy = np.mean([interp_many(cur[m][0][None], cur[m][1][None], lv, False)[0] for m in FAM[Y]], 0)
    return lv, float(100 * np.mean(fx - fy))


def run(seed, B, mode):
    rng = np.random.default_rng(seed)
    W = np.stack([np.bincount(rng.integers(0, 23, 23), minlength=23) for _ in range(B)]).astype(float)
    SEEDS = rng.integers(0, 3, size=(B, 3))
    if mode == "fixed":
        SEEDS = np.tile(np.arange(3), (B, 1))
    res = {}
    for sname in SETS:
        tp, fp, ref, sec = PS[sname]
        for q, X, Y in QS:
            lv, stat = levels_and_stat(sname, X, Y)
            curves = {}
            for m in FAM[X] + FAM[Y]:
                t, f, r, s = (a[A[m]] @ W.T for a in (tp, fp, ref, sec))   # [14, B]
                sens, fa = (t / r).T, (f / (s / 86400)).T
                curves[m] = {cl: interp_many(fa, sens, lv, cl) for cl in (False, True)}

            def fam(F, cl):
                ms = FAM[F]
                if len(ms) == 1:
                    return curves[ms[0]][cl]
                stack = np.stack([curves[m][cl] for m in ms])            # [3, B, L]
                picked = stack[SEEDS.T, np.arange(B)[None, :], :]         # [3, B, L]
                if mode == "nanm":
                    with np.errstate(all="ignore"):
                        return np.nanmean(picked, 0)
                return picked.mean(0)
            out = {}
            for cl in (False, True):
                d = 100 * (fam(X, cl) - fam(Y, cl))
                ok = ~np.isnan(d)
                keep = ok.any(1)
                with np.errstate(all="ignore"):
                    st = np.where(keep, np.nanmean(np.where(ok, d, np.nan), 1), np.nan)
                v = st[keep]
                lo, hi = np.percentile(v, [2.5, 97.5])
                out[cl] = (lo, hi, int((keep & ~ok.all(1)).sum()), int((~keep).sum()), st)
            dn, dc = decide(stat, out[False][0], out[False][1]), decide(stat, out[True][0], out[True][1])
            res[(sname, q)] = dict(stat=stat, lo=out[False][0], hi=out[False][1], partial=out[False][2],
                                   dropped=out[False][3], clo=out[True][0], chi=out[True][1],
                                   dec=dn if dn == dc else "UNCLEAR", draws=out[False][4], W=W, SEEDS=SEEDS)
    return res


say("robustness of the bootstrap decisions (own implementation)")
summary = {}
for mode in ("prop", "nanm", "fixed"):
    for seed, B in [(20261109, 2000), (1, 2000), (2, 2000), (3, 2000), (12345, 2000), (987654321, 2000),
                    (20261109, 10000)]:
        if mode != "prop" and B != 2000:
            continue
        r = run(seed, B, mode)
        for q, X, Y in QS:
            a, b = r[("155", q)], r[("168", q)]
            fin = a["dec"] if a["dec"] == b["dec"] else "UNCLEAR"
            summary.setdefault(q, []).append(fin)
            say("%-5s seed %-10d B %-5d %s %s-%s | 155: %+.1f [%+.2f,%+.2f] clamp [%+.2f,%+.2f] p%d d%d %s"
                " | 168: %+.1f [%+.2f,%+.2f] clamp [%+.2f,%+.2f] p%d d%d %s => %s"
                % (mode, seed, B, q, X, Y, a["stat"], a["lo"], a["hi"], a["clo"], a["chi"], a["partial"],
                   a["dropped"], a["dec"], b["stat"], b["lo"], b["hi"], b["clo"], b["chi"], b["partial"],
                   b["dropped"], b["dec"], fin))
        if mode == "prop" and seed == 20261109 and B == 2000:
            # why draws were dropped in Q1
            for sname in SETS:
                d = r[(sname, "Q1")]
                idx = np.where(np.isnan(d["draws"]))[0]
                for i in idx:
                    w = d["W"][i]
                    tp, fp, ref, sec = PS[sname]
                    rr = {}
                    for m in FAM["Tamp"] + FAM["N1"]:
                        f = fp[A[m]] @ w
                        s = sec[A[m]] @ w
                        fa = f / (s / 86400)
                        rr[m] = "%.1f..%.1f" % (fa.min(), fa.max())
                    say("    dropped Q1 draw %d (set %s): seeds %s, ranges %s"
                        % (i, sname, d["SEEDS"][i].tolist(), rr))
            # Q4 lower tail vs CHB12 weight
            ci12 = subjects.index("CHB12")
            for sname in SETS:
                d = r[(sname, "Q4")]
                st = d["draws"]
                lo_cut = np.nanpercentile(st, 5)
                tail = st <= lo_cut
                say("    Q4 set %s: lowest 5%% of draws (<= %.2f): %d draws, CHB12 weight 0 in %d of them;"
                    " overall CHB12 weight 0 in %d of %d draws"
                    % (sname, lo_cut, tail.sum(), int((d["W"][tail, ci12] == 0).sum()),
                       int((d["W"][:, ci12] == 0).sum()), len(st)))
            a, b = r[("155", "Q4")]["draws"], r[("168", "Q4")]["draws"]
            w12 = r[("155", "Q4")]["W"][:, ci12]
            say("    Q4: draws with CHB12 weight 0 give identical 155/168 statistics: max |diff| %.3g"
                % np.nanmax(np.abs(a[w12 == 0] - b[w12 == 0])))

say()
for q in summary:
    say("%s final decisions over all runs: %s" % (q, sorted(set(summary[q]))))
(HERE / "robust.txt").write_text("\n".join(OUT) + "\n")
