"""rescore / steps 1-5: tables, 155/168 sets, matched-FA statistics, bootstrap.

Written without reading clean_score.py. Inputs (all produced by rescore):
  counts.npz      per-block official counts (score_blocks.py)
  block_map.json  block -> file mapping and flat-channel stats (map_blocks.py)
Output: analyze.txt / analyze.json next to them.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
GRID = np.array([50, 60, 75, 90, 100, 115, 125, 150], dtype=float)
EXCLUDE = ["chb12_27_labeled", "chb12_28_labeled", "chb12_29_labeled"]
B = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
SEED = 20261109
OUT = []


def say(s=""):
    print(s)
    OUT.append(s)


c = np.load(HERE / "counts.npz")
arms = list(c["arms"])
targets = list(c["targets"])
case = c["case"]
block = c["block"]
TP, FP, REF, SEC = c["tp"], c["fp"], c["ref"], c["sec"]
A = {a: i for i, a in enumerate(arms)}
bmap = json.loads((HERE / "block_map.json").read_text())

# block -> file stem
stems = np.array([bmap[cs][int(b)]["file"] for cs, b in zip(case, block)])
for cs, b in zip(case, block):
    assert len(bmap[cs]) == int((case == cs).sum())
keep155 = ~np.isin(stems, EXCLUDE)
assert (~keep155).sum() == 3, stems[~keep155]

# sec / ref must not depend on arm or target
for f in (SEC, REF):
    assert np.all(f == f[:1, :1, :]), "sec/ref vary with arm/target"
ref_blk = REF[0, 0]
say("events: all %d, dropped blocks %s hold %d, left %d"
    % (ref_blk.sum(), list(stems[~keep155]), ref_blk[~keep155].sum(), ref_blk[keep155].sum()))
say("hours: all %.3f, 155-set %.3f" % (SEC[0, 0].sum() / 3600, SEC[0, 0][keep155].sum() / 3600))

subj_of_case = {cs: ("CHB01" if cs == "CHB21" else cs) for cs in sorted(set(case))}
subjects = sorted(set(subj_of_case.values()))
assert len(subjects) == 23
S_idx = np.array([subjects.index(subj_of_case[cs]) for cs in case])


def per_subject(mask):
    """[arm, target, subject] sums of tp, fp, ref, sec over the blocks in mask."""
    out = []
    for f in (TP, FP, REF, SEC):
        m = np.zeros(f.shape[:2] + (23,))
        for s in range(23):
            sel = mask & (S_idx == s)
            m[:, :, s] = f[:, :, sel].sum(-1)
        out.append(m)
    return out


def curve(tp, fp, ref, sec):
    """sens, fa/day per target (last axis = targets)."""
    return tp / ref, fp / (sec / 86400.0)


def interp(fa, sens, levels, clamp=False):
    """Linear interpolation of sens vs fa, points sorted by fa (stable)."""
    o = np.argsort(fa, kind="stable")
    x, y = fa[o], sens[o]
    v = np.interp(levels, x, y)
    if not clamp:
        v = np.where((levels >= x[0]) & (levels <= x[-1]), v, np.nan)
    return v


SETS = {"155": keep155, "168": np.ones_like(keep155)}
FAM = {"Tamp": ["Tamp_s0", "Tamp_s1", "Tamp_s2"], "Trel": ["Trel_s0", "Trel_s1", "Trel_s2"],
       "Tfree": ["Tfree_s0", "Tfree_s1", "Tfree_s2"], "N1": ["N1"], "A0": ["A0"]}
QS = [("Q1", "Tamp", "N1"), ("Q2", "Trel", "Tamp"), ("Q3", "N1", "A0"), ("Q4", "Trel", "N1")]

res = {}
for sname, mask in SETS.items():
    tp, fp, ref, sec = per_subject(mask)
    pooled = {a: (tp[A[a]].sum(-1), fp[A[a]].sum(-1), ref[A[a]].sum(-1), sec[A[a]].sum(-1)) for a in arms}
    say()
    say("=" * 78)
    say("SET %s events" % sname)
    say("per arm, 14 targets: caught/ref  FA/day   (FA count)")
    for a in arms:
        t_, f_, r_, s_ = pooled[a]
        say("%-9s " % a + "  ".join("%g:%d/%d@%.1f" % (tg, t_[i], r_[i], f_[i] / (s_[i] / 86400))
                                     for i, tg in enumerate(targets)))
    say()
    ti2 = targets.index(2.0)
    say("README operating point (target 2.0):")
    for a in arms:
        t_, f_, r_, s_ = pooled[a]
        say("  %-9s %d/%d = %.4f  FA %d  %.4f/day" % (a, t_[ti2], r_[ti2], t_[ti2] / r_[ti2], f_[ti2],
                                                  f_[ti2] / (s_[ti2] / 86400)))
    # ties check
    for a in arms:
        sens, fa = curve(*pooled[a])
        o = np.argsort(fa, kind="stable")
        x, y = fa[o], sens[o]
        tie = [(x[i], y[i], y[i + 1]) for i in range(13) if x[i] == x[i + 1] and y[i] != y[i + 1]]
        nonmono = sum(1 for i in range(13) if y[i + 1] < y[i])
        if tie:
            say("  TIE with different sens in %s: %s" % (a, tie))
        if nonmono:
            say("  note: %s sens decreases with FA at %d step(s) after sorting" % (a, nonmono))
    # level sets
    say()
    rng_of = {}
    for a in arms:
        sens, fa = curve(*pooled[a])
        rng_of[a] = (fa.min(), fa.max())
        say("  range %-9s %.2f .. %.2f /day" % (a, fa.min(), fa.max()))
    res[sname] = {}
    for q, X, Y in QS:
        members = FAM[X] + FAM[Y]
        lv = np.array([L for L in GRID if all(rng_of[m][0] <= L <= rng_of[m][1] for m in members)])
        # full-data stat
        def famcurve(F, clamp=False, P=pooled):
            return np.mean([interp(curve(*P[m])[1], curve(*P[m])[0], lv, clamp) for m in FAM[F]], axis=0)
        sx, sy = famcurve(X), famcurve(Y)
        d_lv = 100 * (sx - sy)
        stat = float(np.mean(d_lv)) if len(lv) else float("nan")
        per_seed = {m: float(np.mean(100 * (interp(curve(*pooled[m])[1], curve(*pooled[m])[0], lv) - sy)))
                    for m in FAM[X]} if len(FAM[X]) > 1 else {}
        res[sname][q] = {"levels": lv.tolist(), "stat": stat, "per_level": d_lv.tolist(),
                         "X_per_level": (100 * sx).tolist(), "Y_per_level": (100 * sy).tolist(),
                         "per_seed": per_seed}
        say("  %s %s-%s levels %s  stat %+.4f pts" % (q, X, Y, lv.tolist(), stat))
        say("      per level: " + " ".join("%+.2f" % v for v in d_lv))
        say("      X sens: " + " ".join("%.2f" % v for v in 100 * sx) + " | Y sens: " +
            " ".join("%.2f" % v for v in 100 * sy))
        if per_seed:
            say("      per seed minus %s: %s" % (Y, {k: round(v, 4) for k, v in per_seed.items()}))

# ---------------------------------------------------------------- bootstrap
rng = np.random.default_rng(SEED)
W = np.stack([np.bincount(rng.integers(0, 23, 23), minlength=23) for _ in range(B)]).astype(float)
SEEDS = rng.integers(0, 3, size=(B, 3))          # shared by Tamp and Trel (paired seeds)


def decide(stat, lo, hi):
    if stat >= 5 and lo > 0:
        return "BETTER"
    if stat <= -5 and hi < 0:
        return "WORSE"
    if lo > -5 and hi < 5:
        return "SAME"
    return "UNCLEAR"


say()
say("=" * 78)
say("BOOTSTRAP  %d draws, rng default_rng(%d), subjects resampled with replacement" % (B, SEED))
say("(CHB01+CHB21 one subject), tree seeds resampled with replacement each draw.")
boot = {}
for sname, mask in SETS.items():
    tp, fp, ref, sec = per_subject(mask)
    boot[sname] = {}
    for q, X, Y in QS:
        lv = np.array(res[sname][q]["levels"])
        vals = {"nan": [], "clamp": []}
        partial = dropped = 0
        for b in range(B):
            w = W[b]
            P = {m: (tp[A[m]] @ w, fp[A[m]] @ w, ref[A[m]] @ w, sec[A[m]] @ w) for m in FAM[X] + FAM[Y]}

            def fam(F, clamp):
                ms = FAM[F]
                if len(ms) > 1:                       # tree family: resample its 3 seeds
                    ms = [ms[k] for k in SEEDS[b]]    # same picks for Tamp and Trel (paired seeds)
                return np.mean([interp(curve(*P[m])[1], curve(*P[m])[0], lv, clamp)
                                for m in ms], axis=0)  # NaN in any picked seed -> NaN at that level
            for mode in ("nan", "clamp"):
                cl = mode == "clamp"
                sx = fam(X, cl)
                sy = fam(Y, cl)
                d = 100 * (sx - sy)
                ok = ~np.isnan(d)
                if mode == "nan":
                    if not ok.any():
                        dropped += 1
                        continue
                    if not ok.all():
                        partial += 1
                    vals["nan"].append(d[ok].mean())
                else:
                    vals["clamp"].append(d.mean())
        out = {}
        for mode in ("nan", "clamp"):
            v = np.array(vals[mode])
            lo, hi = np.percentile(v, [2.5, 97.5])
            out[mode] = (float(lo), float(hi), len(v))
        st = res[sname][q]["stat"]
        dn = decide(st, out["nan"][0], out["nan"][1])
        dc = decide(st, out["clamp"][0], out["clamp"][1])
        dec = dn if dn == dc else "UNCLEAR"
        boot[sname][q] = {"stat": st, "ci_nan": out["nan"][:2], "kept": out["nan"][2], "partial": partial,
                          "dropped": dropped, "ci_clamp": out["clamp"][:2], "dec_nan": dn, "dec_clamp": dc,
                          "decision": dec}
        say("  set %s %s %s-%s: stat %+.4f  CI(NaN) [%+.4f, %+.4f] kept %d partial %d dropped %d | "
            "CI(clamp) [%+.4f, %+.4f]  -> %s / %s => %s"
            % (sname, q, X, Y, st, out["nan"][0], out["nan"][1], out["nan"][2], partial, dropped,
               out["clamp"][0], out["clamp"][1], dn, dc, dec))
say()
say("FINAL DECISIONS (BETTER/WORSE must hold on both sets):")
final = {}
for q, X, Y in QS:
    d155, d168 = boot["155"][q]["decision"], boot["168"][q]["decision"]
    f = d155 if d155 == d168 else "UNCLEAR"
    final[q] = f
    say("  %s %s vs %s: 155 %s, 168 %s => %s" % (q, X, Y, d155, d168, f))

(HERE / ("analyze_B%d.txt" % B)).write_text("\n".join(OUT) + "\n")
(HERE / ("analyze_B%d.json" % B)).write_text(json.dumps({"full": res, "boot": boot, "final": final,
                                                          "seed": SEED, "B": B}, indent=1))
