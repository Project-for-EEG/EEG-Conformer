"""Independent levels / statistic / bootstrap / decisions for clean_rule.txt v2 (verify2/rule).
Reads verify2/rule/tables.npz (per-block counts from rule_rescore.py). Does not import
clean_score.py. Writes verify2/rule/rule_stats.json and prints a report.

Variants (all reported; 'main' is the literal reading that should reproduce the log):
  main       rng 7: idx = integers(0, n_subj, (2000, n_subj)), then seeds integers(0,3,(2000,3)),
             the same seed indices applied to X and Y when both are 3-seed arms (paired);
             subsets get their own rng-7 matrix of size n_subj (as clean_score does).
  indep      X and Y seed draws independent (Y seeds from a second rng, seed 8).
  restrict   subsets reuse the 23-subject rng-7 matrix with the other subjects' weights zeroed.
  rng8/rng9  Monte Carlo check: the whole draw with rng 8 / 9.
  tiesmean   equal-FA points averaged instead of keeping the highest sensitivity.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
NEW = Path(r"C:\Users\user\AppData\Local\Temp\claude\36c1f29744f93fe6"
           r"\c7848ca6-bfef-414b-a321-d6686c1a43f8\scratchpad\clean")
OLD = Path(r"C:\Users\user\AppData\Local\Temp\claude\36c1f29744f93fe6"
           r"\b3889703-629c-4c5a-b583-69f722743ff4\scratchpad\trees")
OUT = NEW / "verify2" / "rule"
GRID = [50, 60, 75, 90, 100, 115, 125, 150]
NB = 2000
G = {"A0": ["A0"], "N1": ["N1"], "Tamp": ["Tamp_s0", "Tamp_s1", "Tamp_s2"],
     "Tfree": ["Tfree_s0", "Tfree_s1", "Tfree_s2"], "Tall": ["Tall"],
     "Trel": ["Trel_s0", "Trel_s1", "Trel_s2"], "Trelf": ["Trelf_s0"],
     "Trel_s0": ["Trel_s0"]}
# from lopo_run.log, parsed independently in parse_logs.out
EARLY = ["CHB02", "CHB04", "CHB05", "CHB09", "CHB10", "CHB12", "CHB15", "CHB16", "CHB24"]
SAME15 = ["CHB03", "CHB06", "CHB07", "CHB08", "CHB11", "CHB13", "CHB14", "CHB17", "CHB18",
          "CHB19", "CHB20", "CHB22", "CHB23"]

z = np.load(OUT / "tables.npz")
TP, FP, REF, SEC = z["TP"], z["FP"], z["REF"], z["SEC"]
ARMS = list(z["arms"])
TG = list(z["targets"])
case_of, stem_of, nflat = list(z["case_of"]), list(z["stem_of"]), z["nflat"]
CASES = sorted(set(case_of))
AI = {a: i for i, a in enumerate(ARMS)}

LOG = []


def say(s=""):
    print(s, flush=True)
    LOG.append(s)


def case_tables(keep):
    """per arm: tp, fp (24 x 14), ref, sec (24)."""
    D = {}
    cidx = np.array([CASES.index(c) for c in case_of])
    for a in ARMS:
        tp = np.zeros((len(CASES), len(TG)))
        fp = np.zeros_like(tp)
        ref = np.zeros(len(CASES))
        sec = np.zeros(len(CASES))
        for j in range(len(case_of)):
            if keep[j]:
                tp[cidx[j]] += TP[AI[a], j]
                fp[cidx[j]] += FP[AI[a], j]
                ref[cidx[j]] += REF[j]
                sec[cidx[j]] += SEC[j]
        D[a] = (tp, fp, ref, sec)
    return D


keep168 = np.ones(len(case_of), bool)
keep155 = nflat < 8
excl = [s for s, k in zip(stem_of, keep155) if not k]
say("artifact rule (>=8 channels flat in >50%% of windows), my rebuild from OLD\\feats: "
    "excluded %s; any flat channel: %s" % (excl, ["%s(%d)" % (s, n) for s, n in zip(stem_of, nflat) if n]))
say("events: 168 set %d, 155 set %d, excluded %d; excluded-file cases: %s"
    % (REF.sum(), REF[keep155].sum(), REF[~keep155].sum(),
       sorted(set(c for c, k in zip(case_of, keep155) if not k))))
DS = {"168": case_tables(keep168), "155": case_tables(keep155)}


def cell(D, g, j):
    s = np.mean([D[a][0][:, j].sum() / D[a][2].sum() for a in G[g]])
    f = np.mean([D[a][1][:, j].sum() / (D[a][3].sum() / 86400) for a in G[g]])
    return "%.3f (%6.1f)" % (s, f)


# ---- continuity check, my own parse of OLD\out\score_results_main.txt
old = (OLD / "out" / "score_results_main.txt").read_text().splitlines()
i0 = [i for i, l in enumerate(old) if l.startswith("DEPLOYABLE CURVES")][0]
bad, ncell = [], 0
for line in old[i0 + 2:i0 + 16]:
    parts = line.split()
    tg = parts[0]
    cells = re.findall(r"(\d\.\d{3}) \(\s*([\d.]+)\)", line)
    assert len(cells) == 3, line
    for g, (s, f) in zip(["A0", "Tamp", "Tfree"], cells):
        mine = cell(DS["168"], g, TG.index(tg))
        ncell += 1
        if mine.split()[0] != s or mine.split("(")[1].strip(" )") != f:
            bad.append((g, tg, mine, s, f))
say("CONTINUITY (seed-mean curve cells A0/Tamp/Tfree, 168): %d cells, %d mismatches %s"
    % (ncell, len(bad), bad))
# per seed at 2 FA/h (README OPERATING POINT section of the old file)
j2 = TG.index("2.0")
i1 = [i for i, l in enumerate(old) if l.startswith("README OPERATING POINT")][0]
pbad = []
for line in old[i1 + 1:i1 + 8]:
    m = re.match(r"\s*(\S+)\s+(\d+)/168\s+([\d.]+) FA/day", line)
    a, k, f = m.group(1), int(m.group(2)), m.group(3)
    tp, fp, ref, sec = DS["168"][a]
    mine = (int(tp[:, j2].sum()), "%.1f" % (fp[:, j2].sum() / (sec.sum() / 86400)))
    if mine != (k, f):
        pbad.append((a, mine, k, f))
say("CONTINUITY per seed at 2 FA/h (7 arms): mismatches %s" % pbad)
# per-seed full-curve check for the seeds vs the old preds of old arms is implicit (same files)

say()
say("README point (2 FA/h) per arm, 155 | 168:")
for a in ARMS:
    cs = []
    for k in ("155", "168"):
        tp, fp, ref, sec = DS[k][a]
        cs.append("%d/%d %.1f/day" % (tp[:, j2].sum(), ref.sum(), fp[:, j2].sum() / (sec.sum() / 86400)))
    say("  %-9s %s | %s" % (a, cs[0], cs[1]))


# ---- subjects, weights
def subj_of(c):
    return "CHB01" if c == "CHB21" else c


SUBJ = []
for c in CASES:
    if subj_of(c) not in SUBJ:
        SUBJ.append(subj_of(c))
assert len(SUBJ) == 23


def S_matrix(subjects):
    return np.array([[1.0 if subj_of(c) == s else 0.0 for c in CASES] for s in subjects])


def draws(subjects, seed):
    rng = np.random.default_rng(seed)
    n = len(subjects)
    idx = rng.integers(0, n, size=(NB, n))
    cnt = np.zeros((NB, n))
    for r in range(NB):
        np.add.at(cnt[r], idx[r], 1)
    SI = rng.integers(0, 3, size=(NB, 3))
    return cnt @ S_matrix(subjects), SI


def curve(arm, D, w):
    tp, fp, ref, sec = D[arm]
    fa = (w @ fp) / ((w @ sec) / 86400.0)
    se = (w @ tp) / (w @ ref)
    return fa, se


def interp(fa, se, levels, clamp, ties="max"):
    o = np.argsort(fa, kind="stable")
    fa, se = fa[o], se[o]
    ufa = np.unique(fa)
    if len(ufa) < len(fa):
        if ties == "max":
            se = np.array([se[fa == u].max() for u in ufa])
        else:
            se = np.array([se[fa == u].mean() for u in ufa])
        fa = ufa
    L = np.asarray(levels, float)
    if clamp:
        return np.interp(L, fa, se)
    out = np.interp(L, fa, se)
    out[(L < fa[0]) | (L > fa[-1])] = np.nan
    return out


def reach(g, D, w1):
    lv = []
    for L in GRID:
        if all(curve(a, D, w1)[0].min() <= L <= curve(a, D, w1)[0].max() for a in G[g]):
            lv.append(L)
    return lv


def gval(arms, D, w, levels, clamp, ties):
    return np.mean([interp(*curve(a, D, w), levels, clamp, ties) for a in arms], 0)


def decide(pt, lo, hi):
    if pt >= 0.05 and lo > 0:
        return "BETTER"
    if pt <= -0.05 and hi < 0:
        return "WORSE"
    if lo >= -0.05 and hi <= 0.05:
        return "SAME"
    return "UNCLEAR"


def compare(gx, gy, k, subset=None, variant="main", ties="max", levels=None):
    D = DS[k]
    sub = CASES if subset is None else [c for c in CASES if c in subset or subj_of(c) in subset]
    subjects = [s for s in SUBJ if any(subj_of(c) == s for c in sub)]
    w1 = S_matrix(subjects).sum(0)
    if levels is None:
        levels = [L for L in reach(gx, D, w1) if L in reach(gy, D, w1)]
    if not levels:
        return {"decision": "UNDETERMINED", "levels": []}
    ax, ay = G[gx], G[gy]
    vx = gval(ax, D, w1, levels, False, ties)
    vy = gval(ay, D, w1, levels, False, ties)
    point = float(np.mean(vx - vy))
    seed = {"rng8": 8, "rng9": 9}.get(variant, 7)
    if variant == "restrict" and subset is not None:
        W, SI = draws(SUBJ, 7)
        W = W * w1[None, :]
    else:
        W, SI = draws(subjects, seed)
    SIy = SI
    if variant == "indep":
        SIy = np.random.default_rng(8).integers(0, 3, size=(NB, 3))
    bn = np.full(NB, np.nan)
    bc = np.full(NB, np.nan)
    partial = 0
    for r in range(NB):
        w = W[r]
        if (w @ np.array([DS[k]["A0"][2]]).T).sum() == 0 or w.sum() == 0:
            continue
        sx = [ax[i] for i in SI[r]] if len(ax) == 3 else ax
        sy = [ay[i] for i in SIy[r]] if len(ay) == 3 else ay
        d = gval(sx, D, w, levels, False, ties) - gval(sy, D, w, levels, False, ties)
        ok = ~np.isnan(d)
        if ok.any():
            bn[r] = d[ok].mean()
            partial += int(ok.sum() < len(levels))
        bc[r] = np.mean(gval(sx, D, w, levels, True, ties) - gval(sy, D, w, levels, True, ties))
    kept = ~np.isnan(bn)
    lo, hi = np.percentile(bn[kept], [2.5, 97.5])
    ck = ~np.isnan(bc)
    clo, chi = np.percentile(bc[ck], [2.5, 97.5])
    d1, d2 = decide(point, lo, hi), decide(point, clo, chi)
    per_seed = {a: float(np.mean(gval([a], D, w1, levels, False, ties) - vy)) for a in ax} \
        if len(ax) > 1 else {}
    return {"levels": levels, "point": point, "lo": float(lo), "hi": float(hi),
            "clo": float(clo), "chi": float(chi), "partial": partial,
            "dropped": int((~kept).sum()), "d_nan": d1, "d_clamp": d2,
            "decision": d1 if d1 == d2 else "UNCLEAR",
            "per_level": [float(v) for v in (vx - vy)], "per_seed": per_seed,
            "bn": bn, "w_has_12": W[:, CASES.index("CHB12")] > 0}


def fmt(r):
    if r["decision"] == "UNDETERMINED":
        return "UNDETERMINED"
    return ("%+.1f over %s | 95%% [%+.1f, %+.1f] | clamped [%+.1f, %+.1f] | partial %d dropped %d"
            " -> %s (nan %s, clamp %s)" % (100 * r["point"], r["levels"], 100 * r["lo"], 100 * r["hi"],
                                          100 * r["clo"], 100 * r["chi"], r["partial"], r["dropped"],
                                          r["decision"], r["d_nan"], r["d_clamp"]))


def combine(a, b):
    if a == b:
        return a
    return "UNCLEAR"


say()
for k in ("155", "168"):
    w1 = np.ones(len(CASES))
    say("reach %s: " % k + "; ".join("%s %s" % (g, reach(g, DS[k], w1)) for g in G))
    for a in ARMS:
        fa, se = curve(a, DS[k], w1)
        say("   %-9s FA/day range [%.2f, %.2f]" % (a, fa.min(), fa.max()))

results = {}
QS = [("Q1", "Tamp", "N1", None), ("Q2", "Trel", "Tamp", None), ("Q3all", "N1", "A0", None),
      ("Q4", "Trel", "N1", None),
      ("Q2noMOT", "Trel", "Tamp", [s for s in SUBJ if s not in ("CHB06", "CHB15")]),
      ("Q4noMOT", "Trel", "N1", [s for s in SUBJ if s not in ("CHB06", "CHB15")]),
      ("Q3a", "N1", "A0", EARLY), ("Q3b", "N1", "A0", SAME15),
      ("SEC_Trelf_vs_Trel_s0 (rule)", "Trelf", "Trel_s0", None),
      ("SEC_Trelf_vs_Trel (code)", "Trelf", "Trel", None)]
for name, gx, gy, sub in QS:
    say()
    say("%s %s vs %s%s" % (name, gx, gy, "" if sub is None else " subset %d subjects" % len(sub)))
    res = {}
    for k in ("155", "168"):
        r = compare(gx, gy, k, sub)
        res[k] = r
        say("  [%s] %s" % (k, fmt(r)))
        say("        per level: " + ", ".join("%d: %+.1f" % (L, 100 * v)
                                             for L, v in zip(r["levels"], r["per_level"])))
        if r["per_seed"]:
            say("        per seed: " + ", ".join("%s %+.1f" % (a, 100 * v) for a, v in r["per_seed"].items()))
    say("  COMBINED (155 & 168): %s" % combine(res["155"]["decision"], res["168"]["decision"]))
    results[name] = res

# Q4: why 155 and 168 share the same lower bound
for name in ("Q4", "Q4noMOT"):
    a, b = results[name]["155"], results[name]["168"]
    same = np.isclose(a["bn"], b["bn"], atol=0, rtol=0)
    say("%s: draws with identical statistic on 155 and 168: %d of 2000; draws without CHB12: %d; "
        "identical & without CHB12: %d" % (name, int(np.nansum(same)), int((~a["w_has_12"]).sum()),
                                           int(np.sum(same & ~a["w_has_12"]))))

# ---- variants for robustness
say()
say("ROBUSTNESS VARIANTS (decision only; see variant list in the docstring)")
for name, gx, gy, sub in QS[:8]:
    for var in ("indep", "restrict", "rng8", "rng9", "tiesmean"):
        if var == "restrict" and sub is None:
            continue
        if var == "indep" and not (len(G[gx]) == 3 and len(G[gy]) == 3):
            continue
        rr = []
        for k in ("155", "168"):
            r = compare(gx, gy, k, sub, variant="main" if var == "tiesmean" else var,
                        ties="mean" if var == "tiesmean" else "max")
            rr.append(r)
        say("  %-8s %-9s [155] %s" % (name, var, fmt(rr[0])))
        say("  %-8s %-9s [168] %s  => %s" % ("", "", fmt(rr[1]), combine(rr[0]["decision"], rr[1]["decision"])))

# Q3 on common levels of (a) and (b)
say()
for k in ("155", "168"):
    la, lb = results["Q3a"][k]["levels"], results["Q3b"][k]["levels"]
    com = [L for L in la if L in lb]
    ra = compare("N1", "A0", k, EARLY, levels=com)
    rb = compare("N1", "A0", k, SAME15, levels=com)
    say("Q3 common levels %s [%s]: (a) %s" % (com, k, fmt(ra)))
    say("Q3 common levels %s [%s]: (b) %s" % (com, k, fmt(rb)))

# tie count in full data and in draws (main matrix)
nt = 0
W, _ = draws(SUBJ, 7)
for a in ARMS:
    for k in ("155", "168"):
        for r in range(NB):
            fa, _ = curve(a, DS[k], W[r])
            nt += int(len(np.unique(fa)) < len(fa))
say()
say("draws (arm x set x draw) with a tie in FA/day among the 14 points: %d of %d" % (nt, len(ARMS) * 2 * NB))

json.dump({k: {s: {kk: vv for kk, vv in v.items() if kk not in ("bn", "w_has_12")}
               for s, v in r.items()} for k, r in results.items()},
          open(OUT / "rule_stats.json", "w"), indent=1)
(OUT / "rule_stats.txt").write_text("\n".join(LOG) + "\n")
