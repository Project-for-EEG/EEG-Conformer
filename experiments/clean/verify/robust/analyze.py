"""Robustness critic, step 2: analyses (1)-(5) from verify/robust/tables.npz. Reads only."""
import os

import numpy as np

OUT = os.path.join(r"C:\Users\user\AppData\Local\Temp\claude\36c1f29744f93fe6",
                   r"c7848ca6-bfef-414b-a321-d6686c1a43f8\scratchpad\clean\verify\robust")
Z = np.load(os.path.join(OUT, "tables.npz"))
cases = list(Z["cases"])
TG = list(Z["targets"])
j2 = TG.index(2.0)
AMP = ["Tamp_s0", "Tamp_s1", "Tamp_s2"]
REL = ["Trel_s0", "Trel_s1", "Trel_s2"]
G = {"Tamp": AMP, "Trel": REL}
ref = Z["A0__ref"]
sec = Z["A0__sec"]
for a in AMP + REL:
    assert np.array_equal(Z[a + "__ref"], ref) and np.allclose(Z[a + "__sec"], sec)
day = sec / 86400.0
LINES = []


def say(s=""):
    print(s)
    LINES.append(s)


def subj_of(c):
    return "CHB01" if c == "CHB21" else c


subjects = []
for c in cases:
    if subj_of(c) not in subjects:
        subjects.append(subj_of(c))
S = np.array([[1.0 if subj_of(c) == s else 0.0 for c in cases] for s in subjects])


def pooled(a, w, j):
    return (w @ Z[a + "__tp"][:, j], w @ Z[a + "__fp"][:, j] / (w @ day), w @ ref)


def gmean(g, w, j):
    r = [pooled(a, w, j) for a in G[g]]
    return np.mean([x[0] for x in r]), np.mean([x[1] for x in r]), r[0][2]


ones = np.ones(len(cases))
say("CHECK README point, 155 set (target 2.0 FA/h):")
for a in ["A0"] + AMP + REL:
    tp, fa, n = pooled(a, ones, j2)
    say("  %-8s %d/%d  %.1f/day" % (a, tp, n, fa))
say()

# ---------------------------------------------------------------- (1) per case
say("(1) PER CASE at target 2.0, 155 set. tp per seed s0/s1/s2, fp per seed, hours")
say("%-6s %4s %6s | %-11s %-14s | %-11s %-14s | %7s %6s" % (
    "case", "ref", "hours", "Tamp tp", "Tamp fp", "Trel tp", "Trel fp", "dFP", "dTP"))
rows = []
for i, c in enumerate(cases):
    atp = [Z[a + "__tp"][i, j2] for a in AMP]
    afp = [Z[a + "__fp"][i, j2] for a in AMP]
    rtp = [Z[a + "__tp"][i, j2] for a in REL]
    rfp = [Z[a + "__fp"][i, j2] for a in REL]
    dfp = np.mean(afp) - np.mean(rfp)
    dtp = np.mean(rtp) - np.mean(atp)
    rows.append((c, dfp, dtp, np.mean(afp), np.mean(rfp), np.mean(atp), np.mean(rtp)))
    say("%-6s %4d %6.1f | %-11s %-14s | %-11s %-14s | %+7.1f %+6.2f" % (
        c, ref[i], sec[i] / 3600, "/".join(map(str, atp)), "/".join(map(str, afp)),
        "/".join(map(str, rtp)), "/".join(map(str, rfp)), dfp, dtp))
tot_dfp = sum(r[1] for r in rows)
tot_afp = sum(r[3] for r in rows)
tot_rfp = sum(r[4] for r in rows)
say("TOTAL fp (mean over seeds): Tamp %.1f, Trel %.1f, drop %.1f (%.1f%%) over %.1f h"
    % (tot_afp, tot_rfp, tot_dfp, 100 * tot_dfp / tot_afp, sec.sum() / 3600))
rk = sorted(rows, key=lambda r: -r[1])
say("Ranked by fp reduction (Tamp mean - Trel mean):")
cum = 0.0
for n, r in enumerate(rk):
    cum += r[1]
    say("  %2d %-6s drop %+6.1f  (Tamp %.1f -> Trel %.1f)  cum share of net drop %5.1f%%"
        % (n + 1, r[0], r[1], r[3], r[4], 100 * cum / tot_dfp))
gross = sum(r[1] for r in rows if r[1] > 0)
say("net drop %.1f; gross reductions %.1f in %d cases; increases %.1f in %d cases"
    % (tot_dfp, gross, sum(r[1] > 0 for r in rows), -sum(r[1] for r in rows if r[1] < 0),
       sum(r[1] < 0 for r in rows)))
for k in (1, 2, 3, 5):
    say("  top %d share of net drop %.1f%%, of gross reductions %.1f%%"
        % (k, 100 * sum(r[1] for r in rk[:k]) / tot_dfp, 100 * sum(r[1] for r in rk[:k]) / gross))
say("Seizure-catch changes (Trel mean - Tamp mean), cases with nonzero change:")
for r in sorted(rows, key=lambda r: -abs(r[2])):
    if abs(r[2]) > 1e-9:
        say("  %-6s %+5.2f  (Tamp %.2f -> Trel %.2f)" % (r[0], r[2], r[5], r[6]))
say("  total caught: Tamp mean %.2f, Trel mean %.2f" % (sum(r[5] for r in rows),
                                                      sum(r[6] for r in rows)))
# per seed pair consistency
for a, b in zip(AMP, REL):
    d = Z[a + "__fp"][:, j2] - Z[b + "__fp"][:, j2]
    say("  seed pair %s-%s: fp drop %d, cases with fewer FP under Trel %d, more %d, equal %d"
        % (a, b, d.sum(), (d > 0).sum(), (d < 0).sum(), (d == 0).sum()))
say()

# ---------------------------------------------------------------- (2) LOSO
say("(2) LEAVE-ONE-SUBJECT-OUT, README point, 155 set, mean over seeds")
say("%-7s %10s %10s %8s %8s %8s %8s" % ("drop", "Tamp FA/d", "Trel FA/d", "dFA", "ratio",
                                        "Tamp tp", "Trel tp"))
worst = None
dl = []
for si, s in enumerate(subjects):
    w = 1.0 - S[si]
    atp, afa, n = gmean("Tamp", w, j2)
    rtp, rfa, _ = gmean("Trel", w, j2)
    dl.append((s, afa - rfa, rfa / afa, rtp - atp, afa, rfa, atp, rtp, n))
    say("%-7s %10.1f %10.1f %+8.1f %8.3f %8.2f %8.2f  /%d" % (s, afa, rfa, rfa - afa, rfa / afa,
                                                            atp, rtp, n))
say("Trel fewer FA in %d of %d LOSO sets; smallest drop %.1f/day (without %s), largest ratio %.3f"
    % (sum(d[1] > 0 for d in dl), len(dl), min(d[1] for d in dl),
       min(dl, key=lambda d: d[1])[0], max(d[2] for d in dl)))
say("  caught difference range (Trel - Tamp): %+.2f .. %+.2f"
    % (min(d[3] for d in dl), max(d[3] for d in dl)))
# leave-two-out worst case
worst2 = (None, 1e9)
for i in range(len(subjects)):
    for k in range(i + 1, len(subjects)):
        w = 1.0 - S[i] - S[k]
        d = gmean("Tamp", w, j2)[1] - gmean("Trel", w, j2)[1]
        if d < worst2[1]:
            worst2 = ((subjects[i], subjects[k]), d, gmean("Tamp", w, j2)[1],
                      gmean("Trel", w, j2)[1])
say("  leave-TWO-subjects-out worst case: without %s, Tamp %.1f vs Trel %.1f/day (drop %.1f)"
    % (worst2[0], worst2[2], worst2[3], worst2[1]))
# per-subject FA ratio
say("  per subject at README point: FA/day Tamp-mean vs Trel-mean")
fewer = 0
for si, s in enumerate(subjects):
    w = S[si]
    a, r = gmean("Tamp", w, j2)[1], gmean("Trel", w, j2)[1]
    fewer += r < a
say("  subjects with fewer FA/day under Trel: %d of %d" % (fewer, len(subjects)))
say()

# ---------------------------------------------------------------- bootstrap at README point
rng = np.random.default_rng(11)
NB = 4000
bd_fa, bd_tp = [], []
for _ in range(NB):
    idx = rng.integers(0, len(subjects), len(subjects))
    w = np.bincount(idx, minlength=len(subjects)) @ S
    sa = [AMP[i] for i in rng.integers(0, 3, 3)]
    sr = [REL[i] for i in rng.integers(0, 3, 3)]
    fa_a = np.mean([pooled(a, w, j2)[1] for a in sa])
    fa_r = np.mean([pooled(a, w, j2)[1] for a in sr])
    tp_a = np.mean([pooled(a, w, j2)[0] / pooled(a, w, j2)[2] for a in sa])
    tp_r = np.mean([pooled(a, w, j2)[0] / pooled(a, w, j2)[2] for a in sr])
    bd_fa.append(fa_r / fa_a)
    bd_tp.append(tp_r - tp_a)
say("Subject bootstrap (%d draws, seeds resampled), README point, 155 set:" % NB)
say("  Trel/Tamp FA ratio: point %.3f, 95%% [%.3f, %.3f], P(ratio>=1) %.4f"
    % (gmean("Trel", ones, j2)[1] / gmean("Tamp", ones, j2)[1],
       *np.percentile(bd_fa, [2.5, 97.5]), np.mean(np.array(bd_fa) >= 1)))
say("  Trel-Tamp sensitivity: 95%% [%+.3f, %+.3f]" % tuple(np.percentile(bd_tp, [2.5, 97.5])))
say()

# ---------------------------------------------------------------- (3) curve below 50/day
say("(3) DEPLOYABLE CURVES, 155 set, per seed: target -> sensitivity (FA/day)")
for g in ("Tamp", "Trel"):
    for a in G[g]:
        cells = []
        for j, t in enumerate(TG):
            tp, fa, n = pooled(a, ones, j)
            cells.append("%.2f:%.3f(%.1f)" % (t, tp / n, fa))
        say("  %-8s %s" % (a, " ".join(cells[:6])))


def interp_at(fa, sens, L):
    o = np.lexsort((-sens, fa))
    fa, sens = fa[o], sens[o]
    f = np.concatenate([[True], np.diff(fa) > 0])
    return np.interp(L, fa[f], sens[f], left=np.nan, right=np.nan)


LEV = [10, 25, 50]
say("Deployable sensitivity interpolated at 10/25/50 FA/day (NaN = outside the arm's curve):")
for g in ("Tamp", "Trel"):
    for a in G[g]:
        fa = np.array([pooled(a, ones, j)[1] for j in range(len(TG))])
        se = np.array([pooled(a, ones, j)[0] / ref.sum() for j in range(len(TG))])
        say("  %-8s min FA %.1f/day   %s" % (a, fa.min(), "  ".join(
            "@%d: %.3f" % (L, v) for L, v in zip(LEV, interp_at(fa, se, LEV)))))
say("Best deployable sensitivity with pooled FA <= budget (any target), per seed:")
for g in ("Tamp", "Trel"):
    for a in G[g]:
        fa = np.array([pooled(a, ones, j)[1] for j in range(len(TG))])
        se = np.array([pooled(a, ones, j)[0] / ref.sum() for j in range(len(TG))])
        say("  %-8s " % a + "  ".join("<=%d: %s" % (L, ("%.3f" % se[fa <= L].max())
                                                     if (fa <= L).any() else "none")
                                       for L in (10, 25, 50, 60)))
say("Matched-TARGET comparison (same training-set FA target), 155 set, means over seeds:")
for j, t in enumerate(TG[:8]):
    a = gmean("Tamp", ones, j)
    r = gmean("Trel", ones, j)
    say("  target %.2f/h (%.1f/day): Tamp %.3f @ %.1f/day   Trel %.3f @ %.1f/day"
        % (t, 24 * t, a[0] / a[2], a[1], r[0] / r[2], r[1]))
# dominance: Trel at target 0.05 vs every Tamp point
j0 = 0
r = gmean("Trel", ones, j0)
best_amp_le = max(gmean("Tamp", ones, j)[0] / ref.sum() for j in range(len(TG))
                  if gmean("Tamp", ones, j)[1] <= 60)
say("  Trel @ target 0.05: %.3f at %.1f/day; best Tamp-mean sensitivity at <= 60/day: %.3f"
    % (r[0] / r[2], r[1], best_amp_le))
# bootstrap of the dominance at target 0.05
dom, dsens, rfa_l, afa_l = 0, [], [], []
for _ in range(NB):
    idx = rng.integers(0, len(subjects), len(subjects))
    w = np.bincount(idx, minlength=len(subjects)) @ S
    sa = [AMP[i] for i in rng.integers(0, 3, 3)]
    sr = [REL[i] for i in rng.integers(0, 3, 3)]
    sA = np.mean([pooled(a, w, j0)[0] / pooled(a, w, j0)[2] for a in sa])
    fA = np.mean([pooled(a, w, j0)[1] for a in sa])
    sR = np.mean([pooled(a, w, j0)[0] / pooled(a, w, j0)[2] for a in sr])
    fR = np.mean([pooled(a, w, j0)[1] for a in sr])
    dom += (sR >= sA) and (fR <= fA)
    dsens.append(sR - sA)
    rfa_l.append(fR)
    afa_l.append(fA)
say("  bootstrap at target 0.05: P(Trel >= Tamp in sensitivity AND <= in FA) %.3f; "
    "dSens 95%% [%+.3f, %+.3f]; Trel FA 95%% [%.1f, %.1f]; Tamp FA 95%% [%.1f, %.1f]"
    % (dom / NB, *np.percentile(dsens, [2.5, 97.5]), *np.percentile(rfa_l, [2.5, 97.5]),
       *np.percentile(afa_l, [2.5, 97.5])))
say("  per subject at target 0.05 (Tamp-mean vs Trel-mean): sens, FA/day")
nd = 0
for si, s in enumerate(subjects):
    w = S[si]
    a = gmean("Tamp", w, j0)
    rr = gmean("Trel", w, j0)
    nd += (rr[0] >= a[0]) and (rr[1] <= a[1])
    say("    %-6s Tamp %.2f/%d @ %6.1f   Trel %.2f/%d @ %6.1f" % (s, a[0], a[2], a[1], rr[0],
                                                                 rr[2], rr[1]))
say("  subjects where Trel is at least as good on both axes at target 0.05: %d of %d"
    % (nd, len(subjects)))

say("GLOBAL-THRESHOLD SWEEP (one threshold for all cases, chosen on TEST: oracle, symmetric)")
for g in ("Tamp", "Trel"):
    for a in G[g]:
        gfa = Z[a + "__gfp"].sum(0) / day.sum()
        gse = Z[a + "__gtp"].sum(0) / ref.sum()
        say("  %-8s " % a + "  ".join("<=%d/d: %.3f" % (L, gse[gfa <= L].max() if (gfa <= L).any()
                                                         else np.nan) for L in (5, 10, 25, 50, 100))
            + "   min FA %.1f" % gfa.min())
say()

# ---------------------------------------------------------------- (4) without CHB06, CHB15
say("(4) README point without CHB06 and CHB15, 155 set")
w = np.array([0.0 if c in ("CHB06", "CHB15") else 1.0 for c in cases])
for a in AMP + REL:
    tp, fa, n = pooled(a, w, j2)
    say("  %-8s %d/%d  %.1f/day" % (a, tp, n, fa))
for g in ("Tamp", "Trel"):
    tp, fa, n = gmean(g, w, j2)
    say("  %-5s mean %.2f/%d (%.3f)  %.1f/day" % (g, tp, n, tp / n, fa))
a6 = gmean("Tamp", w, j2)
r6 = gmean("Trel", w, j2)
aa = gmean("Tamp", ones, j2)
rr = gmean("Trel", ones, j2)
say("  FA reduction: all %.1f -> %.1f/day (%.1f%%); without CHB06+15 %.1f -> %.1f/day (%.1f%%)"
    % (aa[1], rr[1], 100 * (1 - rr[1] / aa[1]), a6[1], r6[1], 100 * (1 - r6[1] / a6[1])))
w2 = 1 - w
say("  CHB06+CHB15 alone: Tamp fp mean %.1f, Trel fp mean %.1f, share of net fp drop %.1f%%"
    % (w2 @ np.mean([Z[a + "__fp"][:, j2] for a in AMP], 0),
       w2 @ np.mean([Z[a + "__fp"][:, j2] for a in REL], 0),
       100 * (w2 @ (np.mean([Z[a + "__fp"][:, j2] for a in AMP], 0)
                    - np.mean([Z[a + "__fp"][:, j2] for a in REL], 0))) / tot_dfp))
say("  Matched-target curve without CHB06+CHB15 (means over seeds):")
for j, t in enumerate(TG):
    a = gmean("Tamp", w, j)
    r = gmean("Trel", w, j)
    say("    target %.2f: Tamp %.3f @ %6.1f/day   Trel %.3f @ %6.1f/day"
        % (t, a[0] / a[2], a[1], r[0] / r[2], r[1]))
say()

# ---------------------------------------------------------------- (5) achieved / target
say("(5) ACHIEVED TEST FA / TARGET FA (pooled, 155 set, mean over seeds)")
say("%-7s %10s %12s %12s %8s %8s" % ("target", "tgt/day", "Tamp ach", "Trel ach", "Tamp x",
                                     "Trel x"))
for j, t in enumerate(TG):
    a = gmean("Tamp", ones, j)[1]
    r = gmean("Trel", ones, j)[1]
    say("%-7.2f %10.1f %12.1f %12.1f %8.1f %8.1f" % (t, 24 * t, a, r, a / (24 * t), r / (24 * t)))
say("Per case at target 2.0 (48/day): achieved/target, mean over seeds")
ra = np.mean([Z[a + "__fp"][:, j2] for a in AMP], 0) / day / 48
rr_ = np.mean([Z[a + "__fp"][:, j2] for a in REL], 0) / day / 48
for i, c in enumerate(cases):
    pass
say("  Tamp: median %.2f, cases over target %d of %d, cases over 3x %d"
    % (np.median(ra), (ra > 1).sum(), len(ra), (ra > 3).sum()))
say("  Trel: median %.2f, cases over target %d of %d, cases over 3x %d"
    % (np.median(rr_), (rr_ > 1).sum(), len(rr_), (rr_ > 3).sum()))
say("  cases where Trel ratio < Tamp ratio: %d of %d" % ((rr_ < ra).sum(), len(ra)))
say("  per case ratios Tamp / Trel: " + ", ".join("%s %.2f/%.2f" % (c, x, y)
                                                 for c, x, y in zip(cases, ra, rr_)))
say("Per case at target 0.05 (1.2/day): achieved/target")
j0 = 0
ra0 = np.mean([Z[a + "__fp"][:, j0] for a in AMP], 0) / day / 1.2
rr0 = np.mean([Z[a + "__fp"][:, j0] for a in REL], 0) / day / 1.2
say("  Tamp: median %.1f, cases with zero FA %d; Trel: median %.1f, cases with zero FA %d"
    % (np.median(ra0), (ra0 == 0).sum(), np.median(rr0), (rr0 == 0).sum()))
fa0a = np.mean([Z[a + "__fp"][:, j0] for a in AMP], 0)
fa0r = np.mean([Z[a + "__fp"][:, j0] for a in REL], 0)
o = np.argsort(-fa0a)
say("  Tamp's floor at target 0.05: largest FA cases (Tamp fp / Trel fp, hours): " + ", ".join(
    "%s %.1f/%.1f (%.1fh)" % (cases[i], fa0a[i], fa0r[i], sec[i] / 3600) for i in o[:6]))
say("  share of Tamp fp at target 0.05 from top 1/2/3 cases: %.1f%% %.1f%% %.1f%%" % tuple(
    100 * fa0a[o[:k]].sum() / fa0a.sum() for k in (1, 2, 3)))
thr_a = np.mean([Z[a + "__thr"] for a in AMP], 0)
thr_r = np.mean([Z[a + "__thr"] for a in REL], 0)
say("  thresholds at target 0.05: Tamp median %.3f (max %.3f, # == 1.0: %d); Trel median %.3f "
    "(max %.3f)" % (np.median(thr_a[:, 0]), thr_a[:, 0].max(),
                    int(sum((Z[a + "__thr"][:, 0] >= 0.999999).sum() for a in AMP)),
                    np.median(thr_r[:, 0]), thr_r[:, 0].max()))
# transfer ratio excluding the top fp-reduction cases
for drop in ([rk[0][0]], [rk[0][0], rk[1][0]], [r[0] for r in rk[:3]]):
    wx = np.array([0.0 if c in drop else 1.0 for c in cases])
    say("  without %s: achieved/target at 2.0 Tamp %.2f Trel %.2f; at 0.05 Tamp %.1f Trel %.1f;"
        " README FA Tamp %.1f Trel %.1f, caught Tamp %.2f/%d Trel %.2f/%d"
        % ("+".join(drop), gmean("Tamp", wx, j2)[1] / 48, gmean("Trel", wx, j2)[1] / 48,
           gmean("Tamp", wx, 0)[1] / 1.2, gmean("Trel", wx, 0)[1] / 1.2,
           gmean("Tamp", wx, j2)[1], gmean("Trel", wx, j2)[1], gmean("Tamp", wx, j2)[0],
           gmean("Tamp", wx, j2)[2], gmean("Trel", wx, j2)[0], gmean("Trel", wx, j2)[2]))

with open(os.path.join(OUT, "analysis.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(LINES) + "\n")
