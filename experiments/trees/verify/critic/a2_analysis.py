"""critic step B: readout mechanics, matched-FA contrasts with subject bootstrap, per-case
influence, oracle and AUC breakdowns. Reads verify/critic/deploy_tables.npz, out/oracle_*.npz,
out/preds.npy. Writes verify/critic/a2_results.txt."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon, binomtest
from sklearn.metrics import roc_auc_score
sys.path.insert(0, r"C:\EEG")
from evaluate_temporal import smooth  # noqa: E402

TREES = Path(r"C:\Users\user\AppData\Local\Temp\claude\36c1f29744f93fe6"
             r"\b3889703-629c-4c5a-b583-69f722743ff4\scratchpad\trees")
OUT, ME = TREES / "out", TREES / "verify" / "critic"
D = np.load(ME / "deploy_tables.npz")
TG = [float(x) for x in D["targets"]]
G = {"A0": ["A0"], "Tamp": ["Tamp_s%d" % s for s in range(3)],
     "Tfree": ["Tfree_s%d" % s for s in range(3)]}
P = np.load(OUT / "preds.npy", allow_pickle=True).item()
cases = sorted(P["A0"])
nC = len(cases)
subj = [c for c in cases if c != "CHB21"]
S = np.array([[1.0 if (c == s or (s == "CHB01" and c == "CHB21")) else 0.0 for c in cases]
              for s in subj])
lines = []


def say(s=""):
    print(s, flush=True)
    lines.append(s)


def tab(a):
    return D[a + "__tp"], D[a + "__fp"], D[a + "__ref"], D[a + "__sec"]


def curve(a, w=None):
    tp, fp, ref, sec = tab(a)
    w = np.ones(nC) if w is None else w
    return (w @ fp) / ((w @ sec) / 86400.0), (w @ tp) / (w @ ref)


def interp(a, F, w=None):
    fa, se = curve(a, w)
    o = np.argsort(fa)
    fa, se = fa[o], se[o]
    out = np.interp(np.log(F), np.log(fa), se)
    out = np.where((np.asarray(F) < fa[0] - 1e-9) | (np.asarray(F) > fa[-1] + 1e-9), np.nan, out)
    return out


def ginterp(g, F, w=None):
    return np.mean([interp(a, F, w) for a in G[g]], axis=0)


# ------------------------------------------------------------------ Q1
say("Q1  READOUT MECHANICS")
for g in G:
    for a in G[g]:
        fa, se = curve(a)
        tp, fp, ref, sec = tab(a)
        j = 0
        pc = fp[:, j] / (sec / 86400.0)
        top = np.argsort(-fp[:, j])[:4]
        say("  %-9s strictest target 0.05 FA/h (=1.2/day on training bg): test %.1f FA/day, "
            "sens %.3f; min over targets %.1f/day; FAs %d, top cases %s (%.0f%% of FAs)"
            % (a, fa[0], se[0], fa.min(), fp[:, j].sum(),
               ",".join("%s:%d" % (cases[i], fp[i, j]) for i in top),
               100.0 * fp[top, j].sum() / fp[:, j].sum()))
fa0, _ = curve("A0")
say("  A0 test FA/h over target, per target: " +
    " ".join("%.2g:%.1fx" % (t, f / 24.0 / t) for t, f in zip(TG, fa0)))
fam = np.mean([curve(a)[0] for a in G["Tamp"]], 0)
say("  Tamp test FA/h over target:            " +
    " ".join("%.2g:%.1fx" % (t, f / 24.0 / t) for t, f in zip(TG, fam)))
# how much of the strictest-target FA is from cases with short test periods / few cases
tp, fp, ref, sec = tab("A0")
say("  test hours %.1f; cases with 0 FA at strictest target: A0 %d, Tamp_s0 %d"
    % (sec.sum() / 3600, (fp[:, 0] == 0).sum(), (tab("Tamp_s0")[1][:, 0] == 0).sum()))
for b in (25, 10):
    for g in G:
        need = []
        for a in G[g]:
            fa, _ = curve(a)
            need.append(fa.min() / b)
        say("  budget %d/day: %s strictest point is %.1fx over budget" % (b, g, np.mean(need)))
say()

# ------------------------------------------------------------------ Q2 matched FA
FGRID = np.array([40, 50, 60, 75, 90, 100, 115, 125, 150])
say("Q2  DEPLOYABLE CURVES AT MATCHED ACHIEVED FA/DAY (log-FA interpolation, per seed then mean)")
say("  FA/day   " + " ".join("%7d" % f for f in FGRID))
R = {g: ginterp(g, FGRID) for g in G}
for g in G:
    say("  %-8s " % g + " ".join("%7.3f" % v for v in R[g]))
say("  Tamp-A0  " + " ".join("%+7.3f" % v for v in R["Tamp"] - R["A0"]))
say("  Tfree-A0 " + " ".join("%+7.3f" % v for v in R["Tfree"] - R["A0"]))
say("  Tamp-Tfr " + " ".join("%+7.3f" % v for v in R["Tamp"] - R["Tfree"]))
for a in G["Tamp"]:
    say("  %-8s " % a + " ".join("%+7.3f" % v for v in interp(a, FGRID) - R["A0"]))
BAND = np.exp(np.linspace(np.log(50), np.log(150), 21))


def band(g1, g0, w=None):
    d = ginterp(g1, BAND, w) - ginterp(g0, BAND, w)
    return np.nan if np.isnan(d).any() else d.mean()


say("  band mean over 50-150 FA/day: Tamp-A0 %+.3f, Tfree-A0 %+.3f, Tamp-Tfree %+.3f"
    % (band("Tamp", "A0"), band("Tfree", "A0"), band("Tamp", "Tfree")))
rng = np.random.default_rng(7)
NB = 2000
idx = rng.integers(0, len(subj), size=(NB, len(subj)))
W = np.zeros((NB, len(subj)))
for r in range(NB):
    np.add.at(W[r], idx[r], 1)
W = W @ S
say("  SUBJECT BOOTSTRAP (2000 draws, 23 subjects, CHB01+CHB21 one subject)")
for (g1, g0) in (("Tamp", "A0"), ("Tfree", "A0"), ("Tamp", "Tfree")):
    for F in (50, 75, 100, 125):
        bs = np.array([ginterp(g1, [F], W[r])[0] - ginterp(g0, [F], W[r])[0] for r in range(NB)])
        ok = ~np.isnan(bs)
        lo, hi = np.percentile(bs[ok], [2.5, 97.5])
        say("    %s-%s at %3d/day: point %+.3f  95%% [%+.3f, %+.3f]  P(<=0) %.3f  defined %.0f%%"
            % (g1, g0, F, ginterp(g1, [F])[0] - ginterp(g0, [F])[0], lo, hi,
               (bs[ok] <= 0).mean(), 100 * ok.mean()))
    bs = np.array([band(g1, g0, W[r]) for r in range(NB)])
    ok = ~np.isnan(bs)
    lo, hi = np.percentile(bs[ok], [2.5, 97.5])
    say("    %s-%s band 50-150: point %+.3f  95%% [%+.3f, %+.3f]  P(<=0) %.3f  defined %.0f%%"
        % (g1, g0, band(g1, g0), lo, hi, (bs[ok] <= 0).mean(), 100 * ok.mean()))
say("  LEAVE-ONE-SUBJECT-OUT, Tamp-A0 band 50-150 and at 100/day:")
loo = []
for i, s in enumerate(subj):
    w = 1.0 - S[i]
    loo.append((s, band("Tamp", "A0", w), ginterp("Tamp", [100], w)[0] - ginterp("A0", [100], w)[0]))
loo.sort(key=lambda x: x[1])
say("    " + " ".join("%s:%+.3f/%+.3f" % x for x in loo))
for drop in (["CHB21"], ["CHB01", "CHB21"], ["CHB05"], ["CHB15"], ["CHB14"], ["CHB20"],
             ["CHB01", "CHB21", "CHB05", "CHB15"], ["CHB14", "CHB20"],
             ["CHB01", "CHB21", "CHB05", "CHB15", "CHB14", "CHB20"]):
    w = np.array([0.0 if c in drop else 1.0 for c in cases])
    say("    without %-34s band Tamp-A0 %+.3f | at 50/75/100/125: %s"
        % ("+".join(drop), band("Tamp", "A0", w),
           " ".join("%+.3f" % v for v in ginterp("Tamp", [50, 75, 100, 125], w)
                    - ginterp("A0", [50, 75, 100, 125], w))))
say()

# README operating point per case
j2 = TG.index(2.0)
say("Q2  README OPERATING POINT (target 2 FA/h) PER CASE: caught (A0 | Tamp s0,s1,s2) and FA")
tpA, fpA, ref, sec = tab("A0")
tpT = np.stack([tab(a)[0][:, j2] for a in G["Tamp"]])
fpT = np.stack([tab(a)[1][:, j2] for a in G["Tamp"]])
tpF = np.stack([tab(a)[0][:, j2] for a in G["Tfree"]])
for i, c in enumerate(cases):
    dt = tpT[:, i].mean() - tpA[i, j2]
    if abs(dt) >= 0.5 or abs(fpT[:, i].mean() - fpA[i, j2]) >= 5:
        say("  %s ref %2d  caught A0 %2d | Tamp %s (Tfree %s)  FA A0 %3d | Tamp %s  hours %.1f"
            % (c, ref[i], tpA[i, j2], ",".join(str(x) for x in tpT[:, i]),
               ",".join(str(x) for x in tpF[:, i]), fpA[i, j2],
               ",".join(str(x) for x in fpT[:, i]), sec[i] / 3600))
say("  totals: caught A0 %d, Tamp mean %.1f; FA A0 %d, Tamp mean %.1f"
    % (tpA[:, j2].sum(), tpT.sum(1).mean(), fpA[:, j2].sum(), fpT.sum(1).mean()))
for drop in (["CHB21"], ["CHB01", "CHB21", "CHB05", "CHB15"]):
    k = np.array([c not in drop for c in cases])
    say("  without %s: caught A0 %d vs Tamp %.1f; FA/day A0 %.1f vs Tamp %.1f"
        % ("+".join(drop), tpA[k, j2].sum(), tpT[:, k].sum(1).mean(),
           fpA[k, j2].sum() / (sec[k].sum() / 86400), fpT[:, k].sum(1).mean() / (sec[k].sum() / 86400)))
say()

# per-patient oracle per case
say("Q2  PER-PATIENT ORACLE PER CASE (seizures caught at each case's own best threshold)")
O = {a: np.load(OUT / ("oracle_%s.npz" % a)) for g in G for a in G[g]}


def pp(a, b):
    o = O[a]
    fa = o["fp"] / (o["sec"][:, None] / 86400.0)
    return np.where(fa <= b + 1e-12, o["tp"], 0).max(1)


def pooled_or(a, b, w=None):
    o = O[a]
    w = np.ones(nC) if w is None else w
    tp, fp = w @ o["tp"], w @ o["fp"]
    fa = fp / ((w @ o["sec"]) / 86400.0)
    ok = fa <= b + 1e-12
    return tp[ok].max() / (w @ o["ref"])


refo = O["A0"]["ref"]
for b in (10, 25):
    a0 = pp("A0", b)
    ta = np.mean([pp(a, b) for a in G["Tamp"]], 0)
    tf = np.mean([pp(a, b) for a in G["Tfree"]], 0)
    d = ta - a0
    o = np.argsort(-np.abs(d))
    say("  <=%d/day: total A0 %.0f, Tamp %.1f, Tfree %.1f of %d. Largest Tamp-A0: %s"
        % (b, a0.sum(), ta.sum(), tf.sum(), refo.sum(),
           " ".join("%s:%+.1f" % (cases[i], d[i]) for i in o[:8])))
    k = np.array([c not in ("CHB01", "CHB21", "CHB05", "CHB15") for c in cases])
    say("     Tamp-A0 total %+.1f; without CHB01/21/05/15 %+.1f; cases Tamp>A0 %d, <A0 %d"
        % (d.sum(), d[k].sum(), (d > 0).sum(), (d < 0).sum()))
    ppw = np.array([[pp(a, b)[i] for i in range(nC)] for a in ["A0"] + G["Tamp"]])
    bs = []
    for r in range(NB):
        w = W[r]
        bs.append(np.mean([w @ ppw[1 + s] for s in range(3)]) / (w @ refo) - (w @ ppw[0]) / (w @ refo))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    say("     subject bootstrap Tamp-A0 per-patient oracle: %+.3f  95%% [%+.3f, %+.3f]  P(<=0) %.3f"
        % (d.sum() / refo.sum(), lo, hi, (np.array(bs) <= 0).mean()))
for b in (10, 25, 50):
    say("  pooled oracle <=%d/day: A0 %.3f | Tamp %s | Tfree %s" % (
        b, pooled_or("A0", b), " ".join("%.3f" % pooled_or(a, b) for a in G["Tamp"]),
        " ".join("%.3f" % pooled_or(a, b) for a in G["Tfree"])))
say()

# AUC per case, subject level
say("Q2/Q3  WINDOW AUC (smoothed k=5), subject level")
auc = {}
for g in G:
    auc[g] = np.array([np.mean([roc_auc_score(
        np.concatenate([y for _, y in P[a][c]]),
        np.concatenate([smooth(np.asarray(p, np.float64), 5) for p, _ in P[a][c]]))
        for a in G[g]]) for c in cases])
aucS = {g: (S @ auc[g]) / S.sum(1) for g in G}
for g1, g0 in (("Tamp", "A0"), ("Tfree", "A0"), ("Tamp", "Tfree")):
    d = aucS[g1] - aucS[g0]
    say("  %s-%s: subjects better %d/23, median %+.3f, mean %+.3f, Wilcoxon p %.3f, sign p %.3f"
        % (g1, g0, (d > 0).sum(), np.median(d), d.mean(), wilcoxon(d).pvalue,
           binomtest(int((d > 0).sum()), 23).pvalue))
    for drop in (["CHB01"], ["CHB01", "CHB05"], ["CHB01", "CHB05", "CHB15"]):
        k = np.array([s not in drop for s in subj])
        say("     without %s: better %d/%d, Wilcoxon p %.3f"
            % ("+".join(drop) + ("(+21)" if "CHB01" in drop else ""), (d[k] > 0).sum(), k.sum(),
               wilcoxon(d[k]).pvalue))
say("  AUC-difference vs seizures weighting: Tamp-A0 per case, ref seizures")
d = auc["Tamp"] - auc["A0"]
say("    " + " ".join("%s:%+.2f(%d)" % (c, d[i], refo[i]) for i, c in enumerate(cases)))
(ME / "a2_results.txt").write_text("\n".join(lines) + "\n", encoding="ascii")
np.savez(ME / "auc_per_case.npz", cases=np.array(cases), **auc)
