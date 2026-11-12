"""Recompute deployable curves, readout, README operating point, bootstrap from
out/preds_{a,b,c}.npy + thresholds_{a,b,c}.json. Run with cwd C:\\EEG, python -B."""
import sys
sys.dont_write_bytecode = True
import numpy as np

sys.path.insert(0, r"C:\EEG")
from rs_common import (load_all, my_smooth, my_detect_from_smoothed, case_count_sm,
                       TARGETS, BUDGETS, ARMS, GROUPS, HERE, K, MINRUN)

lines = []


def say(s=""):
    print(s, flush=True)
    lines.append(s)


P, T = load_all()
say("arms in files: " + ", ".join(sorted(P)))
assert sorted(P) == sorted(ARMS), sorted(P)
cases = sorted(P["A0"])
say("cases: %d" % len(cases))

# ---- structural checks -------------------------------------------------------------
for a in ARMS:
    assert sorted(P[a]) == cases and sorted(T[a]) == cases, a
    for c in cases:
        assert len(P[a][c]) == len(P["A0"][c]), (a, c)
        for (p, y), (p0, y0) in zip(P[a][c], P["A0"][c]):
            assert len(p) == len(y) == len(y0) and np.array_equal(y, y0), (a, c)
            assert np.all(np.isfinite(p)) and p.min() >= 0 and p.max() <= 1, (a, c)
            assert len(y) >= 10
        assert sorted(T[a][c]) == sorted(TARGETS), (a, c)
say("structure: every arm has the same 24 cases, block lengths and labels as A0; "
    "probs finite in [0,1]; 14 targets each")

# fold-mates share thresholds
for a in ARMS:
    assert T[a]["CHB01"] == T[a]["CHB21"], a
say("CHB01 and CHB21 share thresholds in every arm")

# threshold monotone in target (stricter target -> higher or equal threshold)
nonmono = 0
for a in ARMS:
    for c in cases:
        v = [T[a][c][t] for t in TARGETS]
        nonmono += int(np.any(np.diff(v) > 1e-12))
say("thresholds non-increasing in target for every arm/case: %s" % (nonmono == 0))

# ---- A0 vs endtoend_uncalibrated.npz -----------------------------------------------
ref = np.load(r"C:\EEG\endtoend_uncalibrated.npz", allow_pickle=True)["data"].item()
assert sorted(ref) == cases
dp = dt = 0.0
nb = 0
for c in cases:
    assert len(ref[c]["blocks"]) == len(P["A0"][c]), c
    for (p1, y1), (p2, y2) in zip(P["A0"][c], ref[c]["blocks"]):
        assert len(p1) == len(p2) and np.array_equal(np.asarray(y1, int), np.asarray(y2, int))
        dp = max(dp, float(np.max(np.abs(np.asarray(p1, float) - np.asarray(p2, float)))))
        nb += 1
    dt = max(dt, abs(T["A0"][c]["2.0"] - float(ref[c]["t"])))
say("A0 vs endtoend_uncalibrated.npz: %d blocks, labels identical, max|dp| %.3g, "
    "max|t(2.0) - t_npz| %.3g" % (nb, dp, dt))

# ---- smoothing / count vs repo implementation ---------------------------------------
from evaluate_temporal import smooth as repo_smooth
from score_official import count as repo_count
ds = 0.0
for a in ARMS:
    for c in cases:
        for p, _ in P[a][c]:
            ds = max(ds, float(np.max(np.abs(my_smooth(p) - repo_smooth(np.asarray(p, float), K)))))
say("my smooth vs evaluate_temporal.smooth: max diff %.3g" % ds)

SM = {a: {c: [my_smooth(p) for p, _ in P[a][c]] for c in cases} for a in ARMS}
Y = {c: [np.asarray(y) for _, y in P["A0"][c]] for c in cases}

tot_ref = tot_sec = 0
D = {}
for a in ARMS:
    tp = np.zeros((len(cases), len(TARGETS)), int)
    fp = np.zeros_like(tp)
    rf = np.zeros(len(cases), int)
    sc = np.zeros(len(cases))
    for i, c in enumerate(cases):
        for j, tg in enumerate(TARGETS):
            a_, b_, r_, s_ = case_count_sm(SM[a][c], Y[c], float(T[a][c][tg]))
            tp[i, j], fp[i, j], rf[i], sc[i] = a_, b_, r_, s_
    D[a] = dict(tp=tp, fp=fp, ref=rf, sec=sc)
mism = 0
for a in ARMS:
    j = TARGETS.index("2.0")
    for i, c in enumerate(cases):
        r = repo_count([(np.asarray(p, float), y) for p, y in P[a][c]],
                       float(T[a][c]["2.0"]), K, MINRUN)
        if (r[0], r[1], r[2]) != (D[a]["tp"][i, j], D[a]["fp"][i, j], D[a]["ref"][i]) or \
                abs(r[3] - D[a]["sec"][i]) > 1e-6:
            mism += 1
            say("  count mismatch %s %s mine %s repo %s" % (a, c, (D[a]["tp"][i, j],
                D[a]["fp"][i, j], D[a]["ref"][i]), r))
say("my count vs score_official.count at target 2.0, all arms x cases: %d mismatches" % mism)
R0 = D["A0"]["ref"].sum()
H0 = D["A0"]["sec"].sum() / 3600
say("test set: %d reference events, %.2f h, %d windows" %
    (R0, H0, sum(len(y) for c in cases for y in Y[c])))
say()

# ---- deployable curves ---------------------------------------------------------------
def pooled(a, j):
    d = D[a]
    return d["tp"][:, j].sum() / d["ref"].sum(), d["fp"][:, j].sum() / (d["sec"].sum() / 86400)


say("DEPLOYABLE CURVES per arm: sens (FA/day)")
say("%-6s" % "tgt" + "".join("%16s" % a for a in ARMS))
for j, tg in enumerate(TARGETS):
    say("%-6s" % tg + "".join("%16s" % ("%.3f (%6.1f)" % pooled(a, j)) for a in ARMS))
say()
say("DEPLOYABLE CURVES, group means (trees mean over seeds)")
say("%-6s" % "tgt" + "".join("%18s" % g for g in GROUPS))
for j, tg in enumerate(TARGETS):
    row = "%-6s" % tg
    for g, arms in GROUPS.items():
        s = np.mean([pooled(a, j)[0] for a in arms])
        f = np.mean([pooled(a, j)[1] for a in arms])
        row += "%18s" % ("%.3f (%6.1f)" % (s, f))
    say(row)
say()
say("caught/168 per arm per target")
say("%-6s" % "tgt" + "".join("%10s" % a for a in ARMS))
for j, tg in enumerate(TARGETS):
    say("%-6s" % tg + "".join("%10s" % ("%d/%d" % (D[a]["tp"][:, j].sum(), D[a]["ref"].sum()))
                             for a in ARMS))
say()


# ---- readout --------------------------------------------------------------------------
def readout(a, b, w=None):
    d = D[a]
    w = np.ones(len(cases)) if w is None else w
    tp, fp, rf, sc = w @ d["tp"], w @ d["fp"], w @ d["ref"], w @ d["sec"]
    fa = fp / (sc / 86400)
    ok = np.flatnonzero(fa <= b + 1e-12)
    if not len(ok):
        return 0.0, float("nan"), None
    j = ok[np.argmax(tp[ok])]
    return tp[j] / rf, fa[j], TARGETS[j]


say("READOUT (best deployable sens with achieved FA/day <= budget; 0 if none)")
say("%-9s" % "" + "".join("%22s" % ("<=%d/d" % b) for b in BUDGETS))
R = {}
for a in ARMS:
    R[a] = {b: readout(a, b) for b in BUDGETS}
    say("%-9s" % a + "".join("%22s" % ("%.3f (%5.1f @%s)" % (R[a][b][0], R[a][b][1], R[a][b][2]))
                             for b in BUDGETS))
for g, arms in GROUPS.items():
    R[g] = {b: float(np.mean([R[a][b][0] for a in arms])) for b in BUDGETS}
    say("%-9s" % g + "".join("%22.3f" % R[g][b] for b in BUDGETS))
say("lowest achieved FA/day on each arm's curve: " +
    ", ".join("%s %.1f" % (a, min(pooled(a, j)[1] for j in range(len(TARGETS)))) for a in ARMS))
d25 = R["Tamp"][25] - R["A0"][25]
d10 = R["Tamp"][10] - R["A0"][10]
say("PRIMARY Tamp-A0: 25/d %+.3f, 10/d %+.3f; AMPLITUDE Tamp-Tfree 25/d %+.3f; "
    "Tfree-A0 25/d %+.3f 10/d %+.3f" % (d25, d10, R["Tamp"][25] - R["Tfree"][25],
                                        R["Tfree"][25] - R["A0"][25],
                                        R["Tfree"][10] - R["A0"][10]))
say()

j2 = TARGETS.index("2.0")
say("README OPERATING POINT (target 2 FA/h)")
for a in ARMS:
    s, f = pooled(a, j2)
    say("  %-9s %3d/%d  %6.1f FA/day  (sens %.4f)" % (a, D[a]["tp"][:, j2].sum(),
                                                     D[a]["ref"].sum(), f, s))
say()

# ---- bootstrap (same recipe and seed as described, recomputed independently) ----------
subj = []
for c in cases:
    s = "CHB01" if c == "CHB21" else c
    if s not in subj:
        subj.append(s)
S = np.array([[1.0 if (c == s or (s == "CHB01" and c == "CHB21")) else 0.0 for c in cases]
              for s in subj])


def contrast(w, b, g1="Tamp", g0="A0"):
    return (np.mean([readout(a, b, w)[0] for a in GROUPS[g1]]) -
            np.mean([readout(a, b, w)[0] for a in GROUPS[g0]]))


rng = np.random.default_rng(20261104)
idx = rng.integers(0, len(subj), size=(2000, len(subj)))
W = np.zeros((2000, len(subj)))
for r in range(2000):
    np.add.at(W[r], idx[r], 1)
W = W @ S
boot = {b: np.array([contrast(W[r], b) for r in range(2000)]) for b in (10, 25)}
bamp = np.array([contrast(W[r], 25, "Tamp", "Tfree") for r in range(2000)])
say("BOOTSTRAP (seed 20261104, 2000 draws, %d subjects)" % len(subj))
for b in (10, 25):
    lo, hi = np.percentile(boot[b], [2.5, 97.5])
    say("  Tamp-A0 %d/d: [%+.3f, %+.3f] P(<=0) %.3f  share of draws where the A0 readout "
        "is 0 (budget unreachable): %.3f" % (b, lo, hi, (boot[b] <= 0).mean(),
        np.mean([readout("A0", b, W[r])[2] is None for r in range(2000)])))


def verdict(x25, x10):
    if x25 >= 0.05 and x10 >= 0:
        return "TREES BETTER"
    if x25 <= -0.05:
        return "TREES WORSE"
    return "SAME"


vs = [verdict(boot[25][r], boot[10][r]) for r in range(2000)]
say("  verdict share: " + ", ".join("%s %.1f%%" % (v, 100.0 * vs.count(v) / 2000)
                                    for v in ("TREES BETTER", "SAME", "TREES WORSE")))
lo, hi = np.percentile(bamp, [2.5, 97.5])
say("  Tamp-Tfree 25/d: [%+.3f, %+.3f]" % (lo, hi))
loo = [contrast(1.0 - S[i], 25) for i in range(len(subj))]
say("  LOSO Tamp-A0 25/d: min %+.3f (without %s), max %+.3f (without %s)"
    % (min(loo), subj[int(np.argmin(loo))], max(loo), subj[int(np.argmax(loo))]))
say("  LOSO values: " + " ".join("%s %+.3f" % (s, v) for s, v in zip(subj, loo)))

np.savez(HERE / "deploy_tables.npz", cases=np.array(cases),
         **{"%s__%s" % (a, k): D[a][k] for a in ARMS for k in D[a]})
(HERE / "rs_deploy_out.txt").write_text("\n".join(lines) + "\n", encoding="ascii")
