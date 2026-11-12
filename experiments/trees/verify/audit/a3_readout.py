"""Audit step 3: what the readout does at budgets no target reaches. Read-only."""
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np

TR = Path(r"C:\Users\user\AppData\Local\Temp\claude\36c1f29744f93fe6"
          r"\b3889703-629c-4c5a-b583-69f722743ff4\scratchpad\trees")
OUT = TR / "verify" / "audit"
sys.path.insert(0, str(TR))
import trees_score as ts                                               # noqa: E402

ARMS = ["A0", "Tamp_s0", "Tamp_s1", "Tamp_s2", "Tfree_s0", "Tfree_s1", "Tfree_s2"]
D = {a: dict(np.load(OUT / ("D_%s.npz" % a))) for a in ARMS}
cases = list(D["A0"]["cases"])
G = {"A0": ["A0"], "Tamp": ARMS[1:4], "Tfree": ARMS[4:7]}

print("pooled events %d, hours %.2f" % (D["A0"]["ref"].sum(), D["A0"]["sec"].sum() / 3600))
print("\nstrictest achieved pooled test FA/day per arm (target 0.05 FA/h) and sens there")
for a in ARMS:
    fa = D[a]["fp"].sum(0) / (D[a]["sec"].sum() / 86400)
    j = int(np.argmin(fa))
    print("  %-9s min FA/day %6.1f at target %s, sens %.3f | monotone FA in target: %s"
          % (a, fa[j], ts.TARGETS[j], D[a]["tp"][:, j].sum() / D[a]["ref"].sum(),
             bool(np.all(np.diff(fa) >= 0))))

print("\nreadout as printed (0.0 fill) vs undefined (None target) per budget")
for a in ARMS:
    row = []
    for b in ts.BUDGETS:
        s, f, t = ts.readout(D[a], b)
        row.append("%s" % ("UNDEF" if t is None else "%.3f@%s" % (s, t)))
    print("  %-9s " % a + "  ".join("%14s" % r for r in row))

# Primary contrast with nothing at 25/day: any defined?
subj = []
for c in cases:
    s = "CHB01" if c == "CHB21" else c
    if s not in subj:
        subj.append(s)
S = np.array([[1.0 if (c == s or (s == "CHB01" and c == "CHB21")) else 0.0 for c in cases]
              for s in subj])
rng = np.random.default_rng(20261104)
idx = rng.integers(0, len(subj), size=(2000, len(subj)))
Wc = np.zeros((2000, len(subj)))
for r in range(2000):
    np.add.at(Wc[r], idx[r], 1)
Wc = Wc @ S
for b in (10, 25):
    defined = {a: np.array([ts.readout(D[a], b, Wc[r])[2] is not None for r in range(2000)])
               for a in ARMS}
    allA = defined["A0"]
    allT = np.all([defined[a] for a in G["Tamp"]], axis=0)
    anyT = np.any([defined[a] for a in G["Tamp"]], axis=0)
    print("\nbootstrap at %d/day: draws with A0 defined %d, all 3 Tamp defined %d, any Tamp "
          "defined %d, both A0 and all Tamp defined %d, neither A0 nor any Tamp %d"
          % (b, allA.sum(), allT.sum(), anyT.sum(), (allA & allT).sum(),
             (~allA & ~anyT).sum()))
    con = np.array([np.mean([ts.readout(D[a], b, Wc[r])[0] for a in G["Tamp"]])
                    - ts.readout(D["A0"], b, Wc[r])[0] for r in range(2000)])
    print("  contrast as printed: share exactly 0 %.3f, share > 0 %.3f, share < 0 %.3f"
          % ((con == 0).mean(), (con > 0).mean(), (con < 0).mean()))
    both = allA & allT
    if both.any():
        print("  among draws where both are defined: median %+.3f, 2.5-97.5%% [%+.3f, %+.3f]"
              % (np.median(con[both]), *np.percentile(con[both], [2.5, 97.5])))
    only = allA & ~anyT
    print("  draws where only A0 is defined: %d, their contrast mean %+.3f"
          % (only.sum(), con[only].mean() if only.any() else np.nan))
    onlyT = ~allA & anyT
    print("  draws where only (some) Tamp is defined: %d, their contrast mean %+.3f"
          % (onlyT.sum(), con[onlyT].mean() if onlyT.any() else np.nan))

print("\nleave-one-subject-out at 25/day: which arms are defined")
for i, s in enumerate(subj):
    w = 1.0 - S[i]
    r = {a: ts.readout(D[a], 25, w) for a in ARMS}
    dflt = [a for a in ARMS if r[a][2] is not None]
    if dflt:
        print("  without %-6s defined: %s" % (s, ", ".join("%s %.3f@%.1f/d" % (a, r[a][0], r[a][1])
                                                        for a in dflt)))
print("\nmatched-FA view (not the rule): per-arm sens at the target whose achieved FA is")
print("closest to A0's, and the 100/day and 50/day readouts")
for b in (50, 100):
    for g, arms in G.items():
        rs = [ts.readout(D[a], b) for a in arms]
        print("  %3d/d %-6s per arm %s -> mean %.3f" % (
            b, g, ", ".join("UNDEF" if t is None else "%.3f@%s(%.1f/d)" % (s, t, f)
                            for s, f, t in rs), np.mean([x[0] for x in rs])))
