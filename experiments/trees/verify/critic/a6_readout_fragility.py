"""critic step F (Q1): how fragile is the rule's readout? Readout (best pooled sensitivity among the 14
targets whose achieved FA/day <= budget; 0 if none) per arm under leave-one-subject-out, and at
post-hoc budgets. Reads verify/critic/deploy_tables.npz."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import numpy as np

ME = Path(r"C:\Users\user\AppData\Local\Temp\claude\36c1f29744f93fe6"
          r"\b3889703-629c-4c5a-b583-69f722743ff4\scratchpad\trees\verify\critic")
D = np.load(ME / "deploy_tables.npz")
G = {"A0": ["A0"], "Tamp": ["Tamp_s%d" % s for s in range(3)],
     "Tfree": ["Tfree_s%d" % s for s in range(3)]}
cases = ["CHB%02d" % i for i in range(1, 25)]


def readout(a, b, w):
    tp, fp, ref, sec = (w @ D[a + "__tp"], w @ D[a + "__fp"], w @ D[a + "__ref"], w @ D[a + "__sec"])
    fa = fp / (sec / 86400)
    ok = np.flatnonzero(fa <= b + 1e-12)
    return 0.0 if not len(ok) else tp[ok].max() / ref


def g(gname, b, w):
    return np.mean([readout(a, b, w) for a in G[gname]])


L = []
one = np.ones(24)
L.append("readout at post-hoc budgets (all cases): Tamp-A0 and verdict-style sign")
for b in (25, 30, 40, 50, 60, 75, 100, 125):
    L.append("  <=%3d/day: A0 %.3f Tamp %.3f Tfree %.3f | Tamp-A0 %+.3f" % (
        b, g("A0", b, one), g("Tamp", b, one), g("Tfree", b, one), g("Tamp", b, one) - g("A0", b, one)))
L.append("leave-one-subject-out readout at 25/day (only subjects whose removal makes any arm non-zero):")
subs = [c for c in cases if c != "CHB21"]
for s in subs:
    w = np.array([0.0 if (c == s or (s == "CHB01" and c == "CHB21")) else 1.0 for c in cases])
    r = {k: g(k, 25, w) for k in G}
    if any(v > 0 for v in r.values()):
        L.append("  without %s: A0 %.3f Tamp %.3f Tfree %.3f -> Tamp-A0 %+.3f" % (
            s, r["A0"], r["Tamp"], r["Tfree"], r["Tamp"] - r["A0"]))
for drop in (["CHB06"], ["CHB15"], ["CHB06", "CHB15"], ["CHB06", "CHB09"], ["CHB15", "CHB09"]):
    w = np.array([0.0 if c in drop else 1.0 for c in cases])
    L.append("  without %s: strictest-target FA/day A0 %.1f, Tamp %s; readout@25 A0 %.3f Tamp %.3f; @10 A0 %.3f Tamp %.3f"
             % ("+".join(drop), (w @ D["A0__fp"])[0] / ((w @ D["A0__sec"]) / 86400),
                ",".join("%.1f" % ((w @ D[a + "__fp"])[0] / ((w @ D[a + "__sec"]) / 86400)) for a in G["Tamp"]),
                g("A0", 25, w), g("Tamp", 25, w), g("A0", 10, w), g("Tamp", 10, w)))
(ME / "a6_readout_fragility.txt").write_text("\n".join(L) + "\n", encoding="ascii")
print("\n".join(L))
