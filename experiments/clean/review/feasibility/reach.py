# Arithmetic on the precomputed old deploy tables only (no scoring is run).
import numpy as np
D = np.load(r"C:\Users\user\AppData\Local\Temp\claude\36c1f29744f93fe6\b3889703-629c-4c5a-b583-69f722743ff4\scratchpad\trees\verify\critic\deploy_tables.npz", allow_pickle=True)
arms = sorted({k.split("__")[0] for k in D.files if "__" in k})
print("arms", arms)
cases = 24
# subjects: CHB01+CHB21 merged; cases sorted CHB01..CHB24, CHB21 index 20
names = ["CHB%02d" % i for i in range(1, 25)]
subj = [n for n in names if n != "CHB21"]
S = np.array([[1.0 if (c == s or (s == "CHB01" and c == "CHB21")) else 0.0 for c in names] for s in subj])
def curve(a, w):
    tp, fp, ref, sec = (w @ D[a + "__tp"], w @ D[a + "__fp"], w @ D[a + "__ref"], w @ D[a + "__sec"])
    return fp / (sec / 86400.0), tp / ref
w1 = np.ones(cases)
for a in arms:
    fa, se = curve(a, w1)
    mono = np.all(np.diff(fa) >= 0)
    print("%-9s FA/day %6.1f..%6.1f monotone=%s ties=%d" % (a, fa.min(), fa.max(), mono, int(np.sum(np.diff(np.sort(fa)) == 0))))
rng = np.random.default_rng(7)
idx = rng.integers(0, len(subj), size=(2000, len(subj)))
W = np.zeros((2000, len(subj)))
for r in range(2000):
    np.add.at(W[r], idx[r], 1)
W = W @ S
groups = {"A0": ["A0"], "Tamp": [a for a in arms if a.startswith("Tamp")], "Tfree": [a for a in arms if a.startswith("Tfree")]}
for g, al in groups.items():
    if not al: continue
    for L in (75, 100, 125):
        bad = 0; nonmono = 0
        for r in range(2000):
            out = False
            for a in al:
                fa, se = curve(a, W[r])
                if L < fa.min() or L > fa.max(): out = True
                if not np.all(np.diff(fa) >= 0): nonmono += 1
            bad += out
        print("%-6s L=%3d unreachable in %4d/2000 resamples; non-monotone curves %d" % (g, L, bad, nonmono))
