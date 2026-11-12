"""Robustness critic, step 3: spread of per-case transfer, comparable-range check without
CHB06+CHB15, bootstrap without them. Reads verify/robust/tables.npz only."""
import os

import numpy as np

OUT = os.path.join(r"C:\Users\user\AppData\Local\Temp\claude\36c1f29744f93fe6",
                   r"c7848ca6-bfef-414b-a321-d6686c1a43f8\scratchpad\clean\verify\robust")
Z = np.load(os.path.join(OUT, "tables.npz"))
cases = list(Z["cases"])
TG = list(Z["targets"])
AMP = ["Tamp_s0", "Tamp_s1", "Tamp_s2"]
REL = ["Trel_s0", "Trel_s1", "Trel_s2"]
ref, day = Z["A0__ref"], Z["A0__sec"] / 86400.0
LINES = []


def say(s=""):
    print(s)
    LINES.append(s)


say("Spread of per-case achieved/target (log10, cases with >0 FA counted with +0.5 fp floor)")
for j, t in enumerate(TG):
    if t not in (0.05, 0.2, 0.5, 1.0, 2.0, 5.0):
        continue
    out = []
    for arms in (AMP, REL):
        fp = np.mean([Z[a + "__fp"][:, j] for a in arms], 0)
        r = np.log10((fp + 0.5) / day / (24 * t))
        out.append((np.std(r), np.percentile(r, 90) - np.percentile(r, 10)))
    say("  target %.2f: SD Tamp %.2f Trel %.2f | 10-90%% range Tamp %.2f Trel %.2f (decades)"
        % (t, out[0][0], out[1][0], out[0][1], out[1][1]))


def interp_at(fa, sens, L):
    o = np.lexsort((-sens, fa))
    fa, sens = fa[o], sens[o]
    f = np.concatenate([[True], np.diff(fa) > 0])
    return np.interp(L, fa[f], sens[f], left=np.nan, right=np.nan)


def curve(a, w):
    fa = (w @ Z[a + "__fp"]) / (w @ day)
    se = (w @ Z[a + "__tp"]) / (w @ ref)
    return fa, se


def subj_of(c):
    return "CHB01" if c == "CHB21" else c


for label, excl in (("all cases", []), ("without CHB06+CHB15", ["CHB06", "CHB15"]),
                    ("without CHB06", ["CHB06"]), ("without CHB15", ["CHB15"])):
    w = np.array([0.0 if c in excl else 1.0 for c in cases])
    L = [10, 25, 30, 40, 50, 60, 75, 100]
    va = np.array([interp_at(*curve(a, w), L) for a in AMP])
    vr = np.array([interp_at(*curve(a, w), L) for a in REL])
    say("Deployable curves interpolated, %s (per seed; mean where every seed reaches):" % label)
    for i, lv in enumerate(L):
        say("  %3d/day: Tamp %s  Trel %s  diff(mean) %s" % (
            lv, " ".join("%.3f" % x for x in va[:, i]), " ".join("%.3f" % x for x in vr[:, i]),
            ("%+.3f" % (vr[:, i].mean() - va[:, i].mean())) if not np.isnan(va[:, i]).any()
            and not np.isnan(vr[:, i]).any() else "n/a"))
    say("  min FA/day: Tamp %s, Trel %s" % (" ".join("%.1f" % curve(a, w)[0].min() for a in AMP),
                                           " ".join("%.1f" % curve(a, w)[0].min() for a in REL)))

# bootstrap without CHB06+CHB15 at README point
j2 = TG.index(2.0)
sub = [c for c in cases if c not in ("CHB06", "CHB15")]
subjects = []
for c in sub:
    if subj_of(c) not in subjects:
        subjects.append(subj_of(c))
S = np.array([[1.0 if (c in sub and subj_of(c) == s) else 0.0 for c in cases] for s in subjects])
rng = np.random.default_rng(5)
fr, ds = [], []
for _ in range(4000):
    w = np.bincount(rng.integers(0, len(subjects), len(subjects)), minlength=len(subjects)) @ S
    sa = [AMP[i] for i in rng.integers(0, 3, 3)]
    sr = [REL[i] for i in rng.integers(0, 3, 3)]
    fa = lambda arms: np.mean([(w @ Z[a + "__fp"][:, j2]) / (w @ day) for a in arms])  # noqa
    se = lambda arms: np.mean([(w @ Z[a + "__tp"][:, j2]) / (w @ ref) for a in arms])  # noqa
    fr.append(fa(sr) / fa(sa))
    ds.append(se(sr) - se(sa))
say("Bootstrap without CHB06+CHB15 (21 subjects), README point: FA ratio Trel/Tamp 95%% "
    "[%.3f, %.3f], P(>=1) %.4f; sensitivity diff 95%% [%+.3f, %+.3f], P(diff<0) %.3f"
    % (*np.percentile(fr, [2.5, 97.5]), np.mean(np.array(fr) >= 1),
       *np.percentile(ds, [2.5, 97.5]), np.mean(np.array(ds) < 0)))
with open(os.path.join(OUT, "extra.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(LINES) + "\n")
