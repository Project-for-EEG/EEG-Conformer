"""rescore / extra checks.
(a) continuity: my 168-event A0 / Tamp / Tfree curves vs OLD\\out\\score_results_main.txt (42 cells)
(b) README procedure: endtoend_uncalibrated_fixed15.npz scored with score_official.count by my own
    loop, and aligned with preds_n1 / thresholds_n1 (2.0)
(c) CHB01 and CHB21 share thresholds (one fold) in every arm; N1 is not a copy of A0
(d) leave-one-subject-out influence on Q1 and Q4 (level set fixed from full data)
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, r"C:\EEG")
from score_official import count  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
NEW = Path(r"C:\Users\user\AppData\Local\Temp\claude\36c1f29744f93fe6\c7848ca6-bfef-414b-a321-d6686c1a43f8\scratchpad\clean")
OLD = Path(r"C:\Users\user\AppData\Local\Temp\claude\36c1f29744f93fe6\b3889703-629c-4c5a-b583-69f722743ff4\scratchpad\trees")
OUT = []


def say(s=""):
    print(s)
    OUT.append(s)


c = np.load(HERE / "counts.npz")
arms = list(c["arms"])
A = {a: i for i, a in enumerate(arms)}
targets = list(c["targets"])
TP, FP, REF, SEC = c["tp"], c["fp"], c["ref"], c["sec"]
case = c["case"]

# (a) continuity
txt = (OLD / "out" / "score_results_main.txt").read_text().splitlines()
rows = {}
for ln in txt:
    m = re.match(r"^(\d+\.\d+)\s+([\d.]+) \(\s*([\d.]+)\)\s+([\d.]+) \(\s*([\d.]+)\)\s+([\d.]+) \(\s*([\d.]+)\)\s*$", ln)
    if m:
        rows[float(m.group(1))] = [float(x) for x in m.groups()[1:]]
assert len(rows) == 14, len(rows)
bad = 0
for ti, tg in enumerate(targets):
    mine = []
    for fam in (["A0"], ["Tamp_s0", "Tamp_s1", "Tamp_s2"], ["Tfree_s0", "Tfree_s1", "Tfree_s2"]):
        s = np.mean([TP[A[m], ti].sum() / REF[A[m], ti].sum() for m in fam])
        f = np.mean([FP[A[m], ti].sum() / (SEC[A[m], ti].sum() / 86400) for m in fam])
        mine += [float("%.3f" % s), float("%.1f" % f)]
    if mine != rows[tg]:
        bad += 1
        say("  continuity MISMATCH at %g: mine %s ref %s" % (tg, mine, rows[tg]))
say("(a) continuity, 14 targets x 3 arms x (sens, FA) = 84 numbers (42 cells): %d mismatching rows"
    % bad)

# (b) README procedure npz
d = np.load(NEW / "endtoend_uncalibrated_fixed15.npz", allow_pickle=True)["data"].item()
pn1 = np.load(NEW / "trees_out" / "preds_n1.npy", allow_pickle=True).item()["N1"]
pa0 = np.load(OLD / "out" / "preds_a.npy", allow_pickle=True).item()["A0"]
tn1 = json.load(open(NEW / "trees_out" / "thresholds_n1.json"))["N1"]
dp = dt = 0.0
tp = fp = ref = 0
sec = 0.0
for cs in sorted(d):
    blocks = d[cs]["blocks"]
    assert len(blocks) == len(pn1[cs]) == len(pa0[cs]), cs
    for (p1, l1), (p2, l2), (p3, l3) in zip(blocks, pn1[cs], pa0[cs]):
        assert len(p1) == len(p2) == len(p3)
        assert np.array_equal(np.asarray(l1).astype(int), np.asarray(l2).astype(int))
        assert np.array_equal(np.asarray(l1).astype(int), np.asarray(l3).astype(int))
        dp = max(dp, float(np.max(np.abs(np.asarray(p1, float) - np.asarray(p2, float)))))
    dt = max(dt, abs(float(d[cs]["t"]) - tn1[cs]["2.0"]))
    r = count(blocks, float(d[cs]["t"]), 5, 3)
    tp += r[0]; fp += r[1]; ref += r[2]; sec += r[3]
say("(b) npz vs preds_n1: max |dp| %.3g, max |dt@2.0| %.3g; labels/lengths equal to N1 and A0 blocks"
    % (dp, dt))
say("    README procedure (npz, own loop over score_official.count): %d/%d at %.4f/day (FA %d)"
    % (tp, ref, fp / (sec / 86400), fp))

# (c)
for f, keys in ((OLD / "out", "abcd"), (NEW / "trees_out", ["n1", "rel0", "rel1", "rel2", "relf0"])):
    for k in keys:
        t = json.load(open(f / ("thresholds_%s.json" % k)))
        for a in t:
            same = all(t[a]["CHB01"][x] == t[a]["CHB21"][x] for x in t[a]["CHB01"])
            if not same:
                say("  (c) %s: CHB01 and CHB21 thresholds DIFFER" % a)
say("(c) CHB01/CHB21 thresholds identical in every arm (only differences printed above)")
mx = max(float(np.max(np.abs(np.asarray(p2) - np.asarray(p3)))) for cs in pn1 for (p2, _), (p3, _) in zip(pn1[cs], pa0[cs]))
say("    max |N1 - A0| test probability: %.4f (N1 is not a copy of A0)" % mx)

# (d) leave one subject out
GRID = np.array([50, 60, 75, 90, 100, 115, 125, 150], dtype=float)
bmap = json.loads((HERE / "block_map.json").read_text())
stems = np.array([bmap[cs][int(b)]["file"] for cs, b in zip(case, c["block"])])
keep155 = ~np.isin(stems, ["chb12_27_labeled", "chb12_28_labeled", "chb12_29_labeled"])
subj = np.array(["CHB01" if x == "CHB21" else x for x in case])
FAM = {"Tamp": ["Tamp_s0", "Tamp_s1", "Tamp_s2"], "Trel": ["Trel_s0", "Trel_s1", "Trel_s2"], "N1": ["N1"], "A0": ["A0"]}
LV = {"Q1": [60, 75, 90, 100, 115, 125], "Q3": [50, 60, 75, 90, 100, 115, 125], "Q4": [50, 60, 75, 90, 100]}


def stat(mask, X, Y, lv):
    def fam(F):
        vals = []
        for m in FAM[F]:
            t = TP[A[m]][:, mask].sum(1); r = REF[A[m]][:, mask].sum(1)
            f = FP[A[m]][:, mask].sum(1); s = SEC[A[m]][:, mask].sum(1)
            sens, fa = t / r, f / (s / 86400)
            o = np.argsort(fa, kind="stable")
            v = np.interp(lv, fa[o], sens[o])
            v[(np.array(lv) < fa[o][0]) | (np.array(lv) > fa[o][-1])] = np.nan
            vals.append(v)
        return np.mean(vals, 0)
    d = 100 * (fam(X) - fam(Y))
    return np.nanmean(d), int(np.isnan(d).sum())


for q, X, Y in (("Q1", "Tamp", "N1"), ("Q3", "N1", "A0"), ("Q4", "Trel", "N1")):
    for sname, base in (("155", keep155), ("168", np.ones_like(keep155))):
        full, _ = stat(base, X, Y, LV[q])
        loo = []
        for s in sorted(set(subj)):
            v, nn = stat(base & (subj != s), X, Y, LV[q])
            loo.append((v, s, nn))
        loo.sort()
        say("(d) %s %s-%s set %s: full %+.4f; leave-one-subject-out min %+.2f (without %s), max %+.2f (without %s);"
            " NaN levels in any LOSO: %d" % (q, X, Y, sname, full, loo[0][0], loo[0][1], loo[-1][0], loo[-1][1],
                                              sum(x[2] for x in loo)))
(HERE / "extra.txt").write_text("\n".join(OUT) + "\n")
