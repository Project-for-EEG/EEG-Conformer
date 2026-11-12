"""critic step E: degenerate detections. Per arm, target and test file: flagged fraction after
post-processing, tp, fp. Lists files where an arm alarms most of the time (constant-alarm 'catches'),
the recordings with missing channels, and re-runs the matched-FA contrast without the montage-broken
CHB12 recordings (chb12_27 all channels missing; chb12_28, chb12_29 eight missing).
Run with cwd C:\\EEG."""
import sys
sys.dont_write_bytecode = True
import json
from pathlib import Path
import numpy as np
sys.path.insert(0, r"C:\EEG")
from finetune_per_patient import patient_files, find_split  # noqa: E402
from score_official import count, detect                    # noqa: E402

TREES = Path(r"C:\Users\user\AppData\Local\Temp\claude\36c1f29744f93fe6"
             r"\b3889703-629c-4c5a-b583-69f722743ff4\scratchpad\trees")
OUT, ME = TREES / "out", TREES / "verify" / "critic"
TARGETS = ("0.05", "0.1", "0.15", "0.2", "0.3", "0.4", "0.5", "0.75", "1.0", "1.5", "2.0",
           "3.0", "4.0", "5.0")
P = np.load(OUT / "preds.npy", allow_pickle=True).item()
T = {}
for t in "abc":
    T.update(json.loads((OUT / ("thresholds_%s.json" % t)).read_text()))
cases = sorted(P["A0"])
names = {}
for c in cases:
    fs = patient_files(c)
    fi, wi = find_split(fs, 1)
    nm = []
    for i, f in enumerate(fs):
        if i < fi:
            continue
        with np.load(f) as d:
            n = len(d["labels"])
        if (n - wi if i == fi else n) >= 10:
            nm.append(f.stem.replace("_labeled", ""))
    assert len(nm) == len(P["A0"][c]), c
    names[c] = nm
BROKEN = {"chb12_27", "chb12_28", "chb12_29"}
L = []
# per-file table at three targets
rows = {}
for a in P:
    for c in cases:
        for (p, y), nm in zip(P[a][c], names[c]):
            p = np.asarray(p, np.float64)
            for tg in TARGETS:
                t = float(T[a][c][tg])
                tp, fp, ref, sec = count([(p, y)], t, 5, 3)
                fl = detect(p, t, 5, 3).mean()
                rows[(a, c, nm, tg)] = (tp, fp, ref, sec, fl)
G = {"A0": ["A0"], "Tamp": ["Tamp_s%d" % s for s in range(3)],
     "Tfree": ["Tfree_s%d" % s for s in range(3)]}
L.append("FILES WHERE AN ARM IS ALARMED >= 40% OF THE TIME (target 2.0 and 0.05): arm case file "
         "flagged tp/ref fp")
for tg in ("2.0", "0.05"):
    for a in ("A0", "Tamp_s0", "Tfree_s0"):
        hits = [(k, v) for k, v in rows.items() if k[0] == a and k[3] == tg and v[4] >= 0.4]
        tps = sum(v[0] for _, v in hits)
        fps = sum(v[1] for _, v in hits)
        L.append("  target %s %-8s: %d files, %d tp, %d fp: %s" % (
            tg, a, len(hits), tps, fps,
            " ".join("%s(%.0f%% %d/%d fp%d)" % (k[2], 100 * v[4], v[0], v[2], v[1]) for k, v in hits)))
L.append("")


def curve(arm, exclude):
    fa, se = [], []
    for tg in TARGETS:
        tp = fp = ref = sec = 0
        for c in cases:
            for nm in names[c]:
                if nm in exclude:
                    continue
                v = rows[(arm, c, nm, tg)]
                tp += v[0]; fp += v[1]; ref += v[2]; sec += v[3]
        fa.append(fp / (sec / 86400)); se.append(tp / ref)
    return np.array(fa), np.array(se), ref


def gi(g, F, exclude):
    out = []
    for a in G[g]:
        fa, se, _ = curve(a, exclude)
        o = np.argsort(fa)
        v = np.interp(np.log(F), np.log(fa[o]), se[o])
        v = np.where((np.array(F) < fa.min()) | (np.array(F) > fa.max()), np.nan, v)
        out.append(v)
    return np.mean(out, 0)


F = np.array([50, 60, 75, 90, 100, 115, 125, 150])
for ex, lab in ((set(), "all test files"), (BROKEN, "without chb12_27/28/29")):
    L.append("MATCHED-FA CURVES, %s (seizures %d)" % (lab, curve("A0", ex)[2]))
    L.append("  FA/day   " + " ".join("%7d" % f for f in F))
    R = {g: gi(g, F, ex) for g in G}
    for g in G:
        L.append("  %-8s " % g + " ".join("%7.3f" % v for v in R[g]))
    L.append("  Tamp-A0  " + " ".join("%+7.3f" % v for v in R["Tamp"] - R["A0"]))
    L.append("  Tfree-A0 " + " ".join("%+7.3f" % v for v in R["Tfree"] - R["A0"]))
    for a in ("A0", "Tamp_s0", "Tamp_s1", "Tamp_s2"):
        fa, se, ref = curve(a, ex)
        L.append("  %-8s target 2: %d/%d at %.1f/day; strictest: %.1f/day sens %.3f"
                 % (a, round(se[10] * ref), ref, fa[10], fa[0], se[0]))
    L.append("")
(ME / "a5_degenerate.txt").write_text("\n".join(L) + "\n", encoding="ascii")
np.save(ME / "a5_rows.npy", np.array({"rows": rows, "names": names}, dtype=object), allow_pickle=True)
print("\n".join(L))
