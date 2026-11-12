"""critic step G (Q4): same patient, seen vs unseen. For every case q, FA/h on q's own pre-split and
post-split background (contiguous, +-60 windows around seizures excluded) at each fold model's
deployable thresholds: the fold model that held q out (unseen) vs the median of the 22 fold models
that trained on q (seen). Uses cached A0 logits and out/thresholds_a.json. Run with cwd C:\\EEG."""
import sys
sys.dont_write_bytecode = True
import json
from pathlib import Path
import numpy as np
sys.path.insert(0, r"C:\EEG")
from finetune_per_patient import patient_files, find_split   # noqa: E402
from calibrate_per_patient import fa_per_hour_on_background  # noqa: E402
from evaluate_temporal import find_runs                      # noqa: E402

TREES = Path(r"C:\Users\user\AppData\Local\Temp\claude\36c1f29744f93fe6"
             r"\b3889703-629c-4c5a-b583-69f722743ff4\scratchpad\trees")
ME = TREES / "verify" / "critic"
OLD = Path(r"C:\Users\user\AppData\Local\Temp\claude\36c1f29744f93fe6"
           r"\f241281f-add4-41ac-9210-f47dab0b83b0\scratchpad")
PILOT = ["CHB12", "CHB15", "CHB24", "CHB13", "CHB06", "CHB16"]
SHOW = ("0.05", "0.5", "2.0")
MARGIN = 60
folds = [p.stem[len("lopo_"):] for p in sorted(Path(r"C:\EEG\checkpoints\lopo").glob("lopo_*.pt"))]
cases = sorted(d.name for d in Path(r"C:\EEG\preprocessed_data").iterdir() if d.is_dir())
T = json.loads((TREES / "out" / "thresholds_a.json").read_text())["A0"]
thr = {f: T[f.split("_")[0]] for f in folds}


def segs(p, y):
    bad = (y == 1).copy()
    for lo, hi in find_runs(y == 1):
        bad[max(0, lo - MARGIN):hi + MARGIN] = True
    return [p[lo:hi] for lo, hi in find_runs(~bad) if hi - lo >= 10]


def fa(sg, t):
    r = h = 0.0
    for s in sg:
        x, _ = fa_per_hour_on_background(s, t, 5, 3)
        hh = len(s) * 2.0 / 3600
        r += x * hh
        h += hh
    return (r, h)


L = ["A0 SAME PATIENT, SEEN vs UNSEEN fold model: FA/h on the patient's own background at each "
     "fold's deployable threshold (targets %s)" % (SHOW,)]
tot = {k: np.zeros((len(SHOW), 2)) for k in ("pre_unseen", "pre_seen", "post_unseen", "post_seen")}
rows = []
for c in cases:
    fs = patient_files(c)
    fi, wi = find_split(fs, 1)
    per = {f: [] for f in folds}
    ys = []
    for f in fs:
        with np.load(OLD / "pilot_cache" / ("%s__%s.npz" % (c, f.stem))) as d:
            y = d["labels"].astype(np.int8)
            for k in PILOT:
                per[k].append(1 / (1 + np.exp(-d["logit_" + k].astype(np.float64))))
        with np.load(OLD / "pilot_cache2" / ("%s__%s.npz" % (c, f.stem))) as d:
            for k in folds:
                if k not in PILOT:
                    per[k].append(1 / (1 + np.exp(-d["logit_" + k].astype(np.float64))))
        ys.append(y)
    res = {}
    for k in folds:
        pre_s, post_s = [], []
        for i, (p, y) in enumerate(zip(per[k], ys)):
            if i < fi:
                pre_s += segs(p, y)
            elif i == fi:
                pre_s += segs(p[:wi], y[:wi])
                post_s += segs(p[wi:], y[wi:])
            else:
                post_s += segs(p, y)
        res[k] = [(fa(pre_s, thr[k][t]), fa(post_s, thr[k][t])) for t in SHOW]
    own = [k for k in folds if c in k.split("_")][0]
    seen = [k for k in folds if k != own]
    line = []
    for j, t in enumerate(SHOW):
        (ru, hu), (qu, gu) = res[own][j]
        tot["pre_unseen"][j] += (ru, hu)
        tot["post_unseen"][j] += (qu, gu)
        sp = np.median([res[k][j][0][0] / res[k][j][0][1] if res[k][j][0][1] else np.nan for k in seen])
        sq = np.median([res[k][j][1][0] / res[k][j][1][1] for k in seen])
        tot["pre_seen"][j] += (sp * hu, hu)
        tot["post_seen"][j] += (sq * gu, gu)
        line.append("%s: pre %.2f/%.2f post %.2f/%.2f" % (t, ru / hu if hu else np.nan, sp, qu / gu, sq))
    rows.append("  %s (unseen/seen-median) %s" % (c, " | ".join(line)))
L += rows
for k in tot:
    L.append("  pooled %-12s %s" % (k, " ".join("%.3f" % (tot[k][j, 0] / tot[k][j, 1]) for j in range(len(SHOW)))))
(ME / "a7_seen_vs_unseen_A0.txt").write_text("\n".join(L) + "\n", encoding="ascii")
print("\n".join(L))
