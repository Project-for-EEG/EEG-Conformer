"""Q3 'did early stopping matter' test of clean_rule.txt v2, applied independently.
Per-subject window AUC (smoothed k=5, kept blocks only) for A0 and N1; (a) 9 early-stopped
folds vs (b) 13 folds 15/15 in both runs. Read-only; uses verify2/rule/tables.npz for keep flags."""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\EEG")
from evaluate_temporal import smooth  # noqa: E402

OLD = Path(r"C:\Users\user\AppData\Local\Temp\claude\36c1f29744f93fe6"
           r"\b3889703-629c-4c5a-b583-69f722743ff4\scratchpad\trees")
NEW = Path(r"C:\Users\user\AppData\Local\Temp\claude\36c1f29744f93fe6"
           r"\c7848ca6-bfef-414b-a321-d6686c1a43f8\scratchpad\clean")
OUT = NEW / "verify2" / "rule"
EARLY = ["CHB02", "CHB04", "CHB05", "CHB09", "CHB10", "CHB12", "CHB15", "CHB16", "CHB24"]
SAME15 = ["CHB03", "CHB06", "CHB07", "CHB08", "CHB11", "CHB13", "CHB14", "CHB17", "CHB18",
          "CHB19", "CHB20", "CHB22", "CHB23"]

z = np.load(OUT / "tables.npz")
case_of, nflat = list(z["case_of"]), z["nflat"]
A0 = np.load(OLD / "out" / "preds_a.npy", allow_pickle=True).item()["A0"]
N1 = np.load(NEW / "trees_out" / "preds_n1.npy", allow_pickle=True).item()["N1"]
cases = sorted(A0)
keep = {"168": {c: [] for c in cases}, "155": {c: [] for c in cases}}
for c, n in zip(case_of, nflat):
    keep["168"][c].append(True)
    keep["155"][c].append(bool(n < 8))

res = {}
lines = []
for k in ("155", "168"):
    auc = {}
    for name, P in (("A0", A0), ("N1", N1)):
        auc[name] = {}
        for c in cases:
            bl = [b for b, kk in zip(P[c], keep[k][c]) if kk]
            y = np.concatenate([np.asarray(b[1]) for b in bl])
            s = np.concatenate([smooth(np.asarray(b[0], np.float64), 5) for b in bl])
            auc[name][c] = roc_auc_score(y, s)
    d = {c: auc["N1"][c] - auc["A0"][c] for c in cases}
    da = np.array([d[c] for c in EARLY])
    db = np.array([d[c] for c in SAME15])
    out_a = [c for c in EARLY if d[c] < db.min() or d[c] > db.max()]
    mw = mannwhitneyu(da, db, alternative="two-sided")
    lines.append("[%s] per-fold window AUC N1 - A0" % k)
    lines.append("   (a) " + " ".join("%s %+.4f" % (c, d[c]) for c in EARLY))
    lines.append("   (b) " + " ".join("%s %+.4f" % (c, d[c]) for c in SAME15))
    lines.append("   (a) mean %+.4f median %+.4f | (b) mean %+.4f median %+.4f sd %.4f range [%+.4f, %+.4f]"
                 " IQR [%+.4f, %+.4f]"
                 % (da.mean(), np.median(da), db.mean(), np.median(db), db.std(ddof=1), db.min(), db.max(),
                    *np.percentile(db, [25, 75])))
    lines.append("   (a) folds outside (b)'s range: %s" % out_a)
    lines.append("   (a) mean/median inside (b)'s range: %s / %s; Mann-Whitney (a) vs (b) p %.3f (exploratory)"
                 % (db.min() <= da.mean() <= db.max(), db.min() <= np.median(da) <= db.max(), mw.pvalue))
    res[k] = {"d": d, "outside": out_a, "mw_p": float(mw.pvalue)}
print("\n".join(lines))
(OUT / "q3_mattered.txt").write_text("\n".join(lines) + "\n")
json.dump(res, open(OUT / "q3_mattered.json", "w"), indent=1)
