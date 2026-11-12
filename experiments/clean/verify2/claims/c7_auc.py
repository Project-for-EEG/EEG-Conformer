"""C7/C3: per-subject window AUC from clean_results.json. Read-only."""
import json
import numpy as np
from scipy.stats import wilcoxon, mannwhitneyu
NEW = r"C:\Users\user\AppData\Local\Temp\claude\36c1f29744f93fe6\c7848ca6-bfef-414b-a321-d6686c1a43f8\scratchpad\clean"
d = json.load(open(NEW + r"\score_final\clean_results.json"))
cases = ["CHB%02d" % i for i in range(1, 25)]
subj = [c for c in cases if c != "CHB21"]
EARLY = ["CHB02", "CHB04", "CHB05", "CHB09", "CHB10", "CHB12", "CHB15", "CHB16", "CHB24"]
SAME15 = ["CHB03", "CHB06", "CHB07", "CHB08", "CHB11", "CHB13", "CHB14", "CHB17", "CHB18",
          "CHB19", "CHB20", "CHB22", "CHB23"]
for k in ("155", "168"):
    A = {g: np.array(v) for g, v in d["auc"][k].items()}
    print("set", k, "n subjects", len(A["N1"]))
    for x, y in (("Tamp", "N1"), ("Trel", "N1"), ("Tfree", "N1"), ("Tall", "N1"), ("Trelf", "N1"),
                 ("Tamp", "A0"), ("Trel", "A0"), ("Tfree", "A0"), ("N1", "A0")):
        dd = A[x] - A[y]
        print("  %-5s - %-3s: >0 in %2d, <0 in %2d, =0 in %d of %d | median %+.4f | Wilcoxon zsplit p %.4f | worse in: %s"
              % (x, y, (dd > 0).sum(), (dd < 0).sum(), (dd == 0).sum(), len(dd), np.median(dd),
                 wilcoxon(dd, zero_method="zsplit").pvalue,
                 [(s, round(float(v), 3)) for s, v in zip(subj, dd) if v < 0]))
    dn = A["N1"] - A["A0"]
    a = np.array([dn[subj.index(s)] for s in EARLY]); b = np.array([dn[subj.index(s)] for s in SAME15])
    print("  Q3 AUC N1-A0 (a) early:", dict(zip(EARLY, a.round(3))))
    print("  Q3 AUC N1-A0 (b) same15:", dict(zip(SAME15, b.round(3))))
    print("  (a) median %+.4f mean %+.4f range [%+.3f, %+.3f] | (b) median %+.4f mean %+.4f range [%+.3f, %+.3f] sd %.4f"
          % (np.median(a), a.mean(), a.min(), a.max(), np.median(b), b.mean(), b.min(), b.max(), b.std(ddof=1)))
    print("  (a) folds outside (b)'s [min,max]:", [s for s, v in zip(EARLY, a) if v < b.min() or v > b.max()],
          "| outside (b) 2.5-97.5 pct", np.percentile(b, [2.5, 97.5]).round(3).tolist(),
          "| Mann-Whitney (a) vs (b) two-sided p %.3f" % mannwhitneyu(a, b).pvalue)
    print("  medians:", {g: round(float(np.median(v)), 4) for g, v in A.items()})
