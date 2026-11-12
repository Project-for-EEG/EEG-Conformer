"""critic: per-subject window AUC recomputed from the prediction files (not from
clean_results.json), and compared with the json. Exploratory sign counts."""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\EEG")
from evaluate_temporal import smooth   # noqa: E402

NEW = Path(r"C:\Users\user\AppData\Local\Temp\claude\36c1f29744f93fe6"
           r"\c7848ca6-bfef-414b-a321-d6686c1a43f8\scratchpad\clean")
OUT = NEW / "verify2" / "critic"
sys.path.insert(0, str(OUT))
from c1_tables import load   # noqa: E402

d = np.load(OUT / "c1_counts.npy", allow_pickle=True).item()
META, CASES = d["meta"], d["cases"]
EXCL = {"chb12_27_labeled", "chb12_28_labeled", "chb12_29_labeled"}
GROUPS = {"A0": ["A0"], "N1": ["N1"], "Tamp": ["Tamp_s0", "Tamp_s1", "Tamp_s2"],
          "Tfree": ["Tfree_s0", "Tfree_s1", "Tfree_s2"], "Tall": ["Tall"],
          "Trel": ["Trel_s0", "Trel_s1", "Trel_s2"], "Trelf": ["Trelf_s0"]}
P, _ = load()
js = json.loads((NEW / "score_final" / "clean_results.json").read_text())["auc"]
subj = []
for c in CASES:
    s = "CHB01" if c == "CHB21" else c
    if s not in subj:
        subj.append(s)
lines = []
for k in ("155", "168"):
    auc = {}
    for g, arms in GROUPS.items():
        pc = []
        for c in CASES:
            keep = [not (k == "155" and s in EXCL) for s, _, _ in META[c]]
            vals = []
            for a in arms:
                bl = [b for b, kk in zip(P[a][c], keep) if kk]
                y = np.concatenate([np.asarray(yy) for _, yy in bl])
                p = np.concatenate([smooth(np.asarray(pp, np.float64), 5) for pp, _ in bl])
                vals.append(roc_auc_score(y, p))
            pc.append(np.mean(vals))
        auc[g] = np.array([np.mean([pc[i] for i, c in enumerate(CASES)
                                    if c == s or (s == "CHB01" and c == "CHB21")]) for s in subj])
        dif = np.max(np.abs(auc[g] - np.array(js[k][g])))
        lines.append("[%s] %-6s median %.4f | max|mine - json| %.2e" % (k, g, np.median(auc[g]), dif))
    for x, y in (("Tamp", "N1"), ("Trel", "N1"), ("Tfree", "N1"), ("Tall", "N1"), ("Trelf", "N1"),
                 ("Trel", "Tamp"), ("N1", "A0"), ("Tfree", "A0")):
        dd = auc[x] - auc[y]
        lines.append("[%s] %s - %s: median %+.4f, x better in %d, worse in %d of %d, Wilcoxon zsplit p %.4f"
                     % (k, x, y, np.median(dd), (dd > 0).sum(), (dd < 0).sum(), len(dd),
                        wilcoxon(dd, zero_method="zsplit").pvalue))
print("\n".join(lines))
(OUT / "c3_auc.txt").write_text("\n".join(lines) + "\n")
