"""C3: recompute Q3 (all, a, b) and Q1 without CHB06/CHB15 (exploratory). Read-only."""
import sys
from pathlib import Path
import numpy as np
NEW = Path(r"C:\Users\user\AppData\Local\Temp\claude\36c1f29744f93fe6"
           r"\c7848ca6-bfef-414b-a321-d6686c1a43f8\scratchpad\clean")
sys.path.insert(0, str(NEW)); sys.dont_write_bytecode = True
import clean_score as cs  # noqa: E402
import warnings; warnings.simplefilter("ignore", RuntimeWarning)
P, T, src = cs.load_arms(); cases = sorted(P["A0"]); names = cs.block_names(cases, P)
keep = {"168": {c: [True] * len(names[c]) for c in cases},
        "155": {c: [int((b["flat"].mean(0) > cs.ART_FRAC).sum()) < cs.ART_MIN_CH for b in names[c]] for c in cases}}
arms = ["A0", "N1", "Tamp_s0", "Tamp_s1", "Tamp_s2"]
out = []
for k in ("155", "168"):
    D = {a: cs.deploy_table(P[a], T[a], cases, keep[k]) for a in arms}
    for lab, sub in (("all", cases), ("(a)", cs.EARLY), ("(b)", cs.SAME15)):
        r = cs.Bench(D, cases, sub, k).compare("N1", "A0")
        out.append("[%s %s] N1-A0 %+.2f over %s | 95%% [%+.2f, %+.2f] clamped [%+.2f, %+.2f] partial %d dropped %d -> %s"
                   % (k, lab, 100 * r["point"], r["levels"], 100 * r["lo"], 100 * r["hi"], 100 * r["clo"], 100 * r["chi"],
                      r["partial"], r["dropped"], r["decision"]))
    sub = [c for c in cases if c not in cs.MOTIVATED]
    r = cs.Bench(D, cases, sub, k).compare("Tamp", "N1")
    out.append("[%s w/o CHB06,CHB15] Tamp-N1 %+.2f over %s | 95%% [%+.2f, %+.2f] -> %s | per level %s"
               % (k, 100 * r["point"], r["levels"], 100 * r["lo"], 100 * r["hi"], r["decision"],
                  {L: round(100 * v, 2) for L, v in r["per_level"].items()}))
print("\n".join(out)); (NEW / "verify2" / "claims" / "c3_q3.txt").write_text("\n".join(out) + "\n")
