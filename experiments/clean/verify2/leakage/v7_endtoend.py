"""leakage audit: preds_n1/thresholds_n1 vs the committed uncalibrated_lopo.py output
(NEW endtoend_uncalibrated_fixed15.npz, GPU softmax path). Read-only."""
import json
from pathlib import Path
import numpy as np
NEW = Path(r"C:\Users\user\AppData\Local\Temp\claude\36c1f29744f93fe6\c7848ca6-bfef-414b-a321-d6686c1a43f8\scratchpad\clean")
ref = np.load(NEW / "endtoend_uncalibrated_fixed15.npz", allow_pickle=True)["data"].item()
P = np.load(NEW / "trees_out" / "preds_n1.npy", allow_pickle=True).item()["N1"]
T = json.loads((NEW / "trees_out" / "thresholds_n1.json").read_text())["N1"]
dp = dt = 0.0
nb = 0
models = set()
for c in sorted(ref):
    assert len(ref[c]["blocks"]) == len(P[c]), c
    for (p1, y1), (p2, y2) in zip(P[c], ref[c]["blocks"]):
        assert np.array_equal(np.asarray(y1, np.int64), np.asarray(y2, np.int64)), c
        dp = max(dp, float(np.abs(np.asarray(p1) - np.asarray(p2, np.float64)).max()))
        nb += 1
    dt = max(dt, abs(T[c]["2.0"] - float(ref[c]["t"])))
    models.add(ref[c]["model"])
print("cases %d, blocks %d, max|p_trees_pipeline - p_uncalibrated_lopo| %.3g, max|t@2.0 diff| %.3g, models %s"
      % (len(ref), nb, dp, dt, models))
