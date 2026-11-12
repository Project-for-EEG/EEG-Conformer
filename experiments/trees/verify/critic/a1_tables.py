"""critic step A: per-case deployable tables (tp, fp, ref, sec per target) for every arm,
recomputed from out/preds.npy + out/thresholds_*.json with score_official.count.
Read-only on everything outside verify/critic. Run with cwd C:\\EEG."""
import sys
sys.dont_write_bytecode = True
import json
from pathlib import Path
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, r"C:\EEG")
from score_official import count  # noqa: E402

TREES = Path(r"C:\Users\user\AppData\Local\Temp\claude\36c1f29744f93fe6"
             r"\b3889703-629c-4c5a-b583-69f722743ff4\scratchpad\trees")
OUT = TREES / "out"
ME = TREES / "verify" / "critic"
TARGETS = ("0.05", "0.1", "0.15", "0.2", "0.3", "0.4", "0.5", "0.75", "1.0", "1.5", "2.0",
           "3.0", "4.0", "5.0")


def job(arm):
    P = np.load(OUT / "preds.npy", allow_pickle=True).item()[arm]
    T = {}
    for t in "abc":
        d = json.loads((OUT / ("thresholds_%s.json" % t)).read_text())
        if arm in d:
            T = d[arm]
    cases = sorted(P)
    tp = np.zeros((len(cases), len(TARGETS)), np.int64)
    fp = np.zeros_like(tp)
    ref = np.zeros(len(cases), np.int64)
    sec = np.zeros(len(cases))
    for i, c in enumerate(cases):
        blocks = [(np.asarray(p, np.float64), y) for p, y in P[c]]
        for j, tg in enumerate(TARGETS):
            a, b, r, s = count(blocks, float(T[c][tg]), 5, 3)
            tp[i, j], fp[i, j], ref[i], sec[i] = a, b, r, s
    return arm, tp, fp, ref, sec, np.array([[T[c][tg] for tg in TARGETS] for c in cases])


if __name__ == "__main__":
    arms = ["A0", "Tamp_s0", "Tamp_s1", "Tamp_s2", "Tfree_s0", "Tfree_s1", "Tfree_s2"]
    res = {}
    with Pool(2) as pool:
        for arm, tp, fp, ref, sec, thr in pool.imap_unordered(job, arms):
            res[arm + "__tp"], res[arm + "__fp"] = tp, fp
            res[arm + "__ref"], res[arm + "__sec"], res[arm + "__thr"] = ref, sec, thr
            print(arm, tp.sum(0)[10], ref.sum(), round(fp.sum(0)[10] / (sec.sum() / 86400), 1),
                  flush=True)
    np.savez(ME / "deploy_tables.npz", targets=np.array(TARGETS), **res)
