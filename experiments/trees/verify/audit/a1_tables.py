"""Audit step 1: rebuild the deployable tables from saved preds/thresholds (read-only)."""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np

REPO = Path(r"C:\EEG")
sys.path.insert(0, str(REPO))
TR = Path(r"C:\Users\user\AppData\Local\Temp\claude\36c1f29744f93fe6"
          r"\b3889703-629c-4c5a-b583-69f722743ff4\scratchpad\trees")
OUT = TR / "verify" / "audit"
TARGETS = ("0.05", "0.1", "0.15", "0.2", "0.3", "0.4", "0.5", "0.75", "1.0", "1.5", "2.0",
           "3.0", "4.0", "5.0")


def job(arm):
    from score_official import count
    P = np.load(TR / "out" / "preds.npy", allow_pickle=True).item()[arm]
    T = {}
    for t in ("a", "b", "c"):
        d = json.loads((TR / "out" / ("thresholds_%s.json" % t)).read_text())
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
            tp[i, j], fp[i, j] = a, b
            ref[i], sec[i] = r, s
    return arm, dict(tp=tp, fp=fp, ref=ref, sec=sec, cases=np.array(cases))


if __name__ == "__main__":
    from threadpoolctl import threadpool_limits
    P = np.load(TR / "out" / "preds.npy", allow_pickle=True).item()
    arms = list(P)
    with threadpool_limits(1), Pool(3) as pool:
        for arm, D in pool.imap_unordered(job, arms):
            np.savez(OUT / ("D_%s.npz" % arm), **D)
            print(arm, "events", D["ref"].sum(), "hours %.2f" % (D["sec"].sum() / 3600),
                  flush=True)
