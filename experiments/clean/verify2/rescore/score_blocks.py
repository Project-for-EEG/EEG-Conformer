"""rescore / step 1: per-block official counts for every arm and target.

Written without reading clean_score.py. Run with cwd C:\\EEG.
For each arm, case, target and test block: score_official.count([block], thr,
smooth=5, min_consec=3) -> (tp, fp, refTrue, seconds). Summing over blocks is
exactly what count() does internally (it loops blocks and adds), so per-block
storage loses nothing and lets blocks be dropped / subjects reweighted later.
Output: counts.npz (arrays [arm, target, block]) + blocks index.
"""
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, r"C:\EEG")

HERE = Path(__file__).resolve().parent
NEW = Path(r"C:\Users\user\AppData\Local\Temp\claude\36c1f29744f93fe6\c7848ca6-bfef-414b-a321-d6686c1a43f8\scratchpad\clean")
OLD = Path(r"C:\Users\user\AppData\Local\Temp\claude\36c1f29744f93fe6\b3889703-629c-4c5a-b583-69f722743ff4\scratchpad\trees")

SOURCES = [("a", OLD / "out"), ("b", OLD / "out"), ("c", OLD / "out"), ("d", OLD / "out"),
           ("n1", NEW / "trees_out"), ("rel0", NEW / "trees_out"), ("rel1", NEW / "trees_out"),
           ("rel2", NEW / "trees_out"), ("relf0", NEW / "trees_out")]
TARGETS = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
K, MINC = 5, 3


def load_all():
    preds, thrs = {}, {}
    for tag, d in SOURCES:
        p = np.load(d / ("preds_%s.npy" % tag), allow_pickle=True).item()
        t = json.load(open(d / ("thresholds_%s.json" % tag)))
        assert sorted(p) == sorted(t), tag
        for a in p:
            assert a not in preds, ("arm twice", a)
            preds[a] = p[a]
            thrs[a] = t[a]
    return preds, thrs


def work(arm):
    from score_official import count
    preds, thrs = load_all()
    cases = sorted(preds[arm])
    out = []
    for case in cases:
        tmap = {float(k): v for k, v in thrs[arm][case].items()}
        assert sorted(tmap) == sorted(TARGETS), (arm, case, sorted(tmap))
        for bi, blk in enumerate(preds[arm][case]):
            row = []
            for tg in TARGETS:
                row.append(count([blk], float(tmap[tg]), K, MINC))
            out.append((case, bi, row))
    return arm, out


def main():
    preds, thrs = load_all()
    arms = sorted(preds)
    print("arms:", arms)
    t0 = time.time()
    res = {}
    with Pool(4) as pool:
        for arm, out in pool.imap_unordered(work, arms):
            res[arm] = out
            print("done", arm, "%.0fs" % (time.time() - t0), flush=True)
    index = [(c, b) for c, b, _ in res[arms[0]]]
    for a in arms:
        assert [(c, b) for c, b, _ in res[a]] == index, a
    nb = len(index)
    arr = {f: np.zeros((len(arms), len(TARGETS), nb)) for f in ("tp", "fp", "ref", "sec")}
    for ai, a in enumerate(arms):
        for j, (c, b, row) in enumerate(res[a]):
            for ti, (tp, fp, ref, sec) in enumerate(row):
                arr["tp"][ai, ti, j] = tp
                arr["fp"][ai, ti, j] = fp
                arr["ref"][ai, ti, j] = ref
                arr["sec"][ai, ti, j] = sec
    np.savez(HERE / "counts.npz", arms=np.array(arms), targets=np.array(TARGETS),
             case=np.array([c for c, _ in index]), block=np.array([b for _, b in index]), **arr)
    print("saved counts.npz", nb, "blocks")


if __name__ == "__main__":
    main()
