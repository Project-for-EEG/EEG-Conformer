"""critic: per-block counts for every arm, rebuilt without clean_score.py and without
reading the data folders (block names come from OLD\\feats, which mirrors them).
Writes only to NEW\\verify2\\critic. CPU, 4 processes."""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\EEG")
from score_official import count           # noqa: E402
from evaluate_temporal import find_runs    # noqa: E402

OLD = Path(r"C:\Users\user\AppData\Local\Temp\claude\36c1f29744f93fe6"
           r"\b3889703-629c-4c5a-b583-69f722743ff4\scratchpad\trees")
NEW = Path(r"C:\Users\user\AppData\Local\Temp\claude\36c1f29744f93fe6"
           r"\c7848ca6-bfef-414b-a321-d6686c1a43f8\scratchpad\clean")
OUT = NEW / "verify2" / "critic"
TARGETS = ("0.05", "0.1", "0.15", "0.2", "0.3", "0.4", "0.5", "0.75", "1.0", "1.5", "2.0",
           "3.0", "4.0", "5.0")


def load():
    P, T = {}, {}
    srcs = [(OLD / "out", t) for t in "abcd"] + [(NEW / "trees_out", t)
                                                for t in ("rel0", "rel1", "rel2", "relf0", "n1")]
    for d, tag in srcs:
        part = np.load(d / ("preds_%s.npy" % tag), allow_pickle=True).item()
        tp = json.loads((d / ("thresholds_%s.json" % tag)).read_text())
        for a in part:
            assert a not in P, a
            P[a], T[a] = part[a], tp[a]
    return P, T


def blocks_from_feats(c):
    files = sorted(OLD.joinpath("feats").glob("%s__*.npz" % c))
    labs, flats = [], []
    for f in files:
        with np.load(f) as d:
            labs.append(d["labels"])
            flats.append(np.isnan(d["feats"][:, :, 0]))
    # first seizure end
    split = None
    for fi, lab in enumerate(labs):
        runs = list(find_runs(lab.astype(bool)))
        if runs:
            split = (fi, runs[0][1])
            break
    fi, wi = split
    out = []
    for i in range(fi, len(files)):
        lo = wi if i == fi else 0
        if len(labs[i]) - lo >= 10:
            fr = flats[i][lo:].mean(0)
            out.append({"stem": files[i].stem.split("__", 1)[1], "labels": labs[i][lo:],
                        "nflat": int((fr > 0.5).sum())})
    return out


def job(arm):
    P, T = load()
    res = {}
    for c in sorted(P[arm]):
        rows = []
        for p, y in P[arm][c]:
            r = []
            for tg in TARGETS:
                r.append(count([(np.asarray(p, np.float64), np.asarray(y))],
                               float(T[arm][c][tg]), 5, 3))
            rows.append(r)
        res[c] = np.array(rows, dtype=np.float64)   # blocks x targets x 4
    return arm, res


if __name__ == "__main__":
    P, T = load()
    arms = list(P)
    cases = sorted(P["A0"])
    meta = {}
    for c in cases:
        bl = blocks_from_feats(c)
        assert len(bl) == len(P["A0"][c]), (c, len(bl), len(P["A0"][c]))
        for b, (_, y) in zip(bl, P["A0"][c]):
            assert np.array_equal(b["labels"].astype(np.int64), np.asarray(y, np.int64)), c
        for a in arms:
            assert len(P[a][c]) == len(bl)
            for b, (p, y) in zip(bl, P[a][c]):
                assert len(p) == len(b["labels"])
                assert np.array_equal(np.asarray(y, np.int64), b["labels"].astype(np.int64))
        meta[c] = [(b["stem"], b["nflat"], len(b["labels"])) for b in bl]
    print("blocks", sum(len(v) for v in meta.values()))
    hits = [(c, s, n) for c in cases for s, n, _ in meta[c] if n >= 1]
    print("files with any flat channel:", hits)
    with Pool(4) as pool:
        out = dict(pool.map(job, arms))
    np.save(OUT / "c1_counts.npy", {"counts": out, "meta": meta, "arms": arms, "cases": cases},
            allow_pickle=True)
    print("saved")
