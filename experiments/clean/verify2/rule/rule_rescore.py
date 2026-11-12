"""Independent re-derivation of the clean_rule.txt v2 decisions (verify2/rule). Read-only.

Does NOT import clean_score.py and does NOT read the data folders: the block-to-file
mapping and the artifact rule are rebuilt from OLD\\feats (labels + NaN rms) only.
Uses score_official.count (the official timescoring counter the rule prescribes).
Step 1 (this file, --stage tables): per-block counts for every arm at the 14 targets,
saved to verify2/rule/tables.npz. Step 2 (--stage stats) does levels, bootstrap, decisions.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\EEG")

OLD = Path(r"C:\Users\user\AppData\Local\Temp\claude\36c1f29744f93fe6"
           r"\b3889703-629c-4c5a-b583-69f722743ff4\scratchpad\trees")
NEW = Path(r"C:\Users\user\AppData\Local\Temp\claude\36c1f29744f93fe6"
           r"\c7848ca6-bfef-414b-a321-d6686c1a43f8\scratchpad\clean")
OUT = NEW / "verify2" / "rule"


def load_arms():
    P, T, src = {}, {}, {}
    parts = [(OLD / "out", t) for t in "abcd"] + [(NEW / "trees_out", t)
                                                  for t in ("n1", "rel0", "rel1", "rel2", "relf0")]
    for d, tag in parts:
        part = np.load(d / ("preds_%s.npy" % tag), allow_pickle=True).item()
        tj = json.loads((d / ("thresholds_%s.json" % tag)).read_text())
        assert set(part) == set(tj), tag
        for a in part:
            assert a not in P, a
            P[a], T[a], src[a] = part[a], tj[a], tag
    # also: no other preds file in trees_out that clean_score would have loaded
    extra = sorted(f.name for f in (NEW / "trees_out").glob("preds_*.npy"))
    return P, T, src, extra


def first_split(labs):
    """(file index, window index) just after the first seizure run ends."""
    for fi, lab in enumerate(labs):
        lab = np.asarray(lab).astype(bool)
        if lab.any():
            on = np.flatnonzero(lab)
            start = on[0]
            end = start
            while end < len(lab) and lab[end]:
                end += 1
            return fi, end
    raise ValueError


def stage_tables():
    from score_official import count
    P, T, src, extra = load_arms()
    cases = sorted(P["A0"])
    print("arms:", list(P), "| preds files in trees_out:", extra)
    targets = sorted(T["A0"][cases[0]], key=float)
    print("targets:", targets)
    for a in P:
        assert sorted(P[a]) == cases
        for c in cases:
            assert sorted(T[a][c], key=float) == targets, (a, c)
    blocks_meta = {}
    for c in cases:
        fs = sorted((OLD / "feats").glob("%s__*.npz" % c))
        labs, flats, stems = [], [], []
        for f in fs:
            with np.load(f) as d:
                labs.append(d["labels"])
                flats.append(np.isnan(d["feats"][:, :, 0]))
            stems.append(f.stem.split("__", 1)[1])
        fi, wi = first_split(labs)
        meta = []
        for i in range(fi, len(fs)):
            lo = wi if i == fi else 0
            if len(labs[i]) - lo >= 10:
                fr = flats[i][lo:].mean(0)
                meta.append({"stem": stems[i], "n_flat_ch": int((fr > 0.5).sum()),
                             "labels": labs[i][lo:]})
        assert len(meta) == len(P["A0"][c]), (c, len(meta), len(P["A0"][c]))
        for m, (_, y) in zip(meta, P["A0"][c]):
            assert np.array_equal(np.asarray(m["labels"], np.int64), np.asarray(y, np.int64)), c
        blocks_meta[c] = meta
    # every arm: same labels/lengths as A0, no NaN
    for a in P:
        for c in cases:
            assert len(P[a][c]) == len(P["A0"][c])
            for (p, y), (p0, y0) in zip(P[a][c], P["A0"][c]):
                assert len(p) == len(y0) and np.array_equal(np.asarray(y), np.asarray(y0))
                assert np.isfinite(np.asarray(p, np.float64)).all()
    arms = list(P)
    nb = sum(len(blocks_meta[c]) for c in cases)
    # per-block counts
    TP = np.zeros((len(arms), nb, len(targets)), np.int64)
    FP = np.zeros_like(TP)
    REF = np.zeros(nb, np.int64)
    SEC = np.zeros(nb)
    case_of = []
    stem_of = []
    nflat = []
    t0 = time.time()
    j = 0
    for c in cases:
        for bi, m in enumerate(blocks_meta[c]):
            case_of.append(c)
            stem_of.append(m["stem"])
            nflat.append(m["n_flat_ch"])
            for ai, a in enumerate(arms):
                p, y = P[a][c][bi]
                for ti, tg in enumerate(targets):
                    tp, fp, ref, sec = count([(np.asarray(p, np.float64), np.asarray(y))],
                                             float(T[a][c][tg]), 5, 3)
                    TP[ai, j, ti], FP[ai, j, ti] = tp, fp
                    if ai == 0 and ti == 0:
                        REF[j], SEC[j] = ref, sec
                    else:
                        assert REF[j] == ref and SEC[j] == sec
            j += 1
        print("  %s done, %d blocks, %.0f s" % (c, len(blocks_meta[c]), time.time() - t0), flush=True)
    np.savez(OUT / "tables.npz", TP=TP, FP=FP, REF=REF, SEC=SEC, arms=np.array(arms),
             targets=np.array(targets), case_of=np.array(case_of), stem_of=np.array(stem_of),
             nflat=np.array(nflat))
    print("saved tables.npz")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="tables")
    args = ap.parse_args()
    if args.stage == "tables":
        stage_tables()
