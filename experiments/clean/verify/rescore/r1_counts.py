"""Independent rescoring, step 1: per-block official counts for every arm/case/target.
Uses only score_official.count and numpy (plus json/os). Writes counts.npy here."""
import os, sys, json, time
import numpy as np
sys.path.insert(0, r"C:\EEG")
from score_official import count

T = r"C:\Users\user\AppData\Local\Temp\claude\36c1f29744f93fe6"
OLD = os.path.join(T, "b3889703-629c-4c5a-b583-69f722743ff4", "scratchpad", "trees", "out")
NEW = os.path.join(T, "c7848ca6-bfef-414b-a321-d6686c1a43f8", "scratchpad", "clean", "trees_out")
HERE = os.path.dirname(os.path.abspath(__file__))

SRC = {  # arm -> (preds file, thresholds file)
    "A0": (OLD, "a"), "Tamp_s0": (OLD, "a"), "Tamp_s1": (OLD, "b"), "Tamp_s2": (OLD, "c"),
    "Trel_s0": (NEW, "rel0"), "Trel_s1": (NEW, "rel1"), "Trel_s2": (NEW, "rel2"),
    "Trelf_s0": (NEW, "relf0"),
}
TARGETS = ["0.05", "0.1", "0.15", "0.2", "0.3", "0.4", "0.5", "0.75", "1.0", "1.5", "2.0",
           "3.0", "4.0", "5.0"]

cache = {}
def load(d, tag):
    if (d, tag) not in cache:
        p = np.load(os.path.join(d, "preds_%s.npy" % tag), allow_pickle=True).item()
        with open(os.path.join(d, "thresholds_%s.json" % tag)) as f:
            t = json.load(f)
        cache[(d, tag)] = (p, t)
    return cache[(d, tag)]

out = {}
ref_labels = None
t0 = time.time()
for arm, (d, tag) in SRC.items():
    P, TH = load(d, tag)
    assert arm in P and arm in TH, (arm, tag, list(P), list(TH))
    cases = sorted(P[arm])
    assert len(cases) == 24 and sorted(TH[arm]) == cases
    # labels / block lengths identical to A0's
    labs = {c: [np.asarray(b[1]).astype(int) for b in P[arm][c]] for c in cases}
    if ref_labels is None:
        ref_labels = labs
    else:
        for c in cases:
            assert len(labs[c]) == len(ref_labels[c]), (arm, c)
            for a, b in zip(labs[c], ref_labels[c]):
                assert a.shape == b.shape and np.array_equal(a, b), (arm, c)
    out[arm] = {}
    for c in cases:
        blocks = P[arm][c]
        for b in blocks:
            assert len(b[0]) == len(b[1]) and np.all(np.isfinite(b[0]))
        out[arm][c] = {}
        for tg in TARGETS:
            th = float(TH[arm][c][tg])
            out[arm][c][tg] = [count([(b[0], b[1])], th, 5, 3) for b in blocks]
    print("%-9s done  %.0f s" % (arm, time.time() - t0), flush=True)

np.save(os.path.join(HERE, "counts.npy"), np.array(out, dtype=object), allow_pickle=True)
print("saved")
