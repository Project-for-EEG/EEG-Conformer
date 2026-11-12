"""Robustness critic, step 1: per-case deployable tables (155 set) + global-threshold sweep.
Reads only. Writes verify/robust/tables.npz. Run with cwd C:\\EEG, PYTHONDONTWRITEBYTECODE=1."""
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\EEG")
from score_official import count                      # noqa: E402
from evaluate_temporal import smooth                  # noqa: E402
from finetune_per_patient import patient_files, find_split  # noqa: E402

OLD = os.path.join(r"C:\Users\user\AppData\Local\Temp\claude\36c1f29744f93fe6",
                   r"b3889703-629c-4c5a-b583-69f722743ff4\scratchpad\trees")
NEW = os.path.join(r"C:\Users\user\AppData\Local\Temp\claude\36c1f29744f93fe6",
                   r"c7848ca6-bfef-414b-a321-d6686c1a43f8\scratchpad\clean")
OUT = os.path.join(NEW, "verify", "robust")
TARGETS = ("0.05", "0.1", "0.15", "0.2", "0.3", "0.4", "0.5", "0.75", "1.0", "1.5", "2.0",
           "3.0", "4.0", "5.0")
ARMS = ["A0", "Tamp_s0", "Tamp_s1", "Tamp_s2", "Trel_s0", "Trel_s1", "Trel_s2"]
K, N_RUN = 5, 3
EXCL = {"chb12_27_labeled", "chb12_28_labeled", "chb12_29_labeled"}


def load():
    P, T = {}, {}
    for d, tag in [(os.path.join(OLD, "out"), t) for t in "abc"] + \
                  [(os.path.join(NEW, "trees_out"), "rel%d" % i) for i in range(3)]:
        part = np.load(os.path.join(d, "preds_%s.npy" % tag), allow_pickle=True).item()
        tp = json.load(open(os.path.join(d, "thresholds_%s.json" % tag)))
        for a in part:
            if a in ARMS:
                P[a], T[a] = part[a], tp[a]
    return P, T


def stems(c, n_blocks):
    files = patient_files(c)
    fi, wi = find_split(files, 1)
    out = []
    for i, f in enumerate(files):
        if i < fi:
            continue
        with np.load(os.path.join(OLD, "feats", "%s__%s.npz" % (c, f.stem))) as d:
            n = len(d["labels"])
        lo = wi if i == fi else 0
        if n - lo >= 10:
            out.append(f.stem)
    assert len(out) == n_blocks, (c, len(out), n_blocks)
    return out


def deploy_job(job):
    a, blocks_by_case, thr, cases = job
    tp = np.zeros((len(cases), len(TARGETS)), np.int64)
    fp = np.zeros_like(tp)
    ref = np.zeros(len(cases), np.int64)
    sec = np.zeros(len(cases))
    for i, c in enumerate(cases):
        for j, tg in enumerate(TARGETS):
            x = count(blocks_by_case[c], float(thr[c][tg]), K, N_RUN)
            tp[i, j], fp[i, j], ref[i], sec[i] = x
    return a, tp, fp, ref, sec


def sweep_job(job):
    a, blocks_by_case, cases, grid = job
    tp = np.zeros((len(cases), len(grid)), np.int64)
    fp = np.zeros_like(tp)
    for i, c in enumerate(cases):
        for g, t in enumerate(grid):
            x = count(blocks_by_case[c], float(t), K, N_RUN)
            tp[i, g], fp[i, g] = x[0], x[1]
    return a, tp, fp


def main():
    P, T = load()
    cases = sorted(P["A0"])
    keep = {}
    for c in cases:
        st = stems(c, len(P["A0"][c]))
        keep[c] = [s not in EXCL for s in st]
        if c == "CHB12":
            print("CHB12 excluded block indices:", [i for i, k in enumerate(keep[c]) if not k],
                  [st[i] for i, k in enumerate(keep[c]) if not k])
    bl = {a: {c: [(np.asarray(p, np.float64), np.asarray(y)) for (p, y), k in
                  zip(P[a][c], keep[c]) if k] for c in cases} for a in ARMS}
    thr = {a: {c: {tg: float(T[a][c][tg]) for tg in TARGETS} for c in cases} for a in ARMS}
    # a global grid: dense in the upper tail of each arm's own smoothed test output
    grids = {}
    for a in ARMS:
        sm = np.concatenate([smooth(p, K) for c in cases for p, _ in bl[a][c]])
        q = 1.0 - np.logspace(np.log10(0.2), -6.0, 220)
        grids[a] = np.unique(np.concatenate([np.quantile(sm, q), np.linspace(0.05, 0.995, 60)]))
    with Pool(7) as pool:
        dep = pool.map(deploy_job, [(a, bl[a], thr[a], cases) for a in ARMS])
        sw = pool.map(sweep_job, [(a, bl[a], cases, grids[a]) for a in ARMS])
    out = {"cases": np.array(cases), "targets": np.array(TARGETS, dtype=float)}
    for a, tp, fp, ref, sec in dep:
        out[a + "__tp"], out[a + "__fp"], out[a + "__ref"], out[a + "__sec"] = tp, fp, ref, sec
        out[a + "__thr"] = np.array([[thr[a][c][tg] for tg in TARGETS] for c in cases])
    for a, tp, fp in sw:
        out[a + "__gtp"], out[a + "__gfp"], out[a + "__grid"] = tp, fp, grids[a]
    np.savez(os.path.join(OUT, "tables.npz"), **out)
    for a in ARMS:
        j = TARGETS.index("2.0")
        print("%-8s %d/%d  %.1f/day" % (a, out[a + "__tp"][:, j].sum(), out[a + "__ref"].sum(),
                                         out[a + "__fp"][:, j].sum() / (out[a + "__sec"].sum() / 86400)))


if __name__ == "__main__":
    main()
