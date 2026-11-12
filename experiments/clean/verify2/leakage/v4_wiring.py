"""leakage audit, parts 4 + 6: N1 tree-pipeline wiring. CPU only, read-only. cwd C:\\EEG.

Independent of trees_train.py's own helpers (calib_bg, thresholds_for, test_blocks): uses the
committed finetune_per_patient / calibrate_per_patient functions directly, with the fold
model's cached logits in place of predict_file.
  a) test blocks of every case rebuilt from n1_cache logit_<owner fold>; compared bit-for-bit
     with NEW\\trees_out\\preds_n1.npy; and which fold key reproduces each case's blocks.
  b) thresholds per fold from TRAINING patients' pre-first-seizure reserved background,
     rng 42, via the committed pick_threshold at all 14 targets; compared with
     thresholds_n1.json and with the per-fold values printed in infer_n1.log.
  c) the same rebuild for A0 from the A0 caches vs OLD\\out preds (same pipeline check).
  d) every arm's test blocks: same cases, block count, lengths, labels as N1.
"""
import json
import re
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

REPO = Path(r"C:\EEG")
sys.path.insert(0, str(REPO))
sys.dont_write_bytecode = True
from finetune_per_patient import patient_files, find_split, adapt_background_mask  # noqa: E402
from calibrate_per_patient import pick_threshold  # noqa: E402

NEW = Path(r"C:\Users\user\AppData\Local\Temp\claude\36c1f29744f93fe6"
           r"\c7848ca6-bfef-414b-a321-d6686c1a43f8\scratchpad\clean")
OLD = Path(r"C:\Users\user\AppData\Local\Temp\claude\36c1f29744f93fe6"
           r"\b3889703-629c-4c5a-b583-69f722743ff4\scratchpad\trees")
OLD2 = Path(r"C:\Users\user\AppData\Local\Temp\claude\36c1f29744f93fe6"
            r"\f241281f-add4-41ac-9210-f47dab0b83b0\scratchpad")
N1C = NEW / "n1_cache"
TARGETS = (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0)
TRIAL = ["CHB12", "CHB15", "CHB24", "CHB13", "CHB06", "CHB16"]


def folds_from_ckpt_names():
    import torch
    out = {}
    for p in sorted((REPO / "checkpoints" / "lopo").glob("fixed15_*.pt")):
        ck = torch.load(p, map_location="cpu", weights_only=False)
        out[p.stem[len("fixed15_"):]] = list(ck["val_patients"])
    return out


def sig(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


def per_file_logits(c, key, src="n1"):
    res = []
    for f in patient_files(c):
        name = "%s__%s.npz" % (c, f.stem)
        if src == "n1":
            with np.load(N1C / name) as d:
                res.append(d["logit_" + key].copy())
        else:
            cdir = OLD2 / ("pilot_cache" if key in TRIAL else "pilot_cache2")
            with np.load(cdir / name) as d:
                res.append(d["logit_" + key].copy())
    return res


def labels_of(c):
    out = []
    for f in patient_files(c):
        with np.load(f) as d:
            out.append(d["labels"].copy())
    return out


def blocks(c, probs, labs):
    fi, wi = find_split(patient_files(c), 1)
    out = []
    for i, (p, y) in enumerate(zip(probs, labs)):
        if i < fi:
            continue
        if i == fi:
            p, y = p[wi:], y[wi:]
        if len(y) >= 10:
            out.append((p, y))
    return out


def fold_threshold(args):
    fold, held, cases = args
    train = [q for q in cases if q not in held]
    rng = np.random.default_rng(42)
    pooled, n_from_case = [], {}
    for q in train:
        files = patient_files(q)
        split = find_split(files, 1)
        masks = adapt_background_mask(files, split, 0.3, rng)
        fi, wi = split
        lg = per_file_logits(q, fold)
        labs = labels_of(q)
        reserved, everything = [], []
        for i in range(fi + 1):
            p, y = sig(lg[i]), labs[i]
            if i == fi:
                p, y = p[:wi], y[:wi]
            bg = y == 0
            reserved.append(p[masks[i] & bg])
            everything.append(p[bg])
        h = np.concatenate(reserved)
        h = h if len(h) >= 200 else np.concatenate(everything)
        pooled.append(np.asarray(h, dtype=np.float32))
        n_from_case[q] = len(h)
    bg = np.concatenate(pooled)
    t = {tg: pick_threshold(bg, tg, 5, 3) for tg in TARGETS}
    return fold, held, len(bg), n_from_case, t


def main():
    folds = folds_from_ckpt_names()
    cases = sorted(c for h in folds.values() for c in h)
    owner = {c: f for f, h in folds.items() for c in h}
    print("folds:", len(folds), "cases:", len(cases))

    P1 = np.load(NEW / "trees_out" / "preds_n1.npy", allow_pickle=True).item()
    T1 = json.loads((NEW / "trees_out" / "thresholds_n1.json").read_text())
    print("preds_n1 arms:", list(P1), "| thresholds_n1 arms:", list(T1))
    P1, T1 = P1["N1"], T1["N1"]
    print("preds_n1 cases:", len(P1), sorted(P1) == cases)

    # ---- a) test blocks
    worst = 0.0
    repro = {}
    for c in cases:
        labs = labels_of(c)
        mine = blocks(c, [sig(l) for l in per_file_logits(c, owner[c])], labs)
        st = P1[c]
        assert len(mine) == len(st), (c, len(mine), len(st))
        for (p, y), (ps, ys) in zip(mine, st):
            assert np.array_equal(np.asarray(y, np.int64), np.asarray(ys, np.int64)), c
            worst = max(worst, float(np.max(np.abs(p - np.asarray(ps)))))
        # which fold keys reproduce the stored blocks exactly
        hits = []
        for f in folds:
            b = blocks(c, [sig(l) for l in per_file_logits(c, f)], labs)
            if all(np.array_equal(p, np.asarray(ps)) for (p, _), (ps, _) in zip(b, st)):
                hits.append(f)
        repro[c] = hits
    print("a) max |stored - rebuilt from logit_<owner>| over all test windows: %.3g" % worst)
    print("   fold keys whose logits reproduce each case's stored N1 blocks exactly:")
    for c in cases:
        print("     %s owner %-12s -> %s %s" % (c, owner[c], repro[c],
                                                "OK" if repro[c] == [owner[c]] else "PROBLEM"))

    # ---- b) thresholds
    jobs = [(f, h, cases) for f, h in folds.items()]
    with Pool(3) as pool:
        res = pool.map(fold_threshold, jobs)
    log = (NEW / "infer_n1.log").read_text(encoding="utf-8", errors="replace")
    printed = {}
    for m in re.finditer(r"fixed15_(\S+)\.pt\s+->\s+(.*?)\s+threshold ([\d.]+) from (\d+) training", log):
        printed[m.group(1)] = (float(m.group(3)), int(m.group(4)))
    print("b) fold, n_bg (mine / infer_n1.log), t@2.0 mine / json / log, max |mine-json| over 14 targets")
    tw_json, tw_log, nbad = 0.0, 0.0, []
    out_rows = []
    for fold, held, nbg, nfc, t in res:
        js = [T1[c] for c in held]
        assert all(j == js[0] for j in js), ("held cases of one fold have different thresholds", fold)
        dj = max(abs(t[tg] - float(js[0][str(tg)])) for tg in TARGETS)
        tw_json = max(tw_json, dj)
        pl, pn = printed[fold]
        tw_log = max(tw_log, abs(round(t[2.0], 4) - pl))
        if pn != nbg:
            nbad.append((fold, nbg, pn))
        assert not (set(held) & set(nfc)), fold
        print("   %-12s %6d / %6d   %.6f / %.6f / %.4f   %.3g" % (
            fold, nbg, pn, t[2.0], float(js[0]["2.0"]), pl, dj))
        out_rows.append(dict(fold=fold, held=held, n_bg=nbg, n_bg_log=pn, bg_cases=sorted(nfc),
                             t_mine={str(k): v for k, v in t.items()}, t_json=js[0], t_log=pl))
    print("   max |mine - thresholds_n1.json| over folds x 14 targets: %.3g" % tw_json)
    print("   max |round(mine@2.0,4) - infer_n1.log|: %.3g" % tw_log)
    print("   background-count mismatches vs infer_n1.log:", nbad)

    # ---- c) A0 same pipeline
    PA = {}
    for t in "abcd":
        part = np.load(OLD / "out" / ("preds_%s.npy" % t), allow_pickle=True).item()
        for a in part:
            assert a not in PA
            PA[a] = part[a]
    for t in ("rel0", "rel1", "rel2", "relf0"):
        part = np.load(NEW / "trees_out" / ("preds_%s.npy" % t), allow_pickle=True).item()
        for a in part:
            assert a not in PA
            PA[a] = part[a]
    print("c/d) arms found:", sorted(PA))
    worstA = 0.0
    for c in cases:
        labs = labels_of(c)
        mine = blocks(c, [sig(l) for l in per_file_logits(c, owner[c], "a0")], labs)
        assert len(mine) == len(PA["A0"][c])
        for (p, y), (ps, ys) in zip(mine, PA["A0"][c]):
            assert np.array_equal(np.asarray(y, np.int64), np.asarray(ys, np.int64))
            worstA = max(worstA, float(np.max(np.abs(p - np.asarray(ps)))))
    print("c) A0: max |stored - rebuilt from A0 cache logit_<owner>|: %.3g" % worstA)

    # ---- d) all arms' test blocks identical in structure to N1
    bad = []
    nblk, nwin, nev = 0, 0, 0
    for a, Pa in PA.items():
        if sorted(Pa) != cases:
            bad.append((a, "cases"))
            continue
        for c in cases:
            if len(Pa[c]) != len(P1[c]):
                bad.append((a, c, "nblocks"))
                continue
            for (p, y), (p1, y1) in zip(Pa[c], P1[c]):
                if len(p) != len(y1) or not np.array_equal(np.asarray(y, np.int64), np.asarray(y1, np.int64)):
                    bad.append((a, c, "labels/len"))
    for c in cases:
        for p, y in P1[c]:
            nblk += 1
            nwin += len(y)
            y = np.asarray(y).astype(bool)
            nev += int(np.sum(y[1:] & ~y[:-1]) + y[0])
    print("d) arms whose test blocks differ from N1 in cases/count/length/labels:", bad)
    print("   N1 test set: %d blocks, %d windows, %d label runs (events by run count)" % (nblk, nwin, nev))
    (Path(__file__).with_name("v4_wiring.json")).write_text(json.dumps(
        dict(max_block_diff=worst, repro=repro, thresholds=out_rows, max_t_json=tw_json,
             max_t_log=tw_log, nbg_mismatch=nbad, a0_block_diff=worstA, arm_mismatch=bad,
             n_blocks=nblk, n_windows=nwin, n_runs=nev), indent=1, default=str))


if __name__ == "__main__":
    main()
