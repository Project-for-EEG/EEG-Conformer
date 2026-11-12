"""Thresholds Tamp vs Trel per fold; FA right after each case's split (Trel history holds
the first seizure there by construction); realised/nominal FA ratio."""
import json, os, sys, warnings
sys.dont_write_bytecode = True
import numpy as np
sys.path.insert(0, r"C:\EEG")
from finetune_per_patient import patient_files, find_split
from score_official import detect, MASK_FS
from timescoring import scoring
from timescoring.annotations import Annotation
O = r"C:\Users\user\AppData\Local\Temp\claude\36c1f29744f93fe6\b3889703-629c-4c5a-b583-69f722743ff4\scratchpad\trees\out"
N = r"C:\Users\user\AppData\Local\Temp\claude\36c1f29744f93fe6\c7848ca6-bfef-414b-a321-d6686c1a43f8\scratchpad\clean\trees_out"
F = r"C:\Users\user\AppData\Local\Temp\claude\36c1f29744f93fe6\b3889703-629c-4c5a-b583-69f722743ff4\scratchpad\trees\feats"
EXCL = {"chb12_27_labeled", "chb12_28_labeled", "chb12_29_labeled"}
warnings.simplefilter("ignore")
for k, (ot, nt) in enumerate([("a", "rel0"), ("b", "rel1"), ("c", "rel2")]):
    La = {r["fold"]: r["t_2fa"] for r in json.load(open(os.path.join(O, "train_log_%s.json" % ot))) if r["arm"] == "Tamp_s%d" % k}
    Lr = {r["fold"]: r["t_2fa"] for r in json.load(open(os.path.join(N, "train_log_%s.json" % nt)))}
    d = np.array([Lr[f] - La[f] for f in La])
    print("seed %d: t@2FA/h mean Tamp %.4f, Trel %.4f; Trel-Tamp per fold median %+.4f, Trel higher in %d/23 folds"
          % (k, np.mean(list(La.values())), np.mean(list(Lr.values())), np.median(d), int(np.sum(d > 0))))
# FA in the first 300 test windows after the split (split file only), at target 2.0
arms = {}
for k, (ot, nt) in enumerate([("a", "rel0"), ("b", "rel1"), ("c", "rel2")]):
    arms["Tamp_s%d" % k] = (np.load(os.path.join(O, "preds_%s.npy" % ot), allow_pickle=True).item()["Tamp_s%d" % k],
                           json.load(open(os.path.join(O, "thresholds_%s.json" % ot)))["Tamp_s%d" % k])
    arms["Trel_s%d" % k] = (np.load(os.path.join(N, "preds_%s.npy" % nt), allow_pickle=True).item()["Trel_s%d" % k],
                           json.load(open(os.path.join(N, "thresholds_%s.json" % nt)))["Trel_s%d" % k])
cases = sorted(arms["Tamp_s0"][0])
first = {}
for c in cases:
    files = patient_files(c); fi, wi = find_split(files, 1)
    with np.load(os.path.join(F, "%s__%s.npz" % (c, files[fi].stem))) as dd:
        n = len(dd["labels"])
    first[c] = (n - wi >= 10) and files[fi].stem not in EXCL   # block 0 is the split file
hrs = sum(min(300, len(arms["Tamp_s0"][0][c][0][0])) for c in cases if first[c]) * 2 / 3600
for tg in ("2.0", "0.05"):
    out = []
    for a in arms:
        P, T = arms[a]
        fp = 0
        for c in cases:
            if not first[c]:
                continue
            p, y = P[c][0]
            pred = detect(np.asarray(p, np.float64), float(T[c][tg]), 5, 3)
            s = scoring.EventScoring(Annotation(np.asarray(y).astype(bool), MASK_FS), Annotation(pred, MASK_FS))
            fp += sum(1 for ev in s.hyp.events if ev[0] * MASK_FS < 300 and np.all(~s.tpMask[round(ev[0] * s.fs):round(ev[1] * s.fs)]))
        out.append("%s %d" % (a, fp))
    print("target %s: FA starting in the first 300 windows after the split (%.1f h over %d cases): %s"
          % (tg, hrs, sum(first.values()), ", ".join(out)))
