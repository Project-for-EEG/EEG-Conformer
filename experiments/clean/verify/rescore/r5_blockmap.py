"""Which CHB12 recordings are test blocks 6, 7, 8 (independent rebuild)."""
import os, sys
import numpy as np
sys.path.insert(0, r"C:\EEG")
from finetune_per_patient import patient_files, find_split
from evaluate_temporal import find_runs
T = r"C:\Users\user\AppData\Local\Temp\claude\36c1f29744f93fe6"
P = np.load(os.path.join(T, "b3889703-629c-4c5a-b583-69f722743ff4", "scratchpad", "trees", "out",
                         "preds_a.npy"), allow_pickle=True).item()["A0"]["CHB12"]
fs = patient_files("CHB12"); fi, wi = find_split(fs, 1)
names = []
for i in range(fi, len(fs)):
    with np.load(fs[i]) as d:
        y = d["labels"]
    if i == fi:
        y = y[wi:]
    if len(y) >= 10:
        names.append((os.path.basename(fs[i]), y))
assert len(names) == len(P)
for k, ((n, y), b) in enumerate(zip(names, P)):
    print(k, n, len(y), "labels equal" if np.array_equal(y, b[1]) else "DIFF",
          "label runs", len(find_runs(np.asarray(y).astype(bool))))
