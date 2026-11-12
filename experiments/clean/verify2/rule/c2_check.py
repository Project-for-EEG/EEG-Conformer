"""C2 re-check (read-only, CPU): every fixed15 checkpoint has early_stop False, epochs 15,
n_train_cases = 24 - held-out cases, val_patients equal to the name. No data folder read."""
import sys
from pathlib import Path
sys.dont_write_bytecode = True
import torch
fs = sorted(Path(r"C:\EEG\checkpoints\lopo").glob("fixed15_*.pt"))
print("files:", len(fs))
bad = []
for p in fs:
    ck = torch.load(p, map_location="cpu", weights_only=False)
    name = p.stem[len("fixed15_"):]
    held = name.split("_")
    row = (name, ck.get("early_stop"), ck.get("epochs"), ck.get("n_train_cases"), list(ck.get("val_patients", [])), list(ck.get("held_out", [])))
    ok = ck.get("early_stop") is False and ck.get("epochs") == 15 and ck.get("n_train_cases") == 24 - len(held) and list(ck["val_patients"]) == held
    if not ok:
        bad.append(row)
print("failing:", bad)
