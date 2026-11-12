"""leakage audit, part 2 + 5: checkpoint metadata and A0 vs N1 state_dict comparison.
CPU only, read-only. Run with cwd C:\\EEG."""
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, r"C:\EEG")
sys.dont_write_bytecode = True
from train_lopo import subject_groups  # noqa: E402

CK = Path(r"C:\EEG\checkpoints\lopo")
groups = subject_groups()
all_cases = [c for g in groups for c in g]
print("groups:", len(groups), "cases:", len(all_cases))
print("fold order:", ["_".join(g) for g in groups])

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]

rows = []
for i, g in enumerate(groups):
    name = "_".join(g)
    pn, pa = CK / f"fixed15_{name}.pt", CK / f"lopo_{name}.pt"
    n = torch.load(pn, map_location="cpu", weights_only=False)
    a = torch.load(pa, map_location="cpu", weights_only=False)
    train_implied = [c for gg in groups if gg is not g for c in gg]
    overlap = sorted(set(n["val_patients"]) & set(train_implied))
    sdn, sda = n["model_state_dict"], a["model_state_dict"]
    assert list(sdn.keys()) == list(sda.keys())
    maxdiff, nident, ntens, fmax, nbt = 0.0, 0, 0, 0.0, {}
    for k in sdn:
        tn, ta = sdn[k], sda[k]
        assert tn.shape == ta.shape, k
        ntens += 1
        if tn.dtype.is_floating_point:
            d = (tn.double() - ta.double()).abs().max().item() if tn.numel() else 0.0
        else:
            d = float((tn != ta).sum().item())
        if not tn.dtype.is_floating_point:
            nbt.setdefault(k, (tn.tolist(), ta.tolist()))
        else:
            fmax = max(fmax, d)
        if torch.equal(tn, ta):
            nident += 1
        maxdiff = max(maxdiff, d)
    rows.append(dict(
        fold=i + 1, name=name,
        n_keys=sorted(set(n) - {"model_state_dict"}),
        a_keys=sorted(set(a) - {"model_state_dict"}),
        val_patients=list(n["val_patients"]), held_out=list(n["held_out"]),
        val_eq_group=list(n["val_patients"]) == g and list(n["held_out"]) == g,
        a0_val_eq_group=list(a["val_patients"]) == g,
        n_train_cases=n["n_train_cases"], want=len(train_implied),
        a0_n_train_cases=a.get("n_train_cases"),
        held_in_train=overlap,
        early_stop=n.get("early_stop"), epochs=n.get("epochs"),
        max_segments=n.get("max_segments"), a0_max_segments=a.get("max_segments"),
        drop=n.get("drop_channels"), always=n.get("always_train"),
        a0_drop=a.get("drop_channels"), a0_always=a.get("always_train"),
        cfg_eq=n["model_config"] == a["model_config"],
        n_val_seizure=n.get("n_val_seizure"), a0_n_val_seizure=a.get("n_val_seizure"),
        float_max_abs_diff=fmax, int_tensors=nbt,
        tensors=ntens, tensors_identical=nident, max_abs_param_diff=maxdiff,
        sha_n=sha(pn), sha_a=sha(pa),
        mtime_n=datetime.fromtimestamp(os.path.getmtime(pn)).isoformat(),
        mtime_a=datetime.fromtimestamp(os.path.getmtime(pa)).isoformat(),
        metrics_n=n["metrics"], metrics_a=a["metrics"],
    ))

bad = [r["name"] for r in rows if not (r["val_eq_group"] and r["n_train_cases"] == r["want"]
       and not r["held_in_train"] and r["early_stop"] is False and r["epochs"] == 15
       and r["max_segments"] == 2000 and not r["drop"] and not r["always"] and r["cfg_eq"])]
print("C2-style failures:", bad)
print("model_config N1:", json.dumps(torch.load(CK / "fixed15_CHB02.pt", map_location="cpu",
                                                weights_only=False)["model_config"]))
print("\nfold name             n_train want a0_ntr es    ep  nvs/a0nvs  identical/tensors  max|dparam|  sha_n sha_a")
for r in rows:
    print("%2d %-16s %5s %5s %5s %-5s %3s %5s/%-5s %3d/%3d  %.6g  %s %s  %s  %s" % (
        r["fold"], r["name"], r["n_train_cases"], r["want"], r["a0_n_train_cases"],
        r["early_stop"], r["epochs"], r["n_val_seizure"], r["a0_n_val_seizure"],
        r["tensors_identical"], r["tensors"], r["max_abs_param_diff"], r["sha_n"], r["sha_a"],
        r["mtime_n"][:19], r["mtime_a"][:19]))
print("\nkeys N1:", rows[0]["n_keys"])
print("keys A0:", rows[0]["a_keys"])
print("\nmetrics fold1 N1:", rows[0]["metrics_n"])
print("metrics fold1 A0:", rows[0]["metrics_a"])
for r in rows:
    same = all(abs(r["metrics_n"].get(k, np.nan) - r["metrics_a"].get(k, np.nan)) < 1e-12
               for k in r["metrics_a"])
    print("  %-14s metrics identical A0/N1: %s  N1 auc %.6f A0 auc %.6f" % (
        r["name"], same, r["metrics_n"].get("auc", np.nan), r["metrics_a"].get("auc", np.nan)))
print("\nfloat-only max|diff| and integer buffers (N1, A0):")
for r in rows:
    ib = {k: v for k, v in r["int_tensors"].items()}
    vals = sorted(set((tuple([v[0]]), tuple([v[1]])) for v in ib.values()))
    print("  %-14s float max|d| %.6g | int buffers %d, distinct (N1,A0) values %s" % (
        r["name"], r["float_max_abs_diff"], len(ib), vals))
out = Path(__file__).with_name("v1_checkpoints.json")
out.write_text(json.dumps(rows, indent=1, default=str))
print("saved", out)
