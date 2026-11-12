"""
N1 inference (clean_rule.txt v2, section 1). GPU. Run with cwd C:\\EEG.

For every recording in preprocessed_data, each of the 23 fixed15_ fold models saves every
window's logit difference (float32), the same quantity the old pilot_infer.py saved for A0
(load/run copied here; nothing is imported from the old scratchpad). No fold is skipped.
Writes one .npz per recording to <out>/. Nothing in C:\\EEG is written.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(r"C:\EEG")
sys.path.insert(0, str(REPO))
sys.dont_write_bytecode = True

from config import ModelConfig            # noqa: E402
from model import create_model            # noqa: E402

PREP = REPO / "preprocessed_data"
CKPT = REPO / "checkpoints" / "lopo"
PREFIX = "fixed15_"


def load(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    assert ck.get("early_stop") is False, (path.name, ck.get("early_stop"))
    assert ck.get("epochs") == 15, (path.name, ck.get("epochs"))
    m = create_model(ModelConfig(**ck["model_config"])).to(device)
    m.load_state_dict(ck["model_state_dict"])
    m.eval()
    return m, list(ck["val_patients"])


def run(model, X, device, batch=512):
    """Logit difference (class 1 minus class 0), float32, for every window."""
    logits = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            b = torch.from_numpy(X[i:i + batch]).float().to(device)
            out = model(b)
            logits.append((out[:, 1] - out[:, 0]).double().cpu().numpy())
    return np.concatenate(logits).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")

    cases = sorted(d.name for d in PREP.iterdir() if d.is_dir())
    folds = {}
    for path in sorted(CKPT.glob(PREFIX + "*.pt")):
        name = path.stem[len(PREFIX):]
        m, held = load(path, device)
        assert "_".join(held) == name, (name, held)
        folds[name] = m
    assert len(folds) == 23, sorted(folds)
    held_all = sorted(c for name in folds for c in name.split("_"))
    assert held_all == cases, (held_all, cases)
    print("%d fold models: %s" % (len(folds), ", ".join(folds)), flush=True)

    t0, n_win = time.time(), 0
    for c in cases:
        for f in sorted((PREP / c).glob("*.npz")):
            with np.load(f) as d:
                X, y = d["segments"], d["labels"]
            save = {"labels": y.astype(np.int8)}
            for name, m in folds.items():
                save["logit_" + name] = run(m, X, device)
            np.savez(out / ("%s__%s.npz" % (c, f.stem)), **save)
            n_win += len(y)
        print("  %s done, %d windows so far, %.0f s" % (c, n_win, time.time() - t0),
              flush=True)
    print("done: %d windows in %.0f min" % (n_win, (time.time() - t0) / 60))


if __name__ == "__main__":
    main()
