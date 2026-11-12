"""
Replication, step 1: inference only, for the 17 lopo_ folds that were not in the pilot.

For every recording, each of the 17 fold models saves every window's 32-d pooled vector
(float16) and logit difference (float32). Same code path as pilot_infer.py.
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

from pilot_infer import load, run, PREP, CKPT, TRIAL  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")

    folds = {}
    for path in sorted(CKPT.glob("lopo_*.pt")):
        name = path.stem[len("lopo_"):]
        if name in TRIAL:
            continue
        m, held = load(path, device)
        assert "_".join(held) == name, (name, held)
        folds[name] = m
    assert len(folds) == 17, sorted(folds)
    print("new folds: %s" % ", ".join(folds), flush=True)

    cases = sorted(d.name for d in PREP.iterdir() if d.is_dir())
    t0, n_win = time.time(), 0
    for c in cases:
        for f in sorted((PREP / c).glob("*.npz")):
            with np.load(f) as d:
                X, y = d["segments"], d["labels"]
            save = {"labels": y.astype(np.int8)}
            for name, m in folds.items():
                lg, em = run(m, X, device, want_emb=True)
                save["logit_" + name] = lg
                save["emb_" + name] = em
            np.savez(out / ("%s__%s.npz" % (c, f.stem)), **save)
            n_win += len(y)
        print("  %s done, %d windows so far, %.0f s" % (c, n_win, time.time() - t0),
              flush=True)
    print("done: %d windows in %.0f min" % (n_win, (time.time() - t0) / 60))


if __name__ == "__main__":
    main()
