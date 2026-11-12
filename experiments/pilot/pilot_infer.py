"""
Pilot, step 1: one inference pass over CHB-MIT. No training.

For every recording, loaded once:
  - the 6 trial fold models (lopo_ held-out CHB12, CHB15, CHB24, CHB13, CHB06, CHB16,
    the 6 cases with the most test seizures) save every window's 32-d pooled vector
    (float16) and logit difference (float32);
  - the owning fold model of each CHB-only family (lopo_, lopo20_, lopoA__) saves its
    logit difference for the ensemble check.

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

from config import ModelConfig            # noqa: E402
from model import create_model            # noqa: E402
from evaluate_temporal import match_channels  # noqa: E402

PREP = REPO / "preprocessed_data"
CKPT = REPO / "checkpoints" / "lopo"
TRIAL = ["CHB12", "CHB15", "CHB24", "CHB13", "CHB06", "CHB16"]
FAMILIES = {"lopo": "lopo_*.pt", "lopo20": "lopo20_*.pt", "lopoA": "lopoA__*.pt"}


def load(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    m = create_model(ModelConfig(**ck["model_config"])).to(device)
    m.load_state_dict(ck["model_state_dict"])
    m.eval()
    return m, list(ck["val_patients"])


def run(model, X, device, batch=512, want_emb=False):
    """(logit difference float32, pooled vector float16 or None) for every window."""
    grab = []
    hook = model.global_pool.register_forward_hook(
        lambda mod, inp, out: grab.append(out.detach().cpu())) if want_emb else None
    logits = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            b = torch.from_numpy(X[i:i + batch]).float().to(device)
            out = model(b)
            logits.append((out[:, 1] - out[:, 0]).double().cpu().numpy())
    if hook:
        hook.remove()
    d = np.concatenate(logits).astype(np.float32)
    e = torch.cat(grab).numpy().astype(np.float16) if want_emb else None
    return d, e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")

    trial = {}
    for p in TRIAL:
        m, held = load(CKPT / ("lopo_%s.pt" % p), device)
        assert held == [p], (p, held)
        trial[p] = m
    owners = {}  # family -> {case: model}
    for fam, pat in FAMILIES.items():
        owners[fam] = {}
        paths = sorted(CKPT.glob(pat))
        assert len(paths) == 23, (fam, len(paths))
        for path in paths:
            m, held = load(path, device)
            for c in held:
                owners[fam][c] = m
    cases = sorted(d.name for d in PREP.iterdir() if d.is_dir())
    for fam in owners:
        assert set(owners[fam]) == set(cases), fam
    print("trial models %s | families %s | %d cases"
          % (", ".join(TRIAL), ", ".join(owners), len(cases)), flush=True)

    t0, n_win = time.time(), 0
    for c in cases:
        for f in sorted((PREP / c).glob("*.npz")):
            with np.load(f) as d:
                X, y = d["segments"], d["labels"]
            save = {"labels": y.astype(np.int8)}
            for p, m in trial.items():
                lg, em = run(m, X, device, want_emb=True)
                save["logit_" + p] = lg
                save["emb_" + p] = em
            for fam in owners:
                m = owners[fam][c]
                lg, _ = run(m, match_channels(X, m), device)
                save["own_" + fam] = lg
            np.savez(out / ("%s__%s.npz" % (c, f.stem)), **save)
            n_win += len(y)
        print("  %s done, %d windows so far, %.0f s" % (c, n_win, time.time() - t0),
              flush=True)
    print("done: %d windows in %.0f min" % (n_win, (time.time() - t0) / 60))


if __name__ == "__main__":
    main()
