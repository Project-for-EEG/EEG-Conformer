"""
Honest held-out predictions from a family of leave-one-patient-out models.

A threshold can only be tuned on predictions from a model that never saw the
patient. The deployment model trained for the SzCORE submission holds out one
patient -- PN13, which has three seizure events -- and three events cannot
calibrate anything: one seizure going either way moves F1 by a quarter. So
that model cannot check its own operating point.

The leave-one-patient-out checkpoints already solve this. Each was trained
without one patient, so each can be graded on exactly that patient and nothing
else. Twenty-two of them cover the whole cohort, and pooling their held-out
predictions gives the ~168 unseen events a threshold needs, without training
anything new.

WHY NOT evaluate_end_to_end.py

    That script does the same patient-to-checkpoint mapping, but it has no
    channel-drop step, so handing it a 20-channel checkpoint and 23-channel
    recordings is a shape error. It also runs per-patient adaptation by
    default, which the submitted container cannot do: it is given one
    anonymous recording with no patient identity and no earlier session. A
    cache with adapted models in it would describe a system that is not the
    one being submitted. This writes base-model predictions only, in the same
    format, so retune_f1.py reads it unchanged.

Usage:
    python heldout_probs.py --ckpt-glob "lopo20siena_*.pt"
    python heldout_probs.py --ckpt-glob "lopo20siena_*.pt" --out mycache.npz
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from config import ModelConfig
from model import create_model

CKPT = Path("checkpoints/lopo")

# A patient's recordings live under whichever root holds that cohort. Getting
# this wrong is silent: a CHB checkpoint handed Siena windows still runs.
ROOTS = {"CHB": Path("preprocessed_data"),
         "PN": Path("preprocessed_data_siena23"),
         "HEL": Path("preprocessed_data_helsinki23")}


def root_for(patient):
    for prefix, root in ROOTS.items():
        if patient.startswith(prefix):
            return root
    raise ValueError("no preprocessed root known for %s" % patient)


def load_model(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    model = create_model(ModelConfig(**ck["model_config"])).to(device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    return model, ck


def blocks_for(patient, model, drop, device, batch=256):
    """[(probs, labels)] with one entry per recording, in time order.

    Recordings are kept separate rather than concatenated. Joining two
    together invents a boundary that is neither seizure nor background, and a
    detection spanning it would be booked as one event across both.
    """
    out = []
    for f in sorted(root_for(patient).joinpath(patient).glob("*.npz")):
        with np.load(f) as d:
            X, y = d["segments"], d["labels"]
        if drop:
            X = np.delete(X, drop, axis=1)
        probs = np.empty(len(X), dtype=np.float32)
        with torch.no_grad():
            for i in range(0, len(X), batch):
                t = torch.from_numpy(X[i:i + batch]).float().to(device)
                probs[i:i + len(t)] = torch.softmax(model(t), 1)[:, 1].cpu().numpy()
        out.append((probs, y))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt-glob", default="lopo20siena_*.pt")
    ap.add_argument("--out", default=None,
                    help="defaults to endtoend_<glob stem>.npz")
    args = ap.parse_args()

    paths = sorted(CKPT.glob(args.ckpt_glob))
    if not paths:
        raise SystemExit("no checkpoints match %s in %s" % (args.ckpt_glob, CKPT))
    out_path = args.out or ("endtoend_%s.npz"
                            % args.ckpt_glob.replace("_*.pt", "").replace("*", ""))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("%d checkpoints matching %s | device %s\n"
          % (len(paths), args.ckpt_glob, device))

    cache, seen, channels = {}, {}, set()
    for i, path in enumerate(paths, 1):
        model, ck = load_model(path, device)
        drop = ck.get("drop_channels")
        held = ck.get("val_patients") or ck.get("held_out") or []
        channels.add(ck["model_config"]["num_channels"])

        for patient in held:
            if patient in seen:
                # Two checkpoints claiming the same held-out patient would
                # mean one of them trained on it. Refuse rather than quietly
                # overwrite, which would put a seen patient in the cache.
                raise SystemExit(
                    "%s is held out by both %s and %s; the cache would not be "
                    "held-out any more" % (patient, seen[patient], path.name))
            seen[patient] = path.name
            cache[patient] = {"blocks": blocks_for(patient, model, drop, device),
                              "model": "base", "checkpoint": path.name}

        n_win = sum(len(b[0]) for p in held for b in cache[p]["blocks"])
        n_sz = sum(int(b[1].sum()) for p in held for b in cache[p]["blocks"])
        print("[%2d/%d] %-34s -> %-18s %7d windows, %5d seizure"
              % (i, len(paths), path.name, ", ".join(held), n_win, n_sz))

        del model
        torch.cuda.empty_cache()

    if len(channels) > 1:
        raise SystemExit("checkpoints disagree on channel count: %s" % channels)

    total = sum(len(b[0]) for p in cache for b in cache[p]["blocks"])
    seizure = sum(int(b[1].sum()) for p in cache for b in cache[p]["blocks"])
    print("\n%d patients, %d windows (%.1f h), %d seizure windows (%.2f%%)"
          % (len(cache), total, total * 2 / 3600, seizure, 100 * seizure / total))
    print("every patient predicted by a model that never trained on it")

    np.savez_compressed(out_path, data=np.array(cache, dtype=object))
    print("saved %s" % out_path)


if __name__ == "__main__":
    main()
