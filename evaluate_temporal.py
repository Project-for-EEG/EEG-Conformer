"""
Evaluate fold checkpoints on FULL recordings, with temporal smoothing.

Two things this does that the cross-validation run cannot:

1. It runs on whole, contiguous recordings. `load_patient_data` caps each
   patient at --max-segments and keeps every seizure, so the CV validation set
   sits at ~10% seizure against 1.1% in reality -- precision and false-alarm
   figures from it are optimistic by roughly 8x. Here nothing is subsampled.

2. It smooths over time. A seizure spans 15-100 s, i.e. 8-50 consecutive 4 s
   windows, but the model classifies each window independently, so a single
   stray window counts as a detection. Averaging probabilities over a short
   neighbourhood and requiring a minimum run length removes those blips.

It also reports event-level numbers. Segment-level F1 treats every 4 s window as
an independent trial, which is not how seizure detection is judged -- what
matters is whether each seizure was caught at all, and how often the detector
cries wolf per hour.

Usage:
    python evaluate_temporal.py                     # all folds found
    python evaluate_temporal.py --fold 1
    python evaluate_temporal.py --smooth 5 --min-consec 3
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from config import ModelConfig
from model import create_model

PREP_DIR = Path("preprocessed_data")
CKPT_DIR = Path("checkpoints")

SAMPLING_RATE = 256
STEP = 512                       # window stride in samples
SECONDS_PER_WINDOW = STEP / SAMPLING_RATE   # 2.0 s of new data per window


def find_runs(mask):
    """Maximal runs of True in a 1-D boolean array, as (start, end) exclusive."""
    if not mask.any():
        return []
    padded = np.concatenate(([False], mask, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(edges[::2], edges[1::2]))


def smooth(probs, k):
    """Moving average over k windows, edges handled by shrinking the window."""
    # np.convolve(..., mode="same") returns max(len(a), len(v)) -- for a block
    # shorter than the kernel that is LONGER than the input, which silently
    # desynchronises probabilities from their labels.
    if k <= 1 or len(probs) < k:
        return probs
    kernel = np.ones(k, dtype=np.float64)
    num = np.convolve(probs, kernel, mode="same")
    den = np.convolve(np.ones_like(probs, dtype=np.float64), kernel, mode="same")
    return num / den


def model_channels(model):
    """How many EEG channels this model expects.

    Read off the spatial convolution rather than passed in as a flag. The
    merged-cohort models are 20-channel while preprocessed_data/ still holds
    23-channel CHB-MIT recordings, and a flag that has to be remembered at
    every call site is how the wrong number quietly gets used.
    """
    for name, param in model.named_parameters():
        if name.endswith("spatial_conv.0.weight"):
            return param.shape[2]
    return None


def match_channels(X, model):
    """Slice a 23-channel array down to what the model wants.

    Only the FT9/FT10 derivations at 19, 20, 21 are ever dropped -- those are
    the three Siena cannot reconstruct, and the 20-channel montage is defined
    as the other twenty in their original order.
    """
    want = model_channels(model)
    if want is None or X.shape[1] == want:
        return X
    if X.shape[1] == 23 and want == 20:
        keep = [c for c in range(23) if c not in (19, 20, 21)]
        return X[:, keep, :]
    raise ValueError("cannot map %d channels onto a %d-channel model"
                     % (X.shape[1], want))


def predict_file(model, device, npz_path, batch=256):
    """Seizure probability for every window of one recording, in time order."""
    with np.load(npz_path) as data:
        X = match_channels(data["segments"], model)
        y = data["labels"]
        out = []
        with torch.no_grad():
            for i in range(0, len(X), batch):
                b = torch.from_numpy(X[i:i + batch]).float().to(device)
                out.append(torch.softmax(model(b), dim=1)[:, 1].cpu().numpy())
    return np.concatenate(out), y


def score(per_file, threshold, smooth_k, min_consec):
    """Segment- and event-level scores at one operating point.

    per_file is a list of (probs, labels) kept separate so that runs never span
    a recording boundary.
    """
    tp = fp = fn = tn = 0
    events_total = events_caught = 0
    false_alarm_runs = 0
    windows = 0

    for probs, labels in per_file:
        pred = smooth(probs, smooth_k) >= threshold

        # Drop predicted runs shorter than min_consec windows.
        if min_consec > 1:
            cleaned = np.zeros_like(pred)
            for lo, hi in find_runs(pred):
                if hi - lo >= min_consec:
                    cleaned[lo:hi] = True
            pred = cleaned

        truth = labels.astype(bool)
        tp += int(np.sum(pred & truth))
        fp += int(np.sum(pred & ~truth))
        fn += int(np.sum(~pred & truth))
        tn += int(np.sum(~pred & ~truth))
        windows += len(pred)

        true_events = find_runs(truth)
        events_total += len(true_events)
        events_caught += sum(1 for lo, hi in true_events if pred[lo:hi].any())

        for lo, hi in find_runs(pred):
            if not truth[lo:hi].any():
                false_alarm_runs += 1

    hours = windows * SECONDS_PER_WINDOW / 3600
    return {
        "threshold": threshold,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "event_sensitivity": events_caught / events_total if events_total else 0.0,
        "events": f"{events_caught}/{events_total}",
        "fa_per_hour": false_alarm_runs / hours if hours else 0.0,
        # Share of all time flagged. Without this a saturated detector looks
        # good: predicting positive everywhere merges into one run that overlaps
        # a real seizure, so it books ~0 false alarms while being useless.
        # Anything much above the true seizure rate is a red flag.
        "flagged_pct": 100 * (tp + fp) / windows if windows else 0.0,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "hours": hours,
    }


def evaluate_fold(ckpt_path, device, smooth_k, min_consec):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = create_model(ModelConfig(**ckpt["model_config"])).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    val_patients = ckpt["val_patients"]
    print(f"\n{'='*74}")
    print(f"{ckpt_path.name}  --  held-out patients: {', '.join(val_patients)}")
    print(f"{'='*74}")

    per_file = []
    for patient in val_patients:
        for npz in sorted((PREP_DIR / patient).glob("*.npz")):
            probs, labels = predict_file(model, device, npz)
            per_file.append((probs, labels))

    total = sum(len(p) for p, _ in per_file)
    positives = sum(int(l.sum()) for _, l in per_file)
    events = sum(len(find_runs(l.astype(bool))) for _, l in per_file)
    print(f"{total:,} windows, {positives} seizure windows "
          f"({100*positives/total:.3f}%), {events} seizure events, "
          f"{total*SECONDS_PER_WINDOW/3600:.1f} h of EEG\n")

    grid = [0.10, 0.25, 0.50, 0.75, 0.90]
    for label, k, m in (("raw (no smoothing)", 1, 1),
                        (f"smoothed (k={smooth_k}, min_consec={min_consec})",
                         smooth_k, min_consec)):
        print(f"  {label}")
        print(f"  {'thresh':>7s} {'prec':>7s} {'recall':>7s} "
              f"{'events':>9s} {'evt sens':>9s} {'FA/hr':>8s} {'%flagged':>9s}")
        for t in grid:
            s = score(per_file, t, k, m)
            print(f"  {s['threshold']:7.2f} {s['precision']:7.3f} {s['recall']:7.3f} "
                  f"{s['events']:>9s} {s['event_sensitivity']:9.3f} "
                  f"{s['fa_per_hour']:8.2f} {s['flagged_pct']:8.1f}%")
        print()

    return per_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, default=None,
                        help="Evaluate one fold; default is every fold found")
    parser.add_argument("--smooth", type=int, default=5,
                        help="Moving-average width in windows (5 = 10 s)")
    parser.add_argument("--min-consec", type=int, default=3,
                        help="Discard predicted runs shorter than this (3 = 6 s)")
    args = parser.parse_args()

    ckpts = ([CKPT_DIR / f"fold{args.fold}.pt"] if args.fold
             else sorted(CKPT_DIR.glob("fold*.pt")))
    ckpts = [c for c in ckpts if c.exists()]

    if not ckpts:
        print(f"No fold checkpoints in {CKPT_DIR}/.")
        print("Run train_memory_efficient.py first -- it now saves fold*.pt.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    for ckpt in ckpts:
        evaluate_fold(ckpt, device, args.smooth, args.min_consec)


if __name__ == "__main__":
    main()
