"""
Score the leave-one-patient-out run with no test-patient data anywhere.

The headline uses the held-out patient's own earlier recording twice. The alarm
threshold is picked from that patient's background (evaluate_end_to_end.py:189
calling pick_threshold on adapt_background_probs), and the decision whether to
personalise reads that patient's own first-seizure labels
(base_handles_patient at line 191). Five of 24 cases then fine-tune on that
patient. None of that is training on the test patient, but none of it is
patient-independent either, and the gap between the two is what this measures.

WHAT CHANGES HERE

    threshold      chosen from the TRAINING-fold patients' adaptation
                   background, scored by the same fold model. The held-out
                   patient contributes nothing to it.
    model          always the base fold checkpoint. No fine-tuning, and no
                   base-versus-tuned decision, so no test-patient labels are
                   read at any point.
    test windows   identical to the headline run, so the comparison isolates
                   the calibration and leaves everything else fixed.

WHY THE CACHE COULD NOT BE REUSED

    endtoend_lopo.npz stores, per patient, only that patient's own predictions
    from the model that held them out. A threshold drawn from the training fold
    needs fold model M_P evaluated on P's training patients, which is not on
    disk. Pooling the other patients' cached background instead is not a
    substitute: those probabilities come from 23 different models which are not
    on a common scale -- background medians run from 0.025 to 0.998 -- and doing
    it yields 0.196 sensitivity, which measures the scale mismatch rather than
    the value of calibration.

    So this re-runs inference. It does not retrain: the existing
    checkpoints/lopo/lopo_*.pt are reused unchanged.

ONE THING THIS STILL BORROWS FROM THE TEST PATIENT

    The test period begins after that patient's first seizure, which is where
    the headline run also starts scoring. Finding that boundary reads their
    labels. Keeping it means both runs are scored on exactly the same windows,
    which is the point. It is disclosed rather than hidden, and it affects only
    where scoring starts, never the model or the threshold.

Usage:
    python uncalibrated_lopo.py
    python uncalibrated_lopo.py --target-fa 2.0 --out endtoend_uncalibrated.npz
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch

from config import ModelConfig
from model import create_model
from calibrate_per_patient import pick_threshold
from finetune_per_patient import (patient_files, find_split,
                                  adapt_background_mask,
                                  adapt_background_probs, test_blocks)

CKPT = Path("checkpoints/lopo")


def build_model(ck, device):
    model = create_model(ModelConfig(**ck["model_config"])).to(device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    return model


def training_fold_background(model, device, patients, holdout, n_seizures, seed=42):
    """Pooled adaptation background of every patient in the training fold.

    Each training patient is split at their own first seizure and the reserved
    background slice is taken, exactly as the headline run does for the held-out
    patient. The difference is only whose background it is.
    """
    rng = np.random.default_rng(seed)
    pooled = []
    for q in patients:
        files = patient_files(q)
        if not files:
            continue
        split = find_split(files, n_seizures)
        if split is None:
            continue
        masks = adapt_background_mask(files, split, holdout, rng)
        probs = adapt_background_probs(model, device, files, split, masks)
        if probs is not None and len(probs):
            pooled.append(np.asarray(probs, dtype=np.float32))
    return np.concatenate(pooled) if pooled else None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt-glob", default="lopo_*.pt")
    ap.add_argument("--target-fa", type=float, default=2.0)
    ap.add_argument("--smooth", type=int, default=5)
    ap.add_argument("--min-consec", type=int, default=3)
    ap.add_argument("--calib-holdout", type=float, default=0.3)
    ap.add_argument("--seizures-for-adapt", type=int, default=1)
    ap.add_argument("--out", default="endtoend_uncalibrated.npz")
    args = ap.parse_args()

    paths = sorted(CKPT.glob(args.ckpt_glob))
    if not paths:
        raise SystemExit("no checkpoints match %s in %s" % (args.ckpt_glob, CKPT))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("%d fold checkpoints | target %.1f FA/h | smooth %d | min_consec %d"
          % (len(paths), args.target_fa, args.smooth, args.min_consec))
    print("threshold from TRAINING-fold background only, base model only\n")

    # Every patient any checkpoint holds out. A fold's training set is all of
    # them except its own held-out group.
    owner = {}
    for p in paths:
        ck = torch.load(p, map_location="cpu", weights_only=False)
        for q in ck["val_patients"]:
            owner[q] = p
    everyone = sorted(owner)
    print("cohort: %d cases\n" % len(everyone))

    cache, started = {}, time.time()
    for i, path in enumerate(paths, 1):
        ck = torch.load(path, map_location=device, weights_only=False)
        held = list(ck["val_patients"])
        train_fold = [q for q in everyone if q not in held]
        model = build_model(ck, device)

        bg = training_fold_background(model, device, train_fold,
                                      args.calib_holdout,
                                      args.seizures_for_adapt)
        if bg is None:
            print("[%2d/%d] %-22s no training background, skipped"
                  % (i, len(paths), path.name))
            del model
            continue
        t = pick_threshold(bg, args.target_fa, args.smooth, args.min_consec)

        for q in held:
            files = patient_files(q)
            split = find_split(files, args.seizures_for_adapt)
            if split is None:
                print("      %s has no test period, skipped" % q)
                continue
            blocks = test_blocks(model, device, files, split)
            if not blocks:
                continue
            cache[q] = {"blocks": blocks, "bg": bg[:1], "t": t, "model": "base"}

        print("[%2d/%d] %-22s -> %-14s threshold %.4f from %d training windows "
              "(%.0fs elapsed)"
              % (i, len(paths), path.name, ", ".join(held), t, len(bg),
                 time.time() - started))
        del model, bg
        torch.cuda.empty_cache()

    ts = sorted({round(v["t"], 6) for v in cache.values()})
    print("\n%d cases scored. %d distinct thresholds, %.4f to %.4f"
          % (len(cache), len(ts), min(ts), max(ts)))
    print("every threshold came from other patients' recordings only")
    np.savez_compressed(args.out, data=np.array(cache, dtype=object))
    print("saved %s (%.0f min)" % (args.out, (time.time() - started) / 60))


if __name__ == "__main__":
    main()
