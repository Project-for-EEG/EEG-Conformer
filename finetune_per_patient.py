"""
Per-patient fine-tuning, evaluated with a strict chronological split.

Patient-independent detection has plateaued: seven interventions left AUC at
~0.79, and the failure is not uniform -- 14 of 24 patients already reach 100%
event sensitivity while seven patients holding half the seizures are near-total
failures. That pattern says the remaining gap is patient-specific.

Protocol, per patient:

  1. Start from the fold checkpoint that did NOT train on this patient, so the
     starting model has never seen them.
  2. Order their recordings chronologically and split just after their Nth
     seizure. Everything before is adaptation data; everything after is test.
     Nothing from the test period is ever used -- not for training, not for
     choosing a threshold.
  3. Fine-tune briefly on the adaptation window.
  4. Pick the threshold from a slice of adaptation background HELD OUT from
     fine-tuning, so it is not chosen on data the model just fitted.
  5. Score the base model and the fine-tuned model on the SAME test period with
     the SAME threshold procedure, so fine-tuning is the only difference.

Patients with fewer than N+1 seizures are skipped -- taking the adaptation
window would leave nothing to detect in the test period.

Usage:
    python finetune_per_patient.py --seizures-for-adapt 1 --epochs 8
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from config import ModelConfig
from model import create_model
from evaluate_temporal import find_runs, match_channels, predict_file
from calibrate_per_patient import score_blocks, pick_threshold
from train_memory_efficient import FocalLoss

PREP = Path("preprocessed_data")
CKPT = Path("checkpoints")


def patient_files(patient):
    return sorted((PREP / patient).glob("*.npz"))


def find_split(files, n_seizures):
    """(file_index, window_index) just after the Nth seizure event ends.

    Returns None if the patient has no seizure left after that point, since the
    test period would then contain nothing to detect.
    """
    seen = 0
    for fi, f in enumerate(files):
        with np.load(f) as d:
            lab = d["labels"]
        for lo, hi in find_runs(lab.astype(bool)):
            seen += 1
            if seen == n_seizures:
                remaining = int(lab[hi:].sum())
                for g in files[fi + 1:]:
                    with np.load(g) as d2:
                        remaining += int(d2["labels"].sum())
                return (fi, hi) if remaining > 0 else None
    return None


def adapt_background_mask(files, split, holdout_frac, rng):
    """Per-file boolean masks marking adaptation background reserved for
    threshold selection.

    The threshold cannot be chosen on background the model fine-tuned on: it
    fits that background, so probabilities there are unrepresentatively low and
    the resulting threshold is far too permissive on the test period. Reserving
    a slice keeps calibration honest while still never touching test data.
    """
    fi, wi = split
    masks = []
    for i, f in enumerate(files[:fi + 1]):
        with np.load(f) as d:
            lab = d["labels"][:wi] if i == fi else d["labels"]
        held = np.zeros(len(lab), dtype=bool)
        neg = np.where(lab == 0)[0]
        if len(neg):
            pick = rng.choice(neg, size=int(len(neg) * holdout_frac),
                              replace=False)
            held[pick] = True
        masks.append(held)
    return masks


def load_adapt(files, split, masks, max_background=4000, model=None):
    """Fine-tuning set: every seizure, plus background NOT reserved for
    calibration."""
    fi, wi = split
    per_file_bg = max(1, max_background // (fi + 1))
    X, y = [], []
    for i, f in enumerate(files[:fi + 1]):
        with np.load(f) as d:
            seg, lab = d["segments"], d["labels"]
            if model is not None:
                seg = match_channels(seg, model)
            if i == fi:
                seg, lab = seg[:wi], lab[:wi]
            pos = np.where(lab == 1)[0]
            neg = np.where((lab == 0) & ~masks[i])[0]
            keep = np.random.choice(neg, size=min(len(neg), per_file_bg),
                                    replace=False)
            idx = np.sort(np.concatenate([pos, keep]))
            X.append(seg[idx])
            y.append(lab[idx])
    return np.concatenate(X), np.concatenate(y)


def test_blocks(model, device, files, split):
    """Contiguous (probs, labels) blocks strictly after the split."""
    fi, wi = split
    out = []
    for i, f in enumerate(files):
        if i < fi:
            continue
        probs, lab = predict_file(model, device, f)
        if i == fi:
            probs, lab = probs[wi:], lab[wi:]
        if len(probs) >= 10:
            out.append((probs, lab))
    return out


def adapt_background_probs(model, device, files, split, masks, min_sample=200):
    """Probabilities on the reserved adaptation background.

    Falls back to all adaptation background when the reserved slice is too small
    to estimate a tail percentile from -- a biased threshold beats a threshold
    fitted to noise.
    """
    fi, wi = split
    reserved, everything = [], []
    for i, f in enumerate(files[:fi + 1]):
        probs, lab = predict_file(model, device, f)
        if i == fi:
            probs, lab = probs[:wi], lab[:wi]
        bg = lab == 0
        reserved.append(probs[masks[i] & bg])
        everything.append(probs[bg])
    held = np.concatenate(reserved)
    return held if len(held) >= min_sample else np.concatenate(everything)


def finetune(model, X, y, device, epochs, lr):
    counts = np.bincount(y, minlength=2)
    if counts.min() == 0:
        return model
    ds = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).long())
    weights = 1.0 / counts[y]
    loader = DataLoader(
        ds, batch_size=32,
        sampler=WeightedRandomSampler(weights, len(weights), replacement=True),
        num_workers=0)
    criterion = FocalLoss(alpha=0.25, gamma=2.0, label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seizures-for-adapt", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--target-fa", type=float, default=2.0)
    ap.add_argument("--smooth", type=int, default=5)
    ap.add_argument("--min-consec", type=int, default=3)
    ap.add_argument("--calib-holdout", type=float, default=0.3,
                    help="fraction of adaptation background reserved for "
                         "threshold selection and excluded from fine-tuning")
    ap.add_argument("--cache", type=str, default="finetune_probs.npz")
    args = ap.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)
    rng = np.random.default_rng(42)
    cache = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    K, M = args.smooth, args.min_consec

    owner = {}
    for cp in sorted(CKPT.glob("fold*.pt")):
        ck = torch.load(cp, map_location="cpu", weights_only=False)
        for p in ck["val_patients"]:
            owner[p] = cp

    print(f"adapt on first {args.seizures_for_adapt} seizure(s) | "
          f"{args.epochs} epochs | lr {args.lr} | "
          f"threshold from adaptation background at {args.target_fa} FA/hr")
    print()
    header = ("patient", "events", "base", "tuned", "baseFA", "tunedFA")
    print(f"{header[0]:8s} {header[1]:>6s} {header[2]:>12s} {header[3]:>12s} "
          f"{header[4]:>8s} {header[5]:>9s}")

    agg = dict(base_hit=0, tuned_hit=0, events=0, base_fa=0.0,
               tuned_fa=0.0, hours=0.0)
    skipped = []

    for patient in sorted(owner):
        files = patient_files(patient)
        split = find_split(files, args.seizures_for_adapt)
        if split is None:
            skipped.append(patient)
            continue

        ck = torch.load(owner[patient], map_location=device, weights_only=False)

        def fresh():
            m = create_model(ModelConfig(**ck["model_config"])).to(device)
            m.load_state_dict(ck["model_state_dict"])
            m.eval()
            return m

        masks = adapt_background_mask(files, split, args.calib_holdout, rng)

        base = fresh()
        blocks = test_blocks(base, device, files, split)
        if not blocks:
            skipped.append(patient)
            del base
            continue
        t_base = pick_threshold(
            adapt_background_probs(base, device, files, split, masks),
            args.target_fa, K, M)
        s_base = score_blocks(blocks, t_base, K, M)
        del base

        tuned = fresh()
        Xa, ya = load_adapt(files, split, masks)
        tuned = finetune(tuned, Xa, ya, device, args.epochs, args.lr)
        blocks_t = test_blocks(tuned, device, files, split)
        t_tuned = pick_threshold(
            adapt_background_probs(tuned, device, files, split, masks),
            args.target_fa, K, M)
        s_tuned = score_blocks(blocks_t, t_tuned, K, M)

        # Cache both models' test-period outputs so base and tuned can later be
        # compared at matched false-alarm rates, without re-running any of this.
        cache[patient] = {"base": blocks, "tuned": blocks_t,
                          "t_base": t_base, "t_tuned": t_tuned}
        del tuned, Xa, ya
        torch.cuda.empty_cache()

        agg["events"] += s_base["events"]
        agg["base_hit"] += s_base["caught"]
        agg["tuned_hit"] += s_tuned["caught"]
        agg["base_fa"] += s_base["fa_hr"] * s_base["hours"]
        agg["tuned_fa"] += s_tuned["fa_hr"] * s_tuned["hours"]
        agg["hours"] += s_base["hours"]

        base_txt = f"{s_base['caught']}/{s_base['events']} {s_base['sens']:.2f}"
        tuned_txt = f"{s_tuned['caught']}/{s_tuned['events']} {s_tuned['sens']:.2f}"
        arrow = "  <-- up" if s_tuned["sens"] > s_base["sens"] + 1e-9 else ""
        print(f"{patient:8s} {s_base['events']:6d} {base_txt:>12s} "
              f"{tuned_txt:>12s} {s_base['fa_hr']:8.2f} "
              f"{s_tuned['fa_hr']:9.2f}{arrow}")

    np.savez_compressed(args.cache, data=np.array(cache, dtype=object))

    e = agg["events"]
    print()
    print(f"skipped (too few seizures to split): "
          f"{', '.join(skipped) if skipped else 'none'}")
    print(f"tested on {agg['hours']:.0f} h of held-out recording, {e} events")
    print(f"  base  : {agg['base_hit']}/{e} = {agg['base_hit']/e:.3f} sensitivity "
          f"at {agg['base_fa']/agg['hours']:.2f} FA/hr")
    print(f"  tuned : {agg['tuned_hit']}/{e} = {agg['tuned_hit']/e:.3f} sensitivity "
          f"at {agg['tuned_fa']/agg['hours']:.2f} FA/hr")
    print(f"  gain  : {(agg['tuned_hit'] - agg['base_hit'])/e:+.3f}")
    print()
    print(f"cached per-patient test probabilities -> {args.cache}")
    print("NOTE: the two arms above sit at different FA rates; run "
          "compare_finetune.py for the matched-FA comparison.")


if __name__ == "__main__":
    main()
