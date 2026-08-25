"""
Leave-one-patient-out cross-validation.

The 3-fold result carries +/- 8 points of noise: reshuffling which patients sit
in which fold swung CHB06 from 0.33 to 1.00 and CHB14 from 0.43 to 0.14 with no
other change. Anything smaller than that spread is unmeasurable, which is more
than the entire expected gain from swapping in a pretrained backbone.

LOPO fixes both problems. Every patient is held out exactly once, so the result
is a per-patient distribution rather than an average over three arbitrary
groupings, and the error bars come from 23 estimates instead of 3.

Cases chb01 and chb21 are the SAME child recorded 18 months apart, so they form
one fold. That gives 23 folds over 24 cases.

Hyperparameters match train_memory_efficient.py exactly (batch 32, 15 epochs,
--max-segments 2000, no resampling) so LOPO numbers are directly comparable to
the 3-fold ones.

Resumable: a fold whose checkpoint already exists is skipped, so an interrupted
run continues where it stopped.

Usage:
    python train_lopo.py                      # all 23 folds
    python train_lopo.py --only CHB06 CHB14   # just these
"""
import argparse
import gc
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from config import get_config
from model import create_model
from train_memory_efficient import load_patient_data, train_one_fold, set_seed

PREP = Path("preprocessed_data")
CKPT = Path("checkpoints/lopo")

# chb01 and chb21 are one patient (PhysioNet dataset description). Holding them
# out separately would put the same child on both sides of a "patient
# independent" split.
SAME_SUBJECT = [("CHB01", "CHB21")]


def subject_groups():
    """Patient cases grouped into unique subjects, in stable order."""
    cases = sorted(d.name for d in PREP.iterdir() if d.is_dir())
    merged, seen = [], set()
    for a, b in SAME_SUBJECT:
        if a in cases and b in cases:
            merged.append([a, b])
            seen.update((a, b))
    for c in cases:
        if c not in seen:
            merged.append([c])
    return sorted(merged, key=lambda g: g[0])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--max-segments", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=None,
                    help="default: whatever config.training specifies (32)")
    ap.add_argument("--only", nargs="*", default=None,
                    help="run only folds whose held-out group contains these")
    ap.add_argument("--tag", type=str, default="lopo")
    args = ap.parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = get_config()
    if args.batch_size:
        config.training.batch_size = args.batch_size
    CKPT.mkdir(parents=True, exist_ok=True)

    groups = subject_groups()
    print(f"{len(groups)} folds over {sum(len(g) for g in groups)} cases")
    for g in groups:
        if len(g) > 1:
            print(f"  {' + '.join(g)} held out together (same subject)")
    print(f"epochs {args.epochs} | max-segments {args.max_segments} | "
          f"batch {config.training.batch_size} | device {device}\n")

    results = []
    for i, held in enumerate(groups):
        name = "_".join(held)
        if args.only and not any(p in args.only for p in held):
            continue

        out = CKPT / f"{args.tag}_{name}.pt"
        if out.exists():
            ck = torch.load(out, map_location="cpu", weights_only=False)
            results.append(ck["metrics"] | {"held_out": held})
            print(f"[{i+1}/{len(groups)}] {name}: cached "
                  f"(AUC {ck['metrics'].get('auc', float('nan')):.4f})")
            continue

        train_cases = [c for g in groups if g is not held for c in g]
        print(f"[{i+1}/{len(groups)}] holding out {', '.join(held)} "
              f"| training on {len(train_cases)} cases")

        Xtr, ytr = [], []
        for c in train_cases:
            X, y = load_patient_data(c, PREP, args.max_segments)
            if X is not None:
                Xtr.append(X)
                ytr.append(y)
        X_train = np.concatenate(Xtr, axis=0)
        y_train = np.concatenate(ytr, axis=0)
        del Xtr, ytr

        Xva, yva = [], []
        for c in held:
            X, y = load_patient_data(c, PREP, args.max_segments)
            if X is not None:
                Xva.append(X)
                yva.append(y)
        X_val = np.concatenate(Xva, axis=0)
        y_val = np.concatenate(yva, axis=0)
        del Xva, yva

        print(f"    train {len(y_train):,} seg / {int(y_train.sum())} seizure | "
              f"val {len(y_val):,} / {int(y_val.sum())}")

        # no resampling and no SMOTE: matches --balance none in the 3-fold run
        train_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_train).float(),
                          torch.from_numpy(y_train).long()),
            batch_size=config.training.batch_size, shuffle=True,
            num_workers=0, pin_memory=False, drop_last=True)
        del X_train, y_train
        gc.collect()

        val_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_val).float(),
                          torch.from_numpy(y_val).long()),
            batch_size=config.training.batch_size, shuffle=False,
            num_workers=0, pin_memory=False)
        n_val_seizure = int(y_val.sum())
        del X_val, y_val
        gc.collect()

        model = create_model(config.model).to(device)
        metrics, _, _ = train_one_fold(model, train_loader, val_loader,
                                       config, device, i, args.epochs)

        torch.save({"held_out": held, "model_state_dict": model.state_dict(),
                    "model_config": asdict(config.model),
                    "val_patients": held, "metrics": metrics,
                    "max_segments": args.max_segments,
                    "n_val_seizure": n_val_seizure}, out)

        results.append(metrics | {"held_out": held})
        print(f"    AUC {metrics.get('auc', float('nan')):.4f} | "
              f"recall {metrics['recall']:.3f} | saved {out.name}\n")

        del model, train_loader, val_loader
        gc.collect()
        torch.cuda.empty_cache()

    if not results:
        return

    aucs = [r["auc"] for r in results if "auc" in r and not np.isnan(r["auc"])]
    print(f"\n{'='*60}")
    print(f"LOPO over {len(results)} folds")
    print(f"{'='*60}")
    print(f"  AUC  mean {np.mean(aucs):.4f}  sd {np.std(aucs):.4f}  "
          f"median {np.median(aucs):.4f}")
    print(f"       range {np.min(aucs):.4f} to {np.max(aucs):.4f}")
    print(f"\n  3-fold for comparison: 0.7795 +/- 0.0168 "
          f"(sd over 3 groupings, not 23 patients)")

    print(f"\n  {'held out':<14s} {'AUC':>7s} {'recall':>8s}")
    for r in sorted(results, key=lambda r: r.get("auc", 0)):
        print(f"  {'+'.join(r['held_out']):<14s} {r.get('auc', float('nan')):7.4f} "
              f"{r['recall']:8.3f}")

    Path("cv_results").mkdir(exist_ok=True)
    f = Path("cv_results") / f"{args.tag}_{datetime.now():%Y%m%d_%H%M%S}.json"
    f.write_text(json.dumps(
        [{k: (v if not isinstance(v, (np.floating, np.integer)) else float(v))
          for k, v in r.items()} for r in results], indent=2))
    print(f"\nsaved {f}")


if __name__ == "__main__":
    main()
