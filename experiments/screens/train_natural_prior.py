"""
Train on the TRUE class distribution, using the memmap.

Every model so far was trained at roughly 11% seizure -- `--max-segments 2000`
keeps all of a patient's seizures and caps the background -- against 1.4% in
reality. The consequence shows up as overconfidence on background: for more than
half the patients, 1% of ordinary EEG scores above 0.99, which puts a floor of
~8-10 false alarms/hour under any threshold. No post-processing can fix that,
because thresholding is rank-based and immune to rescaling.

This trains at the real 1.4%, which is only possible via the memmap -- the full
dataset is 35 GB as float32 and will not fit in 16 GB of RAM.

Two things follow from the natural prior and are deliberate:

  * NO weighted sampling, NO SMOTE, and plain cross-entropy rather than focal
    loss. Every one of those re-inflates the effective prior, which is the thing
    being tested.
  * A large batch. At 1.4% positive, a batch of 32 contains 0.45 seizures on
    average and most batches contain none; 256 gives ~3.5 and a usable gradient.

Patient-level folds match the other trainers, so results are comparable.

Usage:
    python train_natural_prior.py --folds 3 --epochs 8
"""
import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from config import get_config
from model import create_model

MM = Path("memmap")
SHAPE_TAIL = (23, 1024)


class MemmapEEG(Dataset):
    """Random access into the packed float16 array.

    The memmap is opened lazily per worker: an open np.memmap cannot be pickled
    across process boundaries, so sharing one would break num_workers > 0.
    """

    def __init__(self, indices, total, labels):
        self.indices = indices
        self.total = total
        self.labels = labels
        self._mm = None

    def _array(self):
        if self._mm is None:
            self._mm = np.memmap(MM / "segments.f16", dtype=np.float16,
                                 mode="r", shape=(self.total,) + SHAPE_TAIL)
        return self._mm

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        gi = self.indices[i]
        x = np.asarray(self._array()[gi], dtype=np.float32)
        return torch.from_numpy(x), torch.tensor(int(self.labels[gi]))


def evaluate(model, loader, device):
    model.eval()
    probs, ys = [], []
    with torch.no_grad():
        for xb, yb in loader:
            out = model(xb.to(device))
            probs.append(torch.softmax(out, 1)[:, 1].cpu().numpy())
            ys.append(yb.numpy())
    return np.concatenate(probs), np.concatenate(ys)


def background_tail(probs, labels):
    """Percentiles of the model's output on background -- the quantity that
    actually determines the false-alarm floor."""
    bg = probs[labels == 0]
    return {f"p{q}": float(np.percentile(bg, q)) for q in (99, 99.9, 99.99)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-train", type=int, default=0,
                    help="optional cap on training segments per fold, "
                         "subsampled WITHOUT changing the class ratio")
    ap.add_argument("--prior", type=float, default=0.0,
                    help="target seizure fraction in the training set. Keeps "
                         "every seizure and varies only the amount of "
                         "background, so this is the one variable that moves. "
                         "0 = use the data as-is (natural 1.4%%).")
    ap.add_argument("--max-val", type=int, default=40000,
                    help="cap validation segments; all seizures kept and background subsampled. Full validation is ~105k segments = 4.9 GB of memmap reads per fold, which stalls without page-cache headroom.")
    ap.add_argument("--tag", type=str, default="natural",
                    help="checkpoint/result prefix, so arms do not overwrite "
                         "each other")
    args = ap.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    meta = np.load(MM / "meta.npz", allow_pickle=True)
    labels = meta["labels"]
    pidx = meta["patient_idx"]
    patients = list(meta["patients"])
    total = int(meta["total"])

    print(f"{total:,} segments, {int(labels.sum()):,} seizure "
          f"({100*labels.mean():.3f}%) -- the true distribution")
    print(f"batch {args.batch_size} => ~{args.batch_size*labels.mean():.1f} "
          f"seizures per batch\n")

    order = np.arange(len(patients))
    np.random.default_rng(42).shuffle(order)
    folds = [list(f) for f in np.array_split(order, args.folds)]

    # chb01 and chb21 are the same child recorded 1.5 years apart (PhysioNet
    # dataset description) -- keep them in one fold or the split leaks.
    for a, b in [("CHB01", "CHB21")]:
        if a not in patients or b not in patients:
            continue
        ia, ib = patients.index(a), patients.index(b)
        fa = next(i for i, f in enumerate(folds) if ia in f)
        fb = next(i for i, f in enumerate(folds) if ib in f)
        if fa != fb:
            partner = next(p for p in folds[fa] if p != ia)
            folds[fa][folds[fa].index(partner)] = ib
            folds[fb][folds[fb].index(ib)] = partner
            print(f"note: {b} moved into {a}'s fold (same subject); "
                  f"{patients[partner]} swapped out")

    config = get_config()
    results = []

    for fi, val_p in enumerate(folds):
        val_mask = np.isin(pidx, val_p)
        tr_idx = np.flatnonzero(~val_mask)
        va_idx = np.flatnonzero(val_mask)

        if args.prior > 0:
            # Keep every seizure; choose how much background sits alongside.
            # Prior = seizures/total, so it cannot be changed without moving
            # one of them -- fixing the seizures makes background the only
            # variable, which is the comparison worth making.
            rng = np.random.default_rng(42)
            pos = tr_idx[labels[tr_idx] == 1]
            neg = tr_idx[labels[tr_idx] == 0]
            n_neg = int(len(pos) * (1 - args.prior) / args.prior)
            n_neg = min(n_neg, len(neg))
            tr_idx = np.sort(np.concatenate(
                [pos, rng.choice(neg, n_neg, replace=False)]))

        if args.max_train and len(tr_idx) > args.max_train:
            # Subsample each class by the same factor, preserving the ratio.
            rng = np.random.default_rng(42)
            keep = args.max_train / len(tr_idx)
            pos = tr_idx[labels[tr_idx] == 1]
            neg = tr_idx[labels[tr_idx] == 0]
            tr_idx = np.sort(np.concatenate([
                rng.choice(pos, int(len(pos) * keep), replace=False),
                rng.choice(neg, int(len(neg) * keep), replace=False)]))

        if args.max_val and len(va_idx) > args.max_val:
            # Full validation is ~105k segments = 4.9 GB of memmap reads
            # per fold, which stalled outright with no page-cache headroom.
            # Subsampling keeps every seizure and is unbiased for AUC and
            # the background percentiles -- only slightly noisier.
            rng = np.random.default_rng(7)
            pos = va_idx[labels[va_idx] == 1]
            neg = va_idx[labels[va_idx] == 0]
            n_neg = max(0, args.max_val - len(pos))
            va_idx = np.sort(np.concatenate(
                [pos, rng.choice(neg, min(n_neg, len(neg)), replace=False)]))

        print(f"{'='*60}\nFOLD {fi+1}/{args.folds}")
        print(f"val patients: {', '.join(patients[i] for i in val_p)}")
        print(f"train {len(tr_idx):,} segments "
              f"({100*labels[tr_idx].mean():.3f}% seizure) | "
              f"val {len(va_idx):,} ({100*labels[va_idx].mean():.3f}%)")

        train_loader = DataLoader(
            MemmapEEG(tr_idx, total, labels), batch_size=args.batch_size,
            shuffle=True, num_workers=args.workers, pin_memory=True,
            persistent_workers=args.workers > 0, drop_last=True)
        val_loader = DataLoader(
            MemmapEEG(va_idx, total, labels), batch_size=512,
            shuffle=False, num_workers=0, pin_memory=False)

        model = create_model(config.model).to(device)
        # Plain CE on purpose: focal loss, class weights and resampling all
        # re-inflate the prior this experiment exists to remove.
        criterion = nn.CrossEntropyLoss()
        opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

        for ep in range(args.epochs):
            model.train()
            run = 0.0
            pbar = tqdm(train_loader, desc=f"fold {fi+1} epoch {ep+1}/{args.epochs}",
                        leave=False)
            for xb, yb in pbar:
                xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                opt.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                run += loss.item()
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            sched.step()

        probs, ys = evaluate(model, val_loader, device)
        auc = roc_auc_score(ys, probs) if len(np.unique(ys)) > 1 else float("nan")
        tail = background_tail(probs, ys)
        results.append({"fold": fi + 1, "auc": float(auc), **tail,
                        "val_patients": [patients[i] for i in val_p]})

        print(f"\nfold {fi+1}: AUC {auc:.4f}")
        print(f"  background p99   {tail['p99']:.4f}   "
              f"(previous models: 0.94-0.99 -- the false-alarm floor)")
        print(f"  background p99.9 {tail['p99.9']:.4f}")
        print(f"  background p99.99 {tail['p99.99']:.4f}")

        ckpt = Path("checkpoints") / f"{args.tag}_fold{fi+1}.pt"
        torch.save({"fold": fi + 1, "model_state_dict": model.state_dict(),
                    "model_config": asdict(config.model),
                    "val_patients": [patients[i] for i in val_p],
                    "auc": float(auc), "background_tail": tail,
                    "trained_on": "natural prior"}, ckpt)
        print(f"  saved {ckpt}")
        del model
        torch.cuda.empty_cache()

    aucs = [r["auc"] for r in results]
    p99 = [r["p99"] for r in results]
    print(f"\n{'='*60}")
    print(f"AUC            {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}")
    print(f"background p99 {np.mean(p99):.4f} +/- {np.std(p99):.4f}")
    print("\nA LOWER background p99 is the point of this run: it is what sets "
          "the floor\non achievable false alarms, and AUC cannot see it.")

    out = Path("cv_results") / f"{args.tag}_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
