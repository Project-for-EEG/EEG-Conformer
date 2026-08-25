"""
Fine-tune CBraMod on CHB-MIT, paired against the from-scratch EEG-Conformer.

Ten interventions changed the data or the training and left AUC at ~0.78. The
architecture was never one of them. CBraMod (ICLR 2025, 4.9M params, pretrained
on 9,000 h of TUEG) is the cleanest test of that last variable: it takes
arbitrary montages without electrode coordinates or a channel vocabulary, and
its authors already ran it on CHB-MIT's bipolar channels.

The comparison is PAIRED. Same folds, same patients, same seed, same windows,
same labels -- only the model differs. That matters: the +/- 8 point spread we
measured came from reshuffling patients between folds, and holding the folds
fixed removes it. Same-fold, different-training spread was ~0.02 in the
class-prior runs, against an expected gain of ~0.07.

CHB03 and CHB05 are excluded. Their local data predates raw_data/ and does not
match the PhysioNet recordings (different labels, and for CHB05 a different
window count), so they cannot be paired. 22 patients remain.

Usage:
    python train_cbramod.py --folds 3 --epochs 15 --max-segments 2000
    python train_cbramod.py --benchmark      # time one epoch, then stop
"""
import argparse
import gc
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)
from tqdm import tqdm

from cbramod_src.cbramod import CBraMod
from train_memory_efficient import (load_patient_data, get_patient_list,
                                    FocalLoss, set_seed)

DATA = Path("cbramod_data")
CKPT = Path("checkpoints")
WEIGHTS = Path("cbramod_ckpt/pretrained_weights.pth")

# same-subject pair: chb01 and chb21 are one child recorded 18 months apart
SAME_SUBJECT = [("CHB01", "CHB21")]
# local copies predate raw_data/ and do not match PhysioNet
EXCLUDE_DEFAULT = ["CHB03", "CHB05"]


class CBraModClassifier(nn.Module):
    """Pretrained CBraMod encoder with a small classification head.

    The published head flattens every patch of every channel, which for 23
    channels is ~14.7M parameters -- three times the backbone, on 22 patients.
    Mean-pooling over channels and patches keeps the head at ~26k, which suits
    a dataset this size and keeps the comparison about the pretrained features
    rather than about head capacity.
    """

    def __init__(self, n_classes=2, dropout=0.3, freeze_backbone=False):
        super().__init__()
        self.backbone = CBraMod(in_dim=200, out_dim=200, d_model=200,
                                dim_feedforward=800, seq_len=30,
                                n_layer=12, nhead=8)
        if WEIGHTS.exists():
            sd = torch.load(WEIGHTS, map_location="cpu", weights_only=False)
            if isinstance(sd, dict) and "state_dict" in sd:
                sd = sd["state_dict"]
            missing, unexpected = self.backbone.load_state_dict(sd, strict=False)
            if missing or unexpected:
                raise RuntimeError(f"weight mismatch: {missing[:3]} {unexpected[:3]}")
        else:
            raise FileNotFoundError(f"{WEIGHTS} not found")

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.head = nn.Sequential(
            nn.LayerNorm(200),
            nn.Dropout(dropout),
            nn.Linear(200, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        # x: (batch, channels, patches, 200) -> (batch, channels, patches, 200)
        feats = self.backbone(x)
        pooled = feats.mean(dim=(1, 2))          # over channels and patches
        return self.head(pooled)


def subject_folds(patients, n_folds, seed=42):
    """Patient-level folds, keeping same-subject cases together."""
    order = list(patients)
    rng = np.random.RandomState(seed)
    rng.shuffle(order)
    size = len(order) // n_folds
    folds = [order[i * size:(i + 1) * size if i < n_folds - 1 else len(order)]
             for i in range(n_folds)]
    for a, b in SAME_SUBJECT:
        fa = next((i for i, f in enumerate(folds) if a in f), None)
        fb = next((i for i, f in enumerate(folds) if b in f), None)
        if fa is None or fb is None or fa == fb:
            continue
        partner = next(p for p in folds[fa] if p != a)
        folds[fa][folds[fa].index(partner)] = b
        folds[fb][folds[fb].index(b)] = partner
        print(f"  note: {b} moved into {a}'s fold (same subject); "
              f"{partner} swapped out")
    return folds


def evaluate(model, loader, device):
    model.eval()
    probs, preds, ys = [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            out = model(xb.to(device))
            probs.append(torch.softmax(out, 1)[:, 1].cpu().numpy())
            preds.append(out.argmax(1).cpu().numpy())
            ys.append(yb.numpy())
    return (np.concatenate(probs), np.concatenate(preds), np.concatenate(ys))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--max-segments", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4,
                    help="CBraMod's published fine-tune lr")
    ap.add_argument("--head-lr-mult", type=float, default=10.0,
                    help="head trains faster than the pretrained backbone")
    ap.add_argument("--freeze-backbone", action="store_true")
    ap.add_argument("--exclude", nargs="*", default=EXCLUDE_DEFAULT)
    ap.add_argument("--benchmark", action="store_true",
                    help="time one epoch and exit, before committing to a run")
    ap.add_argument("--tag", type=str, default="cbramod")
    args = ap.parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    patients = [p for p in get_patient_list(DATA) if p not in (args.exclude or [])]
    print(f"CBraMod fine-tune | {len(patients)} patients "
          f"(excluded {', '.join(args.exclude) if args.exclude else 'none'})")
    folds = subject_folds(patients, args.folds)
    for i, f in enumerate(folds, 1):
        print(f"  fold {i}: {', '.join(f)}")
    print()

    results = []
    for fi, val_p in enumerate(folds):
        train_p = [p for j, f in enumerate(folds) if j != fi for p in f]
        print(f"{'='*60}\nFOLD {fi+1}/{len(folds)} | val: {', '.join(val_p)}")

        Xtr, ytr = [], []
        for p in train_p:
            X, y = load_patient_data(p, DATA, args.max_segments)
            if X is not None:
                Xtr.append(X); ytr.append(y)
        X_train = np.concatenate(Xtr); y_train = np.concatenate(ytr)
        del Xtr, ytr

        Xva, yva = [], []
        for p in val_p:
            X, y = load_patient_data(p, DATA, args.max_segments)
            if X is not None:
                Xva.append(X); yva.append(y)
        X_val = np.concatenate(Xva); y_val = np.concatenate(yva)
        del Xva, yva

        print(f"train {len(y_train):,} seg / {int(y_train.sum())} seizure | "
              f"val {len(y_val):,} / {int(y_val.sum())}")

        train_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_train).float(),
                          torch.from_numpy(y_train).long()),
            batch_size=args.batch_size, shuffle=True, num_workers=0,
            pin_memory=False, drop_last=True)
        del X_train, y_train; gc.collect()

        val_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_val).float(),
                          torch.from_numpy(y_val).long()),
            batch_size=args.batch_size, shuffle=False, num_workers=0,
            pin_memory=False)
        del X_val, y_val; gc.collect()

        model = CBraModClassifier(freeze_backbone=args.freeze_backbone).to(device)
        n_par = sum(p.numel() for p in model.parameters())
        n_trn = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"CBraMod: {n_par/1e6:.2f}M params ({n_trn/1e6:.2f}M trainable)")

        criterion = FocalLoss(alpha=0.25, gamma=2.0, label_smoothing=0.1)
        opt = optim.AdamW([
            {"params": model.backbone.parameters(), "lr": args.lr},
            {"params": model.head.parameters(), "lr": args.lr * args.head_lr_mult},
        ], weight_decay=0.05)
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

        import time
        for ep in range(args.epochs):
            model.train()
            t0 = time.time()
            pbar = tqdm(train_loader, desc=f"fold {fi+1} epoch {ep+1}/{args.epochs}",
                        leave=False)
            for xb, yb in pbar:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            sched.step()

            if args.benchmark and ep == 0:
                dt = time.time() - t0
                vram = (torch.cuda.max_memory_allocated() / 1e9
                        if device.type == "cuda" else 0)
                n_batches = len(train_loader)
                print(f"\nBENCHMARK")
                print(f"  {n_batches} batches in {dt:.0f}s "
                      f"({n_batches/dt:.2f} it/s)")
                print(f"  peak VRAM {vram:.2f} GB of 11 GB")
                print(f"  projected: {dt*args.epochs/60:.0f} min/fold, "
                      f"{dt*args.epochs*args.folds/3600:.1f} h for "
                      f"{args.folds} folds")
                return

        probs, preds, ys = evaluate(model, val_loader, device)
        m = {"accuracy": accuracy_score(ys, preds),
             "precision": precision_score(ys, preds, zero_division=0),
             "recall": recall_score(ys, preds, zero_division=0),
             "f1": f1_score(ys, preds, zero_division=0),
             "auc": roc_auc_score(ys, probs) if len(np.unique(ys)) > 1 else float("nan")}
        results.append(m | {"val_patients": val_p})
        print(f"fold {fi+1}: AUC {m['auc']:.4f} | recall {m['recall']:.3f} | "
              f"precision {m['precision']:.3f}")

        out = CKPT / f"{args.tag}_fold{fi+1}.pt"
        torch.save({"fold": fi + 1, "model_state_dict": model.state_dict(),
                    "val_patients": val_p, "metrics": m,
                    "backbone": "CBraMod", "max_segments": args.max_segments}, out)
        print(f"saved {out.name}\n")

        del model, train_loader, val_loader
        gc.collect(); torch.cuda.empty_cache()

    aucs = [r["auc"] for r in results]
    print(f"{'='*60}")
    print(f"CBraMod  AUC {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}")
    print(f"per fold: {', '.join(f'{a:.4f}' for a in aucs)}")
    print(f"\nEEG-Conformer on the same protocol: 0.7795 +/- 0.0168")
    print("(note: that baseline included CHB03/CHB05; rerun it with "
          "--exclude CHB03 CHB05 for a strictly paired number)")

    Path("cv_results").mkdir(exist_ok=True)
    f = Path("cv_results") / f"{args.tag}_{datetime.now():%Y%m%d_%H%M%S}.json"
    f.write_text(json.dumps([{k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                              for k, v in r.items()} for r in results], indent=2))
    print(f"saved {f}")


if __name__ == "__main__":
    main()
