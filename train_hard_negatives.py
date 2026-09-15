"""
Train on the background the model actually gets wrong.

The standard loader keeps every seizure window and fills the rest of its budget
with background chosen uniformly at random. Most of that background is trivial:
quiet, artifact-free EEG that no model was ever going to flag. It costs
training time and teaches nothing.

The windows that matter are the ones the detector currently fires on -- chewing,
movement, electrode pops, drowsiness transients. Those are what produce 12 to 14
false alarms an hour, and under uniform sampling they are a small and shrinking
fraction of what the model sees.

This replaces the uniform background sample with the highest-scoring background
windows from a model already trained on that fold, keeping the seizure set
untouched. Same number of training windows, same seizure fraction, same
architecture and schedule. The only thing that changes is *which* negatives.

Two things make this different from rebalancing the class ratio:

  * the imbalance ratio is held fixed, so this is not the prior experiment in
    disguise (that one varied 12% / 3.6% / 1.4% and moved AUC by nothing)
  * it targets false alarms directly rather than the ratio, and false alarms
    are the axis this project has not solved

Leakage matters here and is handled explicitly. The mining model for a fold must
never have seen that fold's validation patients, so mining uses the fold's own
checkpoint, which was trained on exactly the training patients. Background is
mined only from training patients. Nothing about the held-out patients
influences which windows are selected.

A pure hardest-K selection collapses onto a handful of pathological recordings,
so a fraction of the background stays uniformly sampled. --hard-fraction
controls the mix; 0.0 reproduces the standard loader.

Usage:
    python train_hard_negatives.py --folds 3 --epochs 15
    python train_hard_negatives.py --hard-fraction 0.5 --tag hn50_
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

from config import get_config, ModelConfig
from model import create_model
from train_memory_efficient import (get_patient_list, resolve_patient_dir,
                                    set_seed, train_one_fold)

PREP = Path("preprocessed_data")
CKPT = Path("checkpoints")
SAME_SUBJECT = [("CHB01", "CHB21")]


def patient_folds(patients, n_folds, seed=42):
    """Patient-level folds, same construction as the other trainers."""
    order = list(patients)
    np.random.seed(seed)
    np.random.shuffle(order)
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
    return folds


def score_background(model, device, patient, batch=512):
    """(file_index, window_index, probability) for every background window.

    Scored in time order, one file at a time, so peak memory is one recording
    rather than one patient.
    """
    pdir = resolve_patient_dir(patient, PREP)
    if pdir is None:
        return []
    out = []
    model.eval()
    for fi, f in enumerate(sorted(pdir.glob("*.npz"))):
        with np.load(f) as d:
            X, lab = d["segments"], d["labels"]
            bg = np.where(lab == 0)[0]
            if len(bg) == 0:
                continue
            probs = np.empty(len(bg), dtype=np.float32)
            with torch.no_grad():
                for i in range(0, len(bg), batch):
                    idx = bg[i:i + batch]
                    t = torch.from_numpy(X[idx]).float().to(device)
                    probs[i:i + len(idx)] = \
                        torch.softmax(model(t), dim=1)[:, 1].cpu().numpy()
        out.extend((fi, int(w), float(p)) for w, p in zip(bg, probs))
    return out


def load_selected(patient, picks):
    """Load exactly the windows named by (file_index, window_index)."""
    pdir = resolve_patient_dir(patient, PREP)
    files = sorted(pdir.glob("*.npz"))
    by_file = {}
    for fi, wi in picks:
        by_file.setdefault(fi, []).append(wi)
    X = []
    for fi, ws in sorted(by_file.items()):
        ws = np.sort(np.array(ws))
        with np.load(files[fi]) as d:
            X.append(d["segments"][ws])
    return np.concatenate(X) if X else None


def build_training_set(patients, miner, device, max_segments, hard_fraction,
                       rng):
    """Seizures kept whole; background split between hardest and uniform."""
    Xs, ys = [], []
    for p in patients:
        pdir = resolve_patient_dir(p, PREP)
        if pdir is None:
            continue
        files = sorted(pdir.glob("*.npz"))

        seiz = []
        for fi, f in enumerate(files):
            with np.load(f) as d:
                for wi in np.where(d["labels"] == 1)[0]:
                    seiz.append((fi, int(wi)))

        budget = max(0, max_segments - len(seiz))
        scored = score_background(miner, device, p)
        if not scored:
            continue

        n_hard = int(budget * hard_fraction)
        n_easy = budget - n_hard

        order = sorted(scored, key=lambda t: -t[2])
        hard = [(fi, wi) for fi, wi, _ in order[:n_hard]]

        remaining = [(fi, wi) for fi, wi, _ in order[n_hard:]]
        if n_easy > 0 and remaining:
            take = rng.choice(len(remaining), size=min(n_easy, len(remaining)),
                              replace=False)
            easy = [remaining[i] for i in take]
        else:
            easy = []

        picks = seiz + hard + easy
        X = load_selected(p, picks)
        if X is None:
            continue
        y = np.array([1] * len(seiz) + [0] * (len(hard) + len(easy)),
                     dtype=np.int64)
        Xs.append(X)
        ys.append(y)
        print("  %-7s %5d seizure %5d hard %5d uniform  (hardest p=%.3f)"
              % (p, len(seiz), len(hard), len(easy),
                 order[0][2] if order else float("nan")))

    return np.concatenate(Xs), np.concatenate(ys)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--max-segments", type=int, default=2000)
    ap.add_argument("--hard-fraction", type=float, default=0.5,
                    help="share of the background budget taken as the "
                         "highest-scoring windows; 0.0 is the standard loader")
    ap.add_argument("--miner-prefix", default="fold",
                    help="checkpoint prefix of the models used to score "
                         "background. Must be models trained on the same "
                         "folds, or held-out patients leak into selection")
    ap.add_argument("--tag", default="hardneg_")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = get_config()
    rng = np.random.default_rng(args.seed)

    patients = get_patient_list(PREP)
    folds = patient_folds(patients, args.folds, args.seed)

    print("hard-negative mining: %.0f%% of background from the windows the "
          "fold's own model scores highest" % (100 * args.hard_fraction))
    print("%d patients, %d folds, miner '%s*'\n"
          % (len(patients), args.folds, args.miner_prefix))

    results = []
    for fi, val_patients in enumerate(folds):
        train_patients = [p for j, f in enumerate(folds) if j != fi for p in f]
        print("[%d/%d] holding out %s" % (fi + 1, args.folds,
                                          ", ".join(val_patients)))

        miner_path = CKPT / ("%s%d.pt" % (args.miner_prefix, fi + 1))
        if not miner_path.exists():
            raise SystemExit("no miner checkpoint at %s" % miner_path)
        mck = torch.load(miner_path, map_location=device, weights_only=False)

        # The miner must not have seen this fold's held-out patients, or the
        # choice of training windows would depend on the test set.
        seen = set(mck.get("val_patients") or [])
        if seen != set(val_patients):
            raise SystemExit(
                "miner %s was validated on %s but this fold holds out %s; "
                "mining with it would leak"
                % (miner_path.name, sorted(seen), sorted(val_patients)))

        miner = create_model(ModelConfig(**mck["model_config"])).to(device)
        miner.load_state_dict(mck["model_state_dict"])
        miner.eval()

        X_train, y_train = build_training_set(
            train_patients, miner, device, args.max_segments,
            args.hard_fraction, rng)
        del miner
        torch.cuda.empty_cache()

        Xv, yv = [], []
        for p in val_patients:
            pdir = resolve_patient_dir(p, PREP)
            if pdir is None:
                continue
            files = sorted(pdir.glob("*.npz"))
            Xp, yp = [], []
            for f in files:
                with np.load(f) as d:
                    Xp.append(d["segments"])
                    yp.append(d["labels"])
            Xp = np.concatenate(Xp)
            yp = np.concatenate(yp)
            if len(yp) > args.max_segments:
                keep = np.sort(rng.choice(len(yp), args.max_segments,
                                          replace=False))
                Xp, yp = Xp[keep], yp[keep]
            Xv.append(Xp)
            yv.append(yp)
        X_val = np.concatenate(Xv)
        y_val = np.concatenate(yv)
        del Xv, yv

        print("    train %d (%d seizure, %.1f%%) | val %d"
              % (len(y_train), int(y_train.sum()),
                 100 * y_train.mean(), len(y_val)))

        train_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_train).float(),
                          torch.from_numpy(y_train).long()),
            batch_size=config.training.batch_size, shuffle=True,
            num_workers=0, pin_memory=False)
        val_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_val).float(),
                          torch.from_numpy(y_val).long()),
            batch_size=config.training.batch_size, shuffle=False,
            num_workers=0, pin_memory=False)
        del X_train, y_train, X_val, y_val
        gc.collect()

        model = create_model(config.model).to(device)
        metrics, _, _ = train_one_fold(model, train_loader, val_loader,
                                       config, device, fi, args.epochs)

        out = CKPT / ("%s%d.pt" % (args.tag, fi + 1))
        torch.save({"fold": fi + 1, "model_state_dict": model.state_dict(),
                    "model_config": asdict(config.model),
                    "val_patients": val_patients, "metrics": metrics,
                    "hard_fraction": args.hard_fraction,
                    "max_segments": args.max_segments}, out)
        print("    AUC %.4f | saved %s\n" % (metrics.get("auc", float("nan")),
                                             out.name))
        results.append(metrics | {"val_patients": val_patients})

        del model, train_loader, val_loader
        gc.collect()
        torch.cuda.empty_cache()

    aucs = [r["auc"] for r in results]
    print("mean AUC %.4f +/- %.4f" % (np.mean(aucs), np.std(aucs)))
    f = Path("cv_results") / ("hardneg_%s.json" % datetime.now().strftime("%Y%m%d_%H%M%S"))
    f.write_text(json.dumps({"hard_fraction": args.hard_fraction,
                             "folds": results}, indent=2, default=str))
    print("saved %s" % f)


if __name__ == "__main__":
    main()
