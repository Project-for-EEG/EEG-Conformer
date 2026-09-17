"""
Does adding patients keep helping, or does it flatten?

Going from 16 to 22 CHB-MIT training subjects gained 0.10 event sensitivity,
the most reliable positive result in this project. Then it stopped being
measurable: CHB-MIT has 23 subjects, so 22 is the largest training set it can
ever provide, and two points do not show the shape of a curve.

Helsinki has 46 babies with seizures, twice that. Training on 8, 16, 32 and 67
patients against a FIXED held-out test set gives four points over an 8x range.
If sensitivity is still climbing at the top, more patients is a real strategy.
If it flattened by 20, the one lever this project found has a low ceiling, and
that changes what is worth doing next.

DESIGN

    79 babies = 46 with seizures + 33 without

    12 seizure-bearing babies are held out as a fixed test set and never
    trained on. Everything else -- 34 with seizures, 33 without -- is the
    training pool, sampled up to each size.

The test set is fixed rather than rotated, because with leave-one-patient-out
the training size cannot be varied independently: it is always N-1. A fixed
test set makes training size the only thing that changes.

Subsets are nested (the 8 are inside the 16, inside the 32) so the curve shows
the effect of adding patients rather than of drawing a different sample. The
proportion of seizure-bearing to background-only patients is held roughly
constant across sizes, so the curve is not confounded by the mix changing.

Babies with no seizures under the majority-vote rule are kept in the training
pool -- they are usable background -- but cannot be scored, so they are never
in the test set.

CAVEAT ON THE NUMBERS

Neonatal EEG is 12.4% seizure against CHB-MIT's 1.4%, a far easier problem.
The absolute sensitivities here are not comparable with the CHB-MIT headline
and should not be quoted beside it. The shape of the curve is the result.

Usage:
    python helsinki_curve.py
    python helsinki_curve.py --sizes 8 16 32 67 --epochs 15
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
from train_memory_efficient import (get_patient_list, load_patient_data,
                                    set_seed, train_one_fold)

ROOT = Path("preprocessed_data_helsinki20")
CKPT = Path("checkpoints")
N_TEST = 12


def split_patients(seed=42):
    """(test, pool_with_seizures, pool_background_only), disjoint."""
    patients = get_patient_list(ROOT)
    with_sz, without = [], []
    for p in patients:
        total = 0
        for f in sorted((ROOT / p).glob("*.npz")):
            with np.load(f) as d:
                total += int(d["labels"].sum())
        (with_sz if total > 0 else without).append(p)

    rng = np.random.RandomState(seed)
    order = list(with_sz)
    rng.shuffle(order)
    test = sorted(order[:N_TEST])
    pool_sz = sorted(order[N_TEST:])

    order_bg = list(without)
    rng.shuffle(order_bg)
    return test, pool_sz, order_bg


def training_subset(pool_sz, pool_bg, n):
    """Nested subset of n patients, keeping the seizure/background mix steady.

    Nested so that a larger size is a superset of a smaller one: the curve then
    measures adding patients, not resampling them.
    """
    total = len(pool_sz) + len(pool_bg)
    n = min(n, total)
    n_sz = min(len(pool_sz), max(1, round(n * len(pool_sz) / total)))
    n_bg = min(len(pool_bg), n - n_sz)
    return pool_sz[:n_sz] + pool_bg[:n_bg], n_sz


def load_many(patients, max_segments):
    """Training data, capped per patient with every seizure kept."""
    X, y = [], []
    for p in patients:
        a, b = load_patient_data(p, ROOT, max_segments)
        if a is not None:
            X.append(a)
            y.append(b)
    return np.concatenate(X), np.concatenate(y)


def load_test(patients):
    """Test data at its NATURAL class balance.

    load_patient_data keeps every seizure window and fills the remainder with
    background, which bounds memory without losing positives. On CHB-MIT at
    1.4% seizure that is harmless. On Helsinki at 12.4% it inverts the balance:
    a 500-window cap on a baby with ~316 seizure windows yields a 63% seizure
    sample, and a test set built that way came out 90.8% seizure -- a set where
    predicting "seizure" always would score 0.908.

    A test set has to look like the data the detector would actually see, so
    this takes a contiguous stretch of each recording instead, keeping seizure
    and background in the proportion they occur.
    """
    X, y = [], []
    for p in patients:
        for f in sorted((ROOT / p).glob("*.npz")):
            with np.load(f) as d:
                X.append(d["segments"])
                y.append(d["labels"])
    return np.concatenate(X), np.concatenate(y)


def true_seizure_rate(patients):
    """Seizure fraction over every window these patients have."""
    total = pos = 0
    for p in patients:
        for f in sorted((ROOT / p).glob("*.npz")):
            with np.load(f) as d:
                lab = d["labels"]
            total += len(lab)
            pos += int(lab.sum())
    return pos / total


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sizes", type=int, nargs="*", default=[8, 16, 32, 67])
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--max-segments", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="helscurve_")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = get_config()
    config.model.num_channels = 20

    test, pool_sz, pool_bg = split_patients(args.seed)
    print("Helsinki only. %d babies with seizures, %d without."
          % (len(pool_sz) + N_TEST, len(pool_bg)))
    print("fixed test set, never trained on: %s" % ", ".join(test))
    print("training pool: %d with seizures + %d background-only\n"
          % (len(pool_sz), len(pool_bg)))

    X_test, y_test = load_test(test)
    print("test set: %d windows, %d seizure (%.1f%%)\n"
          % (len(y_test), int(y_test.sum()), 100 * y_test.mean()))
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_test).float(),
                      torch.from_numpy(y_test).long()),
        batch_size=config.training.batch_size, shuffle=False)
    del X_test

    results = []
    for n in args.sizes:
        subset, n_sz = training_subset(pool_sz, pool_bg, n)
        X_train, y_train = load_many(subset, args.max_segments)
        print("[%d patients, %d with seizures] %d windows, %d seizure (%.1f%%)"
              % (len(subset), n_sz, len(y_train), int(y_train.sum()),
                 100 * y_train.mean()))

        train_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_train).float(),
                          torch.from_numpy(y_train).long()),
            batch_size=config.training.batch_size, shuffle=True)
        del X_train
        gc.collect()

        model = create_model(config.model).to(device)
        metrics, _, _ = train_one_fold(model, train_loader, test_loader,
                                       config, device, 0, args.epochs)

        out = CKPT / ("%s%d.pt" % (args.tag, len(subset)))
        torch.save({"model_state_dict": model.state_dict(),
                    "model_config": asdict(config.model),
                    "val_patients": test,
                    "n_train_patients": len(subset),
                    "n_train_with_seizures": n_sz,
                    "train_patients": subset,
                    "metrics": metrics}, out)

        results.append({"n_patients": len(subset), "n_with_seizures": n_sz,
                        "n_windows": len(y_train), **metrics})
        print("    AUC %.4f | recall %.3f | saved %s\n"
              % (metrics.get("auc", float("nan")), metrics["recall"], out.name))

        del model, train_loader
        gc.collect()
        torch.cuda.empty_cache()

    print("%-10s %-14s %10s %10s" % ("patients", "with seizures", "AUC", "recall"))
    for r in results:
        print("%-10d %-14d %10.4f %10.3f"
              % (r["n_patients"], r["n_with_seizures"], r["auc"], r["recall"]))

    f = Path("cv_results") / ("helsinki_curve_%s.json"
                              % datetime.now().strftime("%Y%m%d_%H%M%S"))
    f.write_text(json.dumps({"test_patients": test, "points": results},
                            indent=2, default=str))
    print("\nsaved %s" % f)
    print("event-level scoring of these checkpoints is the number that counts;")
    print("window AUC has misread three decisions in this project already.")


if __name__ == "__main__":
    main()
