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


def seizure_bearing(root=None):
    """(babies with majority-vote seizures, babies without), both sorted.

    A baby with no majority-vote seizure cannot be scored for sensitivity --
    there is nothing to catch -- so it can only ever be training background.
    Of the 33 such babies, 10 have a seizure that exactly one of the three
    experts marked; one expert of three is not a majority, so those seizures
    are labelled background here. That is a real limitation of the majority
    rule and not something a split can fix.
    """
    root = root or ROOT
    with_sz, without = [], []
    for p in get_patient_list(root):
        total = 0
        for f in sorted((root / p).glob("*.npz")):
            with np.load(f) as d:
                total += int(d["labels"].sum())
        (with_sz if total > 0 else without).append(p)
    return sorted(with_sz), sorted(without)


def reviewer_counts(path="additional_data/helsinki/clinical_information.csv"):
    """{baby: how many of the three experts marked any seizure}, or {}.

    Used only to stratify the folds. All 12 babies in the original fixed test
    set were ones all three experts agreed on, so performance on that split
    described the unambiguous end of the label distribution. Spreading the
    6 testable two-expert babies evenly keeps the folds comparable instead of
    putting all the ambiguity in one of them.
    """
    import csv
    f = Path(path)
    if not f.exists():
        return {}
    rows = list(csv.DictReader(f.open(encoding="utf-8-sig")))
    if not rows:
        return {}
    key = next((k for k in rows[0] if "Reviewer" in k), None)
    idk = list(rows[0])[0]
    out = {}
    for r in rows:
        try:
            out["HEL%02d" % int(str(r[idk]).strip())] = int(str(r[key]).strip())
        except (TypeError, ValueError):
            pass
    return out


def cv_folds(n_folds=4, seed=42, root=None):
    """[[babies], ...] partitioning every scorable baby into n_folds groups.

    A partition, not n_folds random draws: each baby is in exactly one test
    fold, so every baby is scored exactly once by a model that never saw it.
    The fixed 12-baby split this replaces was the limiting instrument -- the
    8-to-67 gain moved 0.285 across three draws of it while moving only 0.057
    when the babies were held fixed and only the initialisation varied.

    Folds are stratified by reviewer count, so label certainty is spread
    rather than concentrated.
    """
    with_sz, _ = seizure_bearing(root)
    nrev = reviewer_counts()
    rng = np.random.RandomState(seed)
    folds = [[] for _ in range(n_folds)]
    # deal each certainty stratum round-robin, so fold sizes stay within one
    for level in sorted({nrev.get(p, -1) for p in with_sz}, reverse=True):
        group = [p for p in with_sz if nrev.get(p, -1) == level]
        rng.shuffle(group)
        for i, baby in enumerate(group):
            folds[i % n_folds].append(baby)
    return [sorted(f) for f in folds]


def split_patients(seed=42, fold=None, n_folds=4):
    """(test, pool_with_seizures, pool_background_only), disjoint.

    With fold=None this keeps the original behaviour: a random N_TEST babies
    held out, which is what every published number in this project used.
    With fold=k it returns the k-th cross-validation fold instead.
    """
    with_sz, without = seizure_bearing()

    if fold is not None:
        folds = cv_folds(n_folds, seed)
        test = folds[fold]
        pool_sz = sorted(p for p in with_sz if p not in set(test))
    else:
        rng = np.random.RandomState(seed)
        order = list(with_sz)
        rng.shuffle(order)
        test = sorted(order[:N_TEST])
        pool_sz = sorted(order[N_TEST:])

    # The guard exists because this project has already shipped a silently
    # wrong answer from a train/test mix-up elsewhere. Cheap to assert.
    overlap = set(test) & set(pool_sz)
    if overlap:
        raise AssertionError("test and training pool overlap: %s" % sorted(overlap))

    order_bg = list(without)
    np.random.RandomState(seed).shuffle(order_bg)
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
    # --seed alone moves the held-out babies AND the initialisation together,
    # which is why repeating the curve at three seeds gave a spread of 0.285:
    # seeds 42 and 43 share only 3 of their 12 test babies, and their test sets
    # hold 158 and 104 events. Pinning --split-seed separates the two, so the
    # variance can be attributed to patient sampling or to training.
    ap.add_argument("--fold", type=int, default=None,
                    help="cross-validation fold to hold out, 0-based. Omit "
                         "for the original single random split.")
    ap.add_argument("--n-folds", type=int, default=4)
    ap.add_argument("--split-seed", type=int, default=None,
                    help="seed for choosing the held-out babies; defaults to "
                         "--seed. Pin it to vary only the initialisation.")
    ap.add_argument("--tag", default="helscurve_")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = get_config()
    config.model.num_channels = 20

    split_seed = args.seed if args.split_seed is None else args.split_seed
    test, pool_sz, pool_bg = split_patients(split_seed, fold=args.fold, n_folds=args.n_folds)
    if split_seed != args.seed:
        print("split seed %d (held-out babies), training seed %d"
              % (split_seed, args.seed))
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
