"""
Will one leave-one-patient-out fold with Helsinki fit in this machine's RAM?

The trainer loads an entire fold into memory at once. With 79 Helsinki patients
pinned into training, arithmetic says 202,000 windows at 20 channels is 16.5 GB
against 17.1 GB of total system RAM -- which does not crash, it spills to the
page file and grinds. That is the most likely cause of fold times swinging
between 46 and 83 minutes, and of a benchmark setting that ran 18x slow.

Rather than commit four days to a configuration that may thrash, this loads one
fold for real and reports the peak.

Tries each setting in turn and stops at the first that leaves comfortable
headroom, so the answer is a concrete --max-segments to use rather than an
estimate.

Usage:
    python memcheck_fold.py
    python memcheck_fold.py --channels 23
"""
import argparse
import gc

import numpy as np
import psutil

from train_memory_efficient import (apply_channel_drop, get_patient_list,
                                    load_patient_data)

ROOTS = ["preprocessed_data", "preprocessed_data_helsinki20"]
HEADROOM_GB = 3.0        # leave this much for Windows, CUDA context, the model


def gb(n):
    return n / 1e9


def try_fold(max_segments, drop_channels, held_out="CHB02"):
    """Load one fold's training set and report peak process memory."""
    proc = psutil.Process()
    gc.collect()
    before = proc.memory_info().rss

    patients = get_patient_list(ROOTS)
    train = [p for p in patients if p != held_out]

    total = 0
    parts = []
    for p in train:
        X, y = load_patient_data(p, ROOTS, max_segments)
        if X is None:
            continue
        X = apply_channel_drop(X, drop_channels)
        parts.append(X)
        total += len(y)

    X_train = np.concatenate(parts, axis=0)
    del parts
    gc.collect()

    peak = proc.memory_info().rss
    avail = psutil.virtual_memory().available
    shape = X_train.shape
    del X_train
    gc.collect()
    return total, shape, gb(peak - before), gb(peak), gb(avail)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--channels", type=int, default=20, choices=(20, 23))
    ap.add_argument("--settings", type=int, nargs="*",
                    default=[2000, 1000, 500])
    args = ap.parse_args()

    v = psutil.virtual_memory()
    print("system RAM %.1f GB total, %.1f GB available now"
          % (gb(v.total), gb(v.available)))
    print("target: keep at least %.1f GB free while the fold is loaded\n"
          % HEADROOM_GB)

    drop = [19, 20, 21] if args.channels == 20 else None
    print("%-14s %10s %22s %10s %10s %s"
          % ("max-segments", "windows", "shape", "fold GB", "free GB", ""))

    ok_setting = None
    for ms in args.settings:
        try:
            n, shape, used, peak, avail = try_fold(ms, drop)
        except MemoryError:
            print("%-14d %10s %22s %10s %10s  OUT OF MEMORY"
                  % (ms, "-", "-", "-", "-"))
            continue
        safe = avail >= HEADROOM_GB
        print("%-14d %10d %22s %10.1f %10.1f  %s"
              % (ms, n, str(shape), used, avail,
                 "OK" if safe else "TOO TIGHT"))
        if safe and ok_setting is None:
            ok_setting = ms

    print()
    if ok_setting:
        print("use --max-segments %d" % ok_setting)
        print("a 23-fold run at that setting is the same load 23 times over")
    else:
        print("no setting tested leaves %.1f GB free." % HEADROOM_GB)
        print("options: lower --max-segments further, store as float16, or")
        print("use build_memmap.py so the data stays on disk")


if __name__ == "__main__":
    main()
