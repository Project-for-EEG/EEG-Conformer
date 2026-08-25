"""
Convert CHB-MIT EDF files to CBraMod's input format.

CBraMod expects (channels, patches, 200) at 200 Hz, with amplitudes in units of
100 uV -- its own CHB-MIT loader divides microvolts by 100. Our existing .npz
files are z-scored at 256 Hz, and z-scoring is irreversible (the per-segment
mean and std are gone), so the data has to be rebuilt from the recordings.

What stays identical to the main pipeline, so the comparison is fair:
  - same 23-channel bipolar montage, same channel order
  - same 4 s windows with 50% overlap, so window i covers the same seconds
  - same labels, from the same corrected annotations
  - same 0.5-50 Hz bandpass

What changes, because CBraMod requires it:
  - 256 Hz -> 200 Hz (1024 -> 800 samples), reshaped to (23, 4, 200)
  - per-channel z-score -> microvolts / 100

Verification compares the label vectors against the existing .npz files: if a
window is labelled differently, the segmentation has drifted and the comparison
would be meaningless.

Usage:
    python edf_to_cbramod.py --verify
    python edf_to_cbramod.py
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import signal

from download_and_convert import TARGET_CHANNELS, find_matching_channel, parse_summary_file

try:
    import mne
    mne.set_log_level("ERROR")
except ImportError:
    print("mne is required: pip install mne")
    sys.exit(1)

RAW = Path("raw_data")
PREP = Path("preprocessed_data")
OUT = Path("cbramod_data")

SRC_RATE = 256
DST_RATE = 200
WINDOW_SEC = 4.0
STEP_SEC = 2.0                       # 50% overlap, same as the main pipeline
SRC_WINDOW = int(WINDOW_SEC * SRC_RATE)   # 1024
SRC_STEP = int(STEP_SEC * SRC_RATE)       # 512
DST_WINDOW = int(WINDOW_SEC * DST_RATE)   # 800
PATCHES = int(WINDOW_SEC)                 # 4 patches of 1 s
PATCH_LEN = DST_RATE                      # 200
LOWCUT, HIGHCUT = 0.5, 50.0


def bandpass(x, fs):
    nyq = 0.5 * fs
    b, a = signal.butter(4, [LOWCUT / nyq, HIGHCUT / nyq], btype="band")
    return signal.filtfilt(b, a, x, axis=-1)


def convert(edf_path, seizures):
    """(segments, labels) for one recording, in CBraMod's format."""
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    data = raw.get_data()                      # volts, as mne returns
    n = data.shape[1]

    channels = np.zeros((len(TARGET_CHANNELS), n))
    for i, target in enumerate(TARGET_CHANNELS):
        idx = find_matching_channel(target, raw.ch_names)
        if idx is not None:
            channels[i] = data[idx]

    # mne gives volts; CBraMod wants microvolts / 100, i.e. volts * 1e6 / 100
    channels = channels * 1e4

    sample_labels = np.zeros(n, dtype=np.int64)
    for s in seizures:
        lo = int(s["start"] * SRC_RATE)
        hi = min(int(s["end"] * SRC_RATE), n)
        if hi > lo:
            sample_labels[lo:hi] = 1

    segments, labels = [], []
    for start in range(0, n - SRC_WINDOW + 1, SRC_STEP):
        seg = channels[:, start:start + SRC_WINDOW]
        try:
            seg = bandpass(seg, SRC_RATE)
        except Exception:
            pass
        # resample the window itself, so window boundaries stay aligned with
        # the main pipeline rather than drifting across the recording
        seg = signal.resample(seg, DST_WINDOW, axis=-1)
        segments.append(seg.reshape(len(TARGET_CHANNELS), PATCHES, PATCH_LEN)
                        .astype(np.float16))
        # identical >50% rule as preprocess_data.py, on the source-rate labels
        labels.append(1 if np.mean(sample_labels[start:start + SRC_WINDOW]) > 0.5
                      else 0)

    return np.stack(segments), np.array(labels, dtype=np.int64)


def annotations_for(patient):
    s = RAW / patient / f"{patient}-summary.txt"
    return parse_summary_file(s) if s.exists() else {}


def verify(limit=5):
    """Labels and window count must match the existing pipeline exactly."""
    print("Comparing against preprocessed_data/ (labels and alignment)\n")
    ok_all = True
    checked = 0
    for edf in sorted(RAW.glob("*/*.edf")):
        patient = edf.parent.name
        ref_path = PREP / patient.upper() / f"{edf.stem}_labeled.npz"
        if not ref_path.exists():
            continue
        seg, lab = convert(edf, annotations_for(patient).get(edf.name, []))
        with np.load(ref_path) as ref:
            ref_lab = ref["labels"]
        same_n = len(lab) == len(ref_lab)
        same_lab = same_n and np.array_equal(lab, ref_lab)
        ok = same_n and same_lab and seg.shape[1:] == (23, PATCHES, PATCH_LEN)
        ok_all &= ok
        print(f"  {edf.name:18s} {str(seg.shape):>22s}  "
              f"windows {'=' if same_n else 'DIFFER'}  "
              f"labels {'=' if same_lab else 'DIFFER'}  "
              f"{'OK' if ok else 'MISMATCH'}")
        checked += 1
        if checked >= limit:
            break
    print(f"\n{'alignment matches the main pipeline' if ok_all else 'MISMATCH -- do not train on this'}")
    return ok_all


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--limit", type=int, default=5)
    args = ap.parse_args()

    if args.verify:
        sys.exit(0 if verify(args.limit) else 1)

    if not verify(3):
        print("\nAborting: conversion does not align with the main pipeline.")
        sys.exit(1)

    print()
    edfs = sorted(RAW.glob("*/*.edf"))
    total_seg = total_sz = 0
    for i, edf in enumerate(edfs, 1):
        patient = edf.parent.name.upper()
        out = OUT / patient / f"{edf.stem}_labeled.npz"
        if out.exists():
            with np.load(out) as d:
                total_seg += len(d["labels"])
                total_sz += int(d["labels"].sum())
            continue
        seg, lab = convert(edf, annotations_for(edf.parent.name).get(edf.name, []))
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, segments=seg, labels=lab)
        tmp.replace(out)
        total_seg += len(lab)
        total_sz += int(lab.sum())
        print(f"[{i}/{len(edfs)}] {patient}/{edf.stem}  {len(lab):,} windows, "
              f"{int(lab.sum())} seizure")

    print(f"\n{total_seg:,} windows, {total_sz:,} seizure "
          f"({100*total_sz/total_seg:.3f}%) -> {OUT}/")


if __name__ == "__main__":
    main()
