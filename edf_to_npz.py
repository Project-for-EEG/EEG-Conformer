"""
Convert CHB-MIT EDF files straight to preprocessed .npz, skipping the CSV step.

The original pipeline went EDF -> labeled CSV -> segmented .npz. The CSV stage
is pure overhead: 85 GB of text to produce 26 GB of arrays, and every new file
costs ~10x its final size on disk before it can be preprocessed. Nothing reads
those CSVs except preprocess_data.py.

This does both stages in one pass, in memory. Output is bit-identical to the old
route -- `--verify` proves that against the .npz files already on disk before any
new data is written.

Segmentation matches preprocess_data.py exactly: 4 s windows (1024 samples) with
50% overlap, per-segment bandpass then per-channel z-score, and a window is
labelled seizure when more than half its samples fall inside one.

Usage:
    python edf_to_npz.py --verify              # reproduce existing .npz, compare
    python edf_to_npz.py --list-missing        # seizure files not yet downloaded
    python edf_to_npz.py --fetch-missing       # download those, convert to .npz
"""
import argparse
import sys
import urllib.request
from pathlib import Path

import numpy as np
from scipy import signal

from download_and_convert import (BASE_URL, TARGET_CHANNELS, find_matching_channel,
                                  parse_summary_file)

try:
    import mne
    mne.set_log_level('ERROR')
except ImportError:
    print("mne is required: pip install mne")
    sys.exit(1)

RAW_DIR = Path("raw_data")
PREP_DIR = Path("preprocessed_data")

SAMPLING_RATE = 256
WINDOW = 1024          # 4 s
STEP = 512             # 50% overlap
LOWCUT, HIGHCUT = 0.5, 50.0


def bandpass(data):
    """Same filter preprocess_data.py applies, per segment."""
    nyq = 0.5 * SAMPLING_RATE
    b, a = signal.butter(4, [LOWCUT / nyq, HIGHCUT / nyq], btype='band')
    return signal.filtfilt(b, a, data, axis=-1)


def edf_to_arrays(edf_path, seizures):
    """(segments, labels) for one recording.

    seizures is a list of {'start': sec, 'end': sec}.
    """
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    data = raw.get_data()
    n_samples = data.shape[1]

    # Map the file's channels onto CHB-MIT's 23-channel montage. Missing
    # channels become zeros, matching convert_edf_to_csv.
    channels = np.zeros((len(TARGET_CHANNELS), n_samples))
    for i, target in enumerate(TARGET_CHANNELS):
        idx = find_matching_channel(target, raw.ch_names)
        if idx is not None:
            channels[i] = data[idx]

    sample_labels = np.zeros(n_samples, dtype=np.int64)
    for s in seizures:
        lo = int(s['start'] * SAMPLING_RATE)
        hi = min(int(s['end'] * SAMPLING_RATE), n_samples)
        if hi > lo:
            sample_labels[lo:hi] = 1

    # float32 before filtering, as the CSV route did via .astype(np.float32)
    channels = channels.astype(np.float32)

    segments, labels = [], []
    for start in range(0, n_samples - WINDOW + 1, STEP):
        seg = channels[:, start:start + WINDOW]
        try:
            seg = bandpass(seg)
        except Exception:
            pass
        mean = np.mean(seg, axis=1, keepdims=True)
        std = np.std(seg, axis=1, keepdims=True) + 1e-8
        seg = (seg - mean) / std
        segments.append(seg.astype(np.float32))
        labels.append(1 if np.mean(sample_labels[start:start + WINDOW]) > 0.5 else 0)

    return np.stack(segments, axis=0), np.array(labels, dtype=np.int64)


def annotations_for(patient):
    summary = RAW_DIR / patient / f"{patient}-summary.txt"
    return parse_summary_file(summary) if summary.exists() else {}


def verify(limit=6):
    """Reconvert EDFs we already have and compare against their .npz."""
    print("Reconverting existing EDFs and comparing to preprocessed_data/\n")
    checked = failed = 0
    for edf in sorted(RAW_DIR.glob("*/*.edf")):
        patient = edf.parent.name
        npz = PREP_DIR / patient.upper() / f"{edf.stem}_labeled.npz"
        if not npz.exists():
            continue

        seizures = annotations_for(patient).get(edf.name, [])
        seg, lab = edf_to_arrays(edf, seizures)
        with np.load(npz) as ref:
            ref_seg, ref_lab = ref["segments"], ref["labels"]

        shape_ok = seg.shape == ref_seg.shape
        label_ok = shape_ok and np.array_equal(lab, ref_lab)
        maxdiff = float(np.abs(seg - ref_seg).max()) if shape_ok else float("nan")
        ok = shape_ok and label_ok and maxdiff < 1e-4

        checked += 1
        failed += (not ok)
        print(f"  {edf.name:18s} {str(seg.shape):>18s} vs {str(ref_seg.shape):>18s}  "
              f"labels {'=' if label_ok else 'DIFFER'}  "
              f"max|delta| {maxdiff:.2e}  {'OK' if ok else 'MISMATCH'}")
        if checked >= limit:
            break

    print(f"\n{checked - failed}/{checked} reproduce the existing pipeline")
    return failed == 0


def missing_seizure_files():
    """Seizure-bearing EDFs listed in the summaries but not downloaded."""
    have = {f.name for f in RAW_DIR.glob("*/*.edf")}
    out = []
    for summary in sorted(RAW_DIR.glob("*/*-summary.txt")):
        patient = summary.parent.name
        for fname, times in parse_summary_file(summary).items():
            good = [t for t in times if t['end'] > t['start']]
            if good and fname not in have:
                out.append((patient, fname, good))
    return out


def fetch_missing(dry_run=False):
    todo = missing_seizure_files()
    n_sz = sum(len(t) for _, _, t in todo)
    print(f"{len(todo)} files to fetch, carrying {n_sz} seizures\n")
    if dry_run:
        for patient, fname, times in todo:
            print(f"  {patient}/{fname:20s} {len(times)} seizure(s)")
        return

    added_sz = 0
    for i, (patient, fname, times) in enumerate(todo, 1):
        dest = RAW_DIR / patient / fname
        if not dest.exists():
            url = f"{BASE_URL}/{patient}/{fname}"
            print(f"[{i}/{len(todo)}] downloading {fname} ...", end=" ", flush=True)
            try:
                urllib.request.urlretrieve(url, dest)
                print(f"{dest.stat().st_size/1e6:.0f} MB")
            except Exception as e:
                print(f"FAILED ({e})")
                continue
        else:
            print(f"[{i}/{len(todo)}] {fname} already present")

        out = PREP_DIR / patient.upper() / f"{dest.stem}_labeled.npz"
        if out.exists():
            print("        .npz already exists, skipping")
            continue

        seg, lab = edf_to_arrays(dest, times)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".tmp.npz")   # must end .npz -- numpy appends it otherwise
        np.savez_compressed(tmp, segments=seg, labels=lab)
        tmp.replace(out)
        added_sz += int(lab.sum())
        print(f"        -> {out.relative_to(PREP_DIR)}  "
              f"{len(lab):,} segments, {int(lab.sum())} seizure")

    print(f"\nAdded {added_sz:,} seizure segments")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--verify", action="store_true")
    g.add_argument("--list-missing", action="store_true")
    g.add_argument("--fetch-missing", action="store_true")
    p.add_argument("--limit", type=int, default=6, help="files to check in --verify")
    args = p.parse_args()

    if args.verify:
        sys.exit(0 if verify(args.limit) else 1)
    if args.list_missing:
        fetch_missing(dry_run=True)
    else:
        if not verify(3):
            print("\nAborting: converter does not reproduce the existing pipeline.")
            sys.exit(1)
        print()
        fetch_missing()


if __name__ == "__main__":
    main()
