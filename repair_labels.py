"""
Repair seizure labels in preprocessed_data/*.npz.

The original conversion (download_and_convert.py) parsed CHB-MIT summary files
with a bare `\\d+` regex, which grabbed the seizure *index* instead of the
timestamp whenever a file contained more than one seizure ("Seizure 1 Start
Time: 1724 seconds" -> 1). Those seizures were written as start=1, end=1, so no
samples were ever labeled. 18 of 24 patients lost their entire ground truth.

The EEG signal in the .npz files is fine -- only the label vector is wrong. The
segmentation in preprocess_data.py is deterministic: segment i covers samples
[i*512, i*512+1024) of the recording, i.e. seconds [i*2, i*2+4). So labels can
be recomputed straight from the corrected annotations without re-reading the
85 GB of CSVs or re-running preprocessing.

Usage:
    python repair_labels.py --verify     # check against known-good patients
    python repair_labels.py --apply      # rewrite the .npz files
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

from download_and_convert import parse_summary_file

RAW_DIR = Path("raw_data")
PREP_DIR = Path("preprocessed_data")

SAMPLING_RATE = 256
WINDOW = 1024          # 4 s
STEP = 512             # 50% overlap
SEIZURE_FRACTION = 0.5  # segment is seizure if >50% of its samples are

# Patients whose labels survived the original bug -- used to prove that the
# segment->time mapping below is correct before touching anything else.
KNOWN_GOOD = ["CHB01", "CHB02", "CHB24"]


def load_annotations():
    """Parse every summary on disk.

    Returns (annotations, covered_patients) where annotations maps EDF filename
    -> [(start_sec, end_sec), ...] and covered_patients is the set of patient
    ids (upper-case, e.g. "CHB06") that actually have a summary file.

    The distinction matters: a file missing from `annotations` is seizure-free
    only if its patient IS in covered_patients. CHB03 and CHB05 came from an
    earlier download and have no summary here -- their existing labels are
    already correct and must be left alone, not overwritten with zeros.
    """
    ann = {}
    covered = set()
    for summary in sorted(RAW_DIR.glob("*/*-summary.txt")):
        covered.add(summary.parent.name.upper())
        for fname, times in parse_summary_file(summary).items():
            ann[fname] = [(t["start"], t["end"]) for t in times]
    return ann, covered


def recompute_labels(n_segments, seizures):
    """Label vector for a recording with n_segments windows.

    Mirrors preprocess_data.py: window i spans samples [i*STEP, i*STEP+WINDOW).
    """
    n_samples = (n_segments - 1) * STEP + WINDOW
    mask = np.zeros(n_samples, dtype=np.float32)
    for start_sec, end_sec in seizures:
        lo = int(start_sec * SAMPLING_RATE)
        hi = min(int(end_sec * SAMPLING_RATE), n_samples)
        if hi > lo:
            mask[lo:hi] = 1.0

    labels = np.zeros(n_segments, dtype=np.int64)
    for i in range(n_segments):
        s = i * STEP
        if mask[s:s + WINDOW].mean() > SEIZURE_FRACTION:
            labels[i] = 1
    return labels


def npz_to_edf_name(npz_path):
    """preprocessed_data/CHB06/chb06_01_labeled.npz -> chb06_01.edf"""
    return npz_path.stem.replace("_labeled", "") + ".edf"


def iter_npz():
    for patient_dir in sorted(PREP_DIR.iterdir()):
        if patient_dir.is_dir():
            for npz in sorted(patient_dir.glob("*.npz")):
                yield patient_dir.name, npz


def verify(ann):
    """Recompute labels for patients that already have valid ones and compare."""
    print("Verifying segment->time alignment against known-good patients\n")
    all_ok = True
    for patient, npz in iter_npz():
        if patient not in KNOWN_GOOD:
            continue
        edf = npz_to_edf_name(npz)
        if edf not in ann:
            print(f"  {npz.name:28s} SKIP (no annotation; seizure-free file)")
            continue

        existing = np.load(npz)["labels"]
        rebuilt = recompute_labels(len(existing), ann[edf])
        match = np.array_equal(existing, rebuilt)
        all_ok &= match
        print(f"  {npz.name:28s} existing={int(existing.sum()):4d}  "
              f"rebuilt={int(rebuilt.sum()):4d}  "
              f"{'MATCH' if match else 'MISMATCH <-- alignment is wrong'}")

    print()
    if all_ok:
        print("All known-good files reproduce exactly. Mapping is correct.")
    else:
        print("Mismatch found. Do NOT apply -- the segment->time mapping is off.")
    return all_ok


def apply(ann, covered):
    """Rewrite label arrays in place (atomically) where they changed."""
    changed = files_touched = skipped = 0
    before = after = 0

    for patient, npz in iter_npz():
        data = np.load(npz)
        existing = data["labels"]
        before += int(existing.sum())

        if patient not in covered:
            # No summary for this patient -- we cannot recompute, so leave the
            # existing (already valid) labels untouched.
            after += int(existing.sum())
            skipped += 1
            del data
            continue

        edf = npz_to_edf_name(npz)
        rebuilt = (recompute_labels(len(existing), ann[edf])
                   if edf in ann else np.zeros_like(existing))
        after += int(rebuilt.sum())

        if np.array_equal(existing, rebuilt):
            del data
            continue

        segments = data["segments"]
        # Suffix must end in .npz -- numpy appends .npz otherwise, and the
        # atomic replace below would then target a file that does not exist.
        tmp = npz.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, segments=segments, labels=rebuilt)
        del data, segments
        os.replace(tmp, npz)

        changed += int(rebuilt.sum()) - int(existing.sum())
        files_touched += 1
        print(f"  {patient}/{npz.name:28s} {int(existing.sum()):4d} -> "
              f"{int(rebuilt.sum()):4d} seizure segments")

    print(f"\nRewrote {files_touched} files "
          f"({skipped} left untouched -- no summary available)")
    print(f"Seizure segments: {before:,} -> {after:,}  (+{changed:,})")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--verify", action="store_true",
                       help="check alignment against known-good patients only")
    group.add_argument("--apply", action="store_true",
                       help="rewrite .npz label arrays")
    args = parser.parse_args()

    ann, covered = load_annotations()
    total = sum(len(v) for v in ann.values())
    print(f"Loaded {total} seizure annotations across {len(ann)} files "
          f"for {len(covered)} patients\n")

    if args.verify:
        sys.exit(0 if verify(ann) else 1)

    if not verify(ann):
        print("\nAborting: verification failed.")
        sys.exit(1)
    print()
    apply(ann, covered)


if __name__ == "__main__":
    main()
