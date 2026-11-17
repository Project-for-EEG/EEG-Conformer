"""
Pack every preprocessed segment into one memory-mapped array.

Training has never seen the true class distribution. `load_patient_data` caps
each patient at --max-segments and keeps all their seizures, so at 2000 the
model trains on ~11% seizure against 1.4% in reality. That is the leading
suspect for why it scores 0.99 on 1% of ordinary background -- and it cannot be
tested in RAM, because the full dataset is 35 GB as float32.

float16 halves that to 17.4 GB, which fits on disk, and np.memmap gives random
access at page granularity instead of decompressing a whole .npz per sample
(measured at 2.1 s/sample -- 74 h per epoch -- when that was tried).

Precision is not a concern: segments are z-scored per channel, so values sit
near N(0,1) and float16 carries ~3 decimal digits there.

Writes:
    memmap/segments.f16   (n, 23, 1024) float16
    memmap/meta.npz       labels, patient index, patient names, source files

Usage:
    python build_memmap.py
    python build_memmap.py --verify     # compare against the source .npz
"""
import argparse
from pathlib import Path

import numpy as np

PREP = Path("preprocessed_data")
OUT = Path("memmap")
SHAPE_TAIL = (23, 1024)


def plan():
    """(patients, files, per-file lengths) in the order they will be packed."""
    patients = sorted(d.name for d in PREP.iterdir() if d.is_dir())
    files, lengths, owner = [], [], []
    for pi, p in enumerate(patients):
        for f in sorted((PREP / p).glob("*.npz")):
            with np.load(f) as d:
                lengths.append(len(d["labels"]))
            files.append(f)
            owner.append(pi)
    return patients, files, np.array(lengths), np.array(owner)


def build():
    patients, files, lengths, owner = plan()
    total = int(lengths.sum())
    OUT.mkdir(exist_ok=True)
    path = OUT / "segments.f16"

    print(f"{len(patients)} patients, {len(files)} files, {total:,} segments")
    print(f"writing {total * 23 * 1024 * 2 / 1e9:.1f} GB to {path}\n")

    mm = np.memmap(path, dtype=np.float16, mode="w+",
                   shape=(total,) + SHAPE_TAIL)
    labels = np.zeros(total, dtype=np.int64)
    patient_idx = np.zeros(total, dtype=np.int32)

    off = 0
    for i, f in enumerate(files):
        with np.load(f) as d:
            seg, lab = d["segments"], d["labels"]
        n = len(lab)
        mm[off:off + n] = seg.astype(np.float16)
        labels[off:off + n] = lab
        patient_idx[off:off + n] = owner[i]
        off += n
        if (i + 1) % 20 == 0 or i + 1 == len(files):
            print(f"  {i+1:3d}/{len(files)} files  {off:,}/{total:,} segments")
    mm.flush()
    del mm

    np.savez_compressed(
        OUT / "meta.npz", labels=labels, patient_idx=patient_idx,
        patients=np.array(patients), files=np.array([str(f) for f in files]),
        lengths=lengths, total=total)

    print(f"\n{total:,} segments, {int(labels.sum()):,} seizure "
          f"({100*labels.mean():.3f}%)")
    print(f"this is the TRUE distribution -- training has been seeing ~11%")


def verify(n_checks=200):
    meta = np.load(OUT / "meta.npz", allow_pickle=True)
    total = int(meta["total"])
    labels = meta["labels"]
    files = [Path(s) for s in meta["files"]]
    lengths = meta["lengths"]
    offsets = np.concatenate([[0], np.cumsum(lengths)])

    mm = np.memmap(OUT / "segments.f16", dtype=np.float16, mode="r",
                   shape=(total,) + SHAPE_TAIL)

    rng = np.random.default_rng(0)
    picks = rng.choice(total, size=n_checks, replace=False)
    worst = 0.0
    bad_label = 0
    cache_i, cache = -1, None
    for gi in sorted(picks):
        fi = int(np.searchsorted(offsets, gi, side="right") - 1)
        local = gi - offsets[fi]
        if fi != cache_i:
            with np.load(files[fi]) as d:
                cache = (d["segments"], d["labels"])
            cache_i = fi
        ref_seg, ref_lab = cache[0][local], cache[1][local]
        worst = max(worst, float(np.abs(mm[gi].astype(np.float32) - ref_seg).max()))
        bad_label += int(ref_lab != labels[gi])

    print(f"checked {n_checks} random segments against their source .npz")
    print(f"  max |delta| from float16 rounding : {worst:.5f}")
    print(f"  label mismatches                  : {bad_label}")
    ok = bad_label == 0 and worst < 0.01
    print("\nOK -- memmap faithfully reproduces the source" if ok
          else "\nMISMATCH -- do not train on this")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    if args.verify:
        verify()
    else:
        build()
        print()
        verify()


if __name__ == "__main__":
    main()
