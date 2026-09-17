"""
Convert the Helsinki neonatal EEG corpus into CHB-MIT's segmented .npz format.

    Stevenson et al., "A dataset of neonatal EEG recordings with seizures
    annotations", doi 10.5281/zenodo.4940267, CC-BY-4.0.
    79 newborns, 111.9 h, three independent expert annotators.

This is the easiest conversion of the three cohorts. Helsinki records at 256 Hz,
the same rate as CHB-MIT, so no resampling is needed, and all 19 electrodes the
bipolar montage requires are present in every recording. Only the referencing
differs: Helsinki stores referential channels ("EEG Fp1-REF") where CHB-MIT
stores differences, and a bipolar derivation is just a subtraction.

    FP1-F7  =  Fp1 - F7        FZ-CZ  =  Fz - Cz
    F7-T7   =  F7  - T3        CZ-PZ  =  Cz - Pz

Two files use "-REF" and the rest "-Ref"; matching is on the letters alone.

THE MISSING CHANNELS

Three CHB-MIT channels use FT9 and FT10 -- inferior-temporal electrodes that
Helsinki, like Siena, does not place. Unlike Siena it also has no F9 or F10, so
the substitution that rescued Siena is unavailable.

    --channels 20      drop them, giving the 20 exactly derivable channels
    --channels 23      substitute the nearest available electrodes

The substitution here is weak and is expected to lose. T7-FT9 becomes T3 - F7,
which is the exact negative of channel 1, and FT10-T8 becomes F8 - T4, the
negative of channel 13. Only FT9-FT10 -> F7 - F8 is a derivation the montage
does not already contain. So two of the three carry no information that is not
already present, and the reason to build it at all is that dropping the
channels was measured to cost 0.109 event sensitivity on CHB-MIT, which is a
large enough penalty to be worth one run to check.

LABELS

Each annotator scored every second of every recording independently, and they
disagree on 47% of the seizure time. That is not noise to be averaged away --
it is the clearest evidence in this project that a seizure boundary is a
clinical judgement rather than a signal transition, which is why hard-negative
mining failed on CHB-MIT.

    --consensus majority    at least 2 of 3 (default; 12.56% of time)
    --consensus unanimous   all 3 agree (9.75%)
    --consensus any         any one annotator (18.33%)

Majority is the default because it neither inflates the positive class with
one reviewer's outliers nor discards the boundary cases entirely.

Usage:
    python helsinki_to_npz.py --verify
    python helsinki_to_npz.py --channels 20 --out preprocessed_data_helsinki20
    python helsinki_to_npz.py --channels 23 --out preprocessed_data_helsinki23
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from scipy import signal

try:
    import mne
    mne.set_log_level("ERROR")
except ImportError:
    raise SystemExit("mne is required: pip install mne")

HELS = Path("additional_data/helsinki")
SAMPLING_RATE = 256
WINDOW = 1024          # 4 s
STEP = 512             # 50% overlap
LOWCUT, HIGHCUT = 0.5, 50.0
MAINS = 50.0           # Finland

ANNOTATORS = ["annotations_2017_A_fixed.csv",
              "annotations_2017_B.csv",
              "annotations_2017_C.csv"]

# CHB-MIT's montage as electrode pairs, in TARGET_CHANNELS order. Old naming on
# the right, because that is what Helsinki writes: T7 = T3, T8 = T4, P7 = T5,
# P8 = T6.
BIPOLAR_20 = [
    ("FP1-F7",  "Fp1", "F7"),     # 0
    ("F7-T7",   "F7",  "T3"),     # 1
    ("T7-P7",   "T3",  "T5"),     # 2
    ("P7-O1",   "T5",  "O1"),     # 3
    ("FP1-F3",  "Fp1", "F3"),     # 4
    ("F3-C3",   "F3",  "C3"),     # 5
    ("C3-P3",   "C3",  "P3"),     # 6
    ("P3-O1",   "P3",  "O1"),     # 7
    ("FP2-F4",  "Fp2", "F4"),     # 8
    ("F4-C4",   "F4",  "C4"),     # 9
    ("C4-P4",   "C4",  "P4"),     # 10
    ("P4-O2",   "P4",  "O2"),     # 11
    ("FP2-F8",  "Fp2", "F8"),     # 12
    ("F8-T8",   "F8",  "T4"),     # 13
    ("T8-P8-0", "T4",  "T6"),     # 14
    ("P8-O2",   "T6",  "O2"),     # 15
    ("FZ-CZ",   "Fz",  "Cz"),     # 16
    ("CZ-PZ",   "Cz",  "Pz"),     # 17
    ("P7-T7",   "T5",  "T3"),     # 18  negative of channel 2
    ("T8-P8-1", "T4",  "T6"),     # 19  duplicate of channel 14
]

# nearest available stand-ins for the FT9/FT10 derivations, inserted at 19-21
APPROX_FT = [
    ("T7-FT9",   "T3", "F7"),     # 19  negative of channel 1
    ("FT9-FT10", "F7", "F8"),     # 20  genuinely new
    ("FT10-T8",  "F8", "T4"),     # 21  negative of channel 13
]


def montage(n_channels):
    if n_channels == 20:
        return BIPOLAR_20
    if n_channels == 23:
        return BIPOLAR_20[:19] + APPROX_FT + BIPOLAR_20[19:]
    raise ValueError("channels must be 20 or 23, got %r" % n_channels)


def electrode_index(name, ch_names):
    """Find a referential electrode. Helsinki writes 'EEG Fp1-Ref' in most
    files and 'EEG Fp1-REF' in 17 of them, so match on letters alone."""
    want = name.lower().replace(" ", "")
    for i, ch in enumerate(ch_names):
        got = (ch.lower().replace("eeg", "").replace("-ref", "")
               .replace(" ", "").strip())
        if got == want:
            return i
    return None


def bandpass(data):
    nyq = 0.5 * SAMPLING_RATE
    b, a = signal.butter(4, [LOWCUT / nyq, HIGHCUT / nyq], btype="band")
    return signal.filtfilt(b, a, data, axis=-1)


def load_annotations():
    """(seconds, babies) arrays, one per annotator. NaN where not recorded."""
    out = []
    header = None
    for name in ANNOTATORS:
        rows = list(csv.reader((HELS / name).open()))
        header = rows[0]
        arr = np.full((len(rows) - 1, len(header)), np.nan, dtype=np.float32)
        for i, row in enumerate(rows[1:]):
            for j, v in enumerate(row[:len(header)]):
                if v.strip() != "":
                    arr[i, j] = float(v)
        out.append(arr)
    return header, np.stack(out)


def second_labels(stack, col, consensus):
    """Per-second 0/1 for one baby under the chosen consensus rule."""
    votes = stack[:, :, col]
    recorded = ~np.isnan(votes[0])
    agree = np.nansum(votes, axis=0)
    if consensus == "unanimous":
        lab = agree == 3
    elif consensus == "any":
        lab = agree >= 1
    else:
        lab = agree >= 2
    return (lab & recorded).astype(np.int64), recorded


def build_bipolar(raw, n_channels):
    data = raw.get_data()
    pairs = montage(n_channels)
    out = np.zeros((len(pairs), data.shape[1]))
    for i, (_, anode, cathode) in enumerate(pairs):
        a = electrode_index(anode, raw.ch_names)
        c = electrode_index(cathode, raw.ch_names)
        if a is None:
            return None, anode
        if c is None:
            return None, cathode
        out[i] = data[a] - data[c]
    return out, None


def convert_one(edf_path, sec_lab, n_channels, notch=True):
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    if abs(raw.info["sfreq"] - SAMPLING_RATE) > 0.5:
        return None, None, "unexpected rate %g Hz" % raw.info["sfreq"]

    bip, missing = build_bipolar(raw, n_channels)
    if bip is None:
        return None, None, "missing electrode " + missing

    info = mne.create_info([n for n, _, _ in montage(n_channels)],
                           SAMPLING_RATE, ch_types="eeg")
    der = mne.io.RawArray(bip, info, verbose="ERROR")
    if notch:
        der.notch_filter(freqs=[MAINS], verbose="ERROR")

    channels = der.get_data().astype(np.float32)
    n_samples = channels.shape[1]

    # per-second annotations to per-sample
    sample_labels = np.zeros(n_samples, dtype=np.int64)
    upto = min(len(sec_lab) * SAMPLING_RATE, n_samples)
    sample_labels[:upto] = np.repeat(sec_lab,
                                     SAMPLING_RATE)[:upto]

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
        labels.append(1 if np.mean(sample_labels[start:start + WINDOW]) > 0.5
                      else 0)

    return np.stack(segments), np.array(labels, dtype=np.int64), None


def verify(n_channels):
    """Check the derivations and that the redundancies CHB-MIT contains hold."""
    edf = sorted(HELS.glob("*.edf"))[0]
    print("verifying on %s at %d channels\n" % (edf.name, n_channels))
    raw = mne.io.read_raw_edf(edf, preload=True, verbose=False)
    bip, missing = build_bipolar(raw, n_channels)
    if bip is None:
        print("FAILED: missing " + missing)
        return False

    data = raw.get_data()
    pairs = montage(n_channels)
    ok = True
    for idx in (0, 5, 16):
        _, a, c = pairs[idx]
        manual = (data[electrode_index(a, raw.ch_names)]
                  - data[electrode_index(c, raw.ch_names)])
        d = float(np.abs(bip[idx] - manual).max())
        print("  channel %2d %-9s max|delta| %.2e  %s"
              % (idx, pairs[idx][0], d, "OK" if d == 0 else "MISMATCH"))
        ok &= d == 0

    neg = float(np.abs(bip[18] + bip[2]).max())
    dup_idx = 22 if n_channels == 23 else 19
    dup = float(np.abs(bip[dup_idx] - bip[14]).max())
    print("  channel 18 == -channel 2    max|delta| %.2e  %s"
          % (neg, "OK" if neg == 0 else "MISMATCH"))
    print("  channel %d == channel 14    max|delta| %.2e  %s"
          % (dup_idx, dup, "OK" if dup == 0 else "MISMATCH"))
    ok &= neg == 0 and dup == 0

    if n_channels == 23:
        # the honest check on the substitution: two of the three are exact
        # negatives of channels already present, and say so out loud
        # ch19 is T3-F7, the exact negative of ch1 (F7-T3).
        # ch21 is F8-T4, the exact DUPLICATE of ch13 (F8-T4) -- testing it as a
        # negative would compare a signal against twice itself and wrongly
        # report that it carries new information.
        n1 = float(np.abs(bip[19] + bip[1]).max())
        d13 = float(np.abs(bip[21] - bip[13]).max())
        print()
        print("  substitute channels, information content:")
        print("    channel 19 == -channel 1   max|delta| %.2e  %s"
              % (n1, "REDUNDANT" if n1 < 1e-12 else "carries something new"))
        print("    channel 21 ==  channel 13  max|delta| %.2e  %s"
              % (d13, "REDUNDANT" if d13 < 1e-12 else "carries something new"))
        print("    channel 20 (F7-F8) is not otherwise in the montage")
        print("    so 2 of the 3 substitutes add nothing the model does not")
        print("    already see; only channel 20 is a new derivation")

    print("\nelectrode coverage across all recordings:")
    need = sorted({e for _, a, c in pairs for e in (a, c)})
    bad = 0
    files = sorted(HELS.glob("*.edf"))
    for f in files:
        r = mne.io.read_raw_edf(f, preload=False, verbose="ERROR")
        absent = [n for n in need if electrode_index(n, r.ch_names) is None]
        if absent:
            print("  %-14s MISSING %s" % (f.name, ", ".join(absent)))
            bad += 1
    if bad == 0:
        print("  all %d recordings carry all %d electrodes" % (len(files),
                                                               len(need)))
    return ok and bad == 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--channels", type=int, default=20, choices=(20, 23))
    ap.add_argument("--consensus", default="majority",
                    choices=("majority", "unanimous", "any"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-notch", action="store_true")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    if args.verify:
        raise SystemExit(0 if verify(args.channels) else 1)

    out_root = Path(args.out or ("preprocessed_data_helsinki%d" % args.channels))
    out_root.mkdir(exist_ok=True)

    header, stack = load_annotations()
    print("converting 79 neonatal recordings to %d-channel CHB-MIT format"
          % args.channels)
    print("consensus rule: %s of 3 annotators" % args.consensus)
    if args.channels == 23:
        print("channels 19-21 substituted from F7/F8; two are negatives of "
              "existing channels")
    print()

    total = seiz = 0
    for col, baby in enumerate(header):
        edf = HELS / ("eeg%s.edf" % baby)
        if not edf.exists():
            print("  eeg%-4s MISSING" % baby)
            continue
        sec_lab, _ = second_labels(stack, col, args.consensus)

        dest_dir = out_root / ("HEL%02d" % int(baby))
        dest_dir.mkdir(exist_ok=True)
        dest = dest_dir / ("eeg%s_labeled.npz" % baby)
        if dest.exists():
            with np.load(dest) as d:
                n, s = len(d["labels"]), int(d["labels"].sum())
            total += n
            seiz += s
            print("  eeg%-4s %6d windows %5d seizure  [have]" % (baby, n, s))
            continue

        seg, lab, err = convert_one(edf, sec_lab, args.channels,
                                    notch=not args.no_notch)
        if err:
            print("  eeg%-4s SKIPPED: %s" % (baby, err))
            continue

        tmp = dest.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, segments=seg, labels=lab)
        tmp.replace(dest)
        total += len(lab)
        seiz += int(lab.sum())
        print("  eeg%-4s %6d windows %5d seizure" % (baby, len(lab),
                                                     int(lab.sum())))

    print("\n%d windows, %d seizure (%.3f%%)"
          % (total, seiz, 100 * seiz / max(total, 1)))
    print("written to %s/" % out_root)


if __name__ == "__main__":
    main()
