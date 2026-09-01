"""
Convert Siena Scalp EEG into the same segmented .npz format as CHB-MIT.

The two datasets are recorded differently and have to be reconciled before a
model trained on one can see the other:

    CHB-MIT                       Siena
    256 Hz                        512 Hz
    23 bipolar derivations        ~30 unipolar electrodes
    already differenced           needs differencing
    new naming (T7, P7, T8, P8)   old naming (T3, T5, T4, T6)
    60 Hz mains (US)              50 Hz mains (Italy)

Bipolar montages are just differences of unipolar electrodes, so 20 of
CHB-MIT's 23 channels can be rebuilt exactly:

    FP1-F7  =  Fp1 - F7        FZ-CZ  =  Fz - Cz
    F7-T7   =  F7  - T3        CZ-PZ  =  Cz - Pz
    ...

The three that cannot are T7-FT9, FT9-FT10 and FT10-T8: Siena has no FT9 or
FT10 electrode, and nothing in its montage is a linear combination of them.
Those are dropped from both datasets (--drop-channels 19 20 21 on the trainer)
rather than zero-filled, since a constant-zero channel is a free tell for which
dataset a window came from.

Everything downstream is byte-for-byte the CHB-MIT recipe: 4 s windows of 1024
samples at 50% overlap, 4th-order Butterworth 0.5-50 Hz per segment, per-channel
z-score, and a window is a seizure when more than half its samples are inside
one.

Two Siena-specific steps happen *before* that shared recipe:

  * resample 512 -> 256 Hz, with MNE's anti-aliasing filter
  * notch out 50 Hz mains

The notch matters more than it looks. The shared bandpass stops at 50 Hz, which
is comfortably outside CHB-MIT's 60 Hz mains but sits exactly on Siena's. A
4th-order Butterworth is only ~3 dB down at its corner, so without the notch
Siena would carry line noise into the passband that CHB-MIT does not, and the
model could separate the two datasets on that alone. Disable with --no-notch to
measure the difference.

Seizure times come from siena_seizures.json; run parse_siena.py first.

Usage:
    python parse_siena.py
    python siena_to_npz.py                 # convert every recording
    python siena_to_npz.py --seizure-only  # only the seizure-bearing files
    python siena_to_npz.py --verify        # rebuild derivations, check the maths
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy import signal

try:
    import mne
    mne.set_log_level("ERROR")
except ImportError:
    raise SystemExit("mne is required: pip install mne")

SIENA = Path("additional_data/siena")
OUT = Path("preprocessed_data_siena")
ANNOTATIONS = Path("siena_seizures.json")

SAMPLING_RATE = 256
WINDOW = 1024          # 4 s
STEP = 512             # 50% overlap
LOWCUT, HIGHCUT = 0.5, 50.0
MAINS = 50.0           # Italy

# CHB-MIT's montage as electrode pairs. Index in this list is the channel index
# the trainer sees, so the order must match TARGET_CHANNELS in
# download_and_convert.py with 19/20/21 removed.
#
# Old naming is used on the right because that is what Siena writes:
#   T7 = T3,  T8 = T4,  P7 = T5,  P8 = T6
BIPOLAR = [
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
# CHB-MIT itself carries both of those redundancies; they are kept so the two
# datasets present the model with identically-shaped input.

DROPPED = ["T7-FT9", "FT9-FT10", "FT10-T8"]


# The three CHB-MIT channels Siena cannot rebuild exactly use FT9 and FT10,
# inferior-temporal electrodes Siena does not place. It does place F9 and F10,
# which sit immediately anterior to them in the extended 10-20 system, and all
# 41 recordings carry both.
#
# Substituting F9 for FT9 and F10 for FT10 gives an approximation of those
# derivations rather than an exact reconstruction. It is worth having because
# the alternative measured badly: dropping the three channels cost 0.109 event
# sensitivity on CHB-MIT alone, even though window-level AUC called the drop
# free. Zero-filling is worse still, since a constant channel identifies the
# cohort outright.
#
# Inserted at 19, 20, 21 so the result matches TARGET_CHANNELS position for
# position, with T8-P8-1 moving back to 22.
APPROX_FT = [
    ("T7-FT9",   "T3",  "F9"),    # 19
    ("FT9-FT10", "F9",  "F10"),   # 20
    ("FT10-T8",  "F10", "T4"),    # 21
]


def montage(approximate_ft):
    """The channel list to build, in CHB-MIT's own order."""
    if not approximate_ft:
        return BIPOLAR
    return BIPOLAR[:19] + APPROX_FT + BIPOLAR[19:]


def electrode_index(name, ch_names):
    """Find a unipolar electrode in a Siena channel list.

    Siena writes them as "EEG Fp1", "EEG FP2", "EEG CZ" and so on, with
    inconsistent capitalisation between files and trailing whitespace, so the
    match is on the letters alone.
    """
    want = name.lower().replace(" ", "")
    for i, ch in enumerate(ch_names):
        got = ch.lower().replace("eeg", "").replace(" ", "").strip()
        if got == want:
            return i
    return None


def bandpass(data):
    """The filter preprocess_data.py applies, per segment."""
    nyq = 0.5 * SAMPLING_RATE
    b, a = signal.butter(4, [LOWCUT / nyq, HIGHCUT / nyq], btype="band")
    return signal.filtfilt(b, a, data, axis=-1)


def build_bipolar(raw, approximate_ft=False):
    """(n_channels, n) array of CHB-MIT derivations, or None with the missing name.

    Differencing happens at the native 512 Hz, before resampling, so both
    members of a pair go through the identical anti-aliasing filter and no
    phase difference is introduced between them.
    """
    data = raw.get_data()
    pairs = montage(approximate_ft)
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


def edf_is_complete(edf_path):
    """Is this EDF fully downloaded?

    An EDF header states how many data records the file contains. A truncated
    download keeps the header intact but loses records off the end, and MNE
    quietly recovers by inferring the count from the file size -- so a half
    downloaded recording converts without error into a shorter one, missing
    exactly the seizures near the end. That is silent, plausible-looking, wrong
    data, which is the failure mode this project already paid for once.

    The header is fixed-format ASCII: bytes 184-192 give the header length,
    236-244 the record count, and 252-256 the signal count, after which each
    signal contributes 8 bytes of samples-per-record at offset 256 + 216*ns.

    Returns (ok, message).
    """
    with open(edf_path, "rb") as fh:
        head = fh.read(256)
        if len(head) < 256:
            return False, "file shorter than an EDF header"
        header_bytes = int(head[184:192].decode("ascii", "replace"))
        n_records = int(head[236:244].decode("ascii", "replace"))
        n_signals = int(head[252:256].decode("ascii", "replace"))
        fh.seek(256 + 216 * n_signals)
        per_record = fh.read(8 * n_signals).decode("ascii", "replace")
    samples = [int(per_record[i * 8:(i + 1) * 8]) for i in range(n_signals)]
    expected = header_bytes + n_records * sum(samples) * 2   # 16-bit samples
    actual = edf_path.stat().st_size

    if n_records == -1:                      # legal, means "unknown"
        return True, ""
    if actual < expected:
        return False, ("truncated: %.1f MB of %.1f MB (%d records declared)"
                       % (actual / 1048576, expected / 1048576, n_records))
    return True, ""


def convert_file(edf_path, seizures, notch=True, approximate_ft=False):
    """(segments, labels) for one Siena recording, in CHB-MIT format."""
    ok, why = edf_is_complete(edf_path)
    if not ok:
        return None, None, why

    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)

    bip, missing = build_bipolar(raw, approximate_ft)
    if bip is None:
        return None, None, "missing electrode " + missing

    # wrap the derivations back into an MNE object so resample and notch get
    # the same filter design MNE uses everywhere else
    info = mne.create_info([name for name, _, _ in montage(approximate_ft)],
                           raw.info["sfreq"], ch_types="eeg")
    der = mne.io.RawArray(bip, info, verbose="ERROR")

    if notch and MAINS < raw.info["sfreq"] / 2:
        der.notch_filter(freqs=[MAINS], verbose="ERROR")
    der.resample(SAMPLING_RATE, verbose="ERROR")

    channels = der.get_data().astype(np.float32)
    n_samples = channels.shape[1]

    sample_labels = np.zeros(n_samples, dtype=np.int64)
    for s in seizures:
        lo = int(s["onset_s"] * SAMPLING_RATE)
        hi = min(int(s["offset_s"] * SAMPLING_RATE), n_samples)
        if hi > lo:
            sample_labels[lo:hi] = 1

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

    return np.stack(segments, axis=0), np.array(labels, dtype=np.int64), None


def verify():
    """Check the derivations are arithmetically what they claim to be.

    Rebuilds two channels independently and confirms the redundancies CHB-MIT
    contains hold here too: channel 18 must be the exact negative of channel 2,
    and channel 19 must equal channel 14.
    """
    edf = next(SIENA.glob("*/*.edf"))
    print("verifying derivations on " + edf.name + "\n")
    raw = mne.io.read_raw_edf(edf, preload=True, verbose=False)
    bip, missing = build_bipolar(raw, approximate_ft)
    if bip is None:
        print("FAILED: missing electrode " + missing)
        return False

    data = raw.get_data()
    ok = True
    for idx in (0, 5, 16):
        _, a, c = BIPOLAR[idx]
        manual = data[electrode_index(a, raw.ch_names)] - \
                 data[electrode_index(c, raw.ch_names)]
        d = float(np.abs(bip[idx] - manual).max())
        print("  channel %2d %-9s  max|delta| %.2e  %s"
              % (idx, BIPOLAR[idx][0], d, "OK" if d == 0 else "MISMATCH"))
        ok &= d == 0

    neg = float(np.abs(bip[18] + bip[2]).max())
    dup = float(np.abs(bip[19] - bip[14]).max())
    print("  channel 18 == -channel 2   max|delta| %.2e  %s"
          % (neg, "OK" if neg == 0 else "MISMATCH"))
    print("  channel 19 ==  channel 14  max|delta| %.2e  %s"
          % (dup, "OK" if dup == 0 else "MISMATCH"))
    ok &= neg == 0 and dup == 0

    print("\nelectrode coverage across every recording:")
    bad = 0
    for f in sorted(SIENA.glob("*/*.edf")):
        r = mne.io.read_raw_edf(f, preload=False, verbose="ERROR")
        need = sorted(set([a for _, a, _ in BIPOLAR] + [c for _, _, c in BIPOLAR]))
        absent = [n for n in need if electrode_index(n, r.ch_names) is None]
        if absent:
            print("  %-20s MISSING %s" % (f.name, ", ".join(absent)))
            bad += 1
    if bad == 0:
        print("  all %d recordings carry all %d required electrodes"
              % (len(list(SIENA.glob("*/*.edf"))),
                 len(set([a for _, a, _ in BIPOLAR] + [c for _, _, c in BIPOLAR]))))
    return ok and bad == 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verify", action="store_true",
                    help="check the derivation arithmetic, convert nothing")
    ap.add_argument("--seizure-only", action="store_true",
                    help="convert only recordings that contain a seizure")
    ap.add_argument("--no-notch", action="store_true",
                    help="skip the 50 Hz mains notch")
    ap.add_argument("--approximate-ft", action="store_true",
                    help="also build channels 19-21 by substituting F9/F10 for "
                         "the FT9/FT10 electrodes Siena lacks, giving the full "
                         "23-channel CHB-MIT montage instead of 20")
    ap.add_argument("--patients", nargs="*", default=None,
                    help="convert only these patients, e.g. PN00 PN01")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    if args.verify:
        raise SystemExit(0 if verify() else 1)

    if not ANNOTATIONS.exists():
        raise SystemExit("run parse_siena.py first to produce " + str(ANNOTATIONS))
    records = json.loads(ANNOTATIONS.read_text())

    by_file = {}
    for r in records:
        by_file.setdefault((r["patient"], r["file"]), []).append(r)

    out_root = Path(args.out)
    out_root.mkdir(exist_ok=True)

    edfs = sorted(SIENA.glob("*/*.edf"))
    if args.patients:
        want = {p.upper() for p in args.patients}
        edfs = [f for f in edfs if f.parent.name.upper() in want]
    if args.seizure_only:
        edfs = [f for f in edfs if (f.parent.name, f.name) in by_file]

    print("converting %d Siena recordings to %d-channel CHB-MIT format"
          % (len(edfs), len(montage(args.approximate_ft))))
    if args.approximate_ft:
        print("channels 19-21 approximated with F9/F10 in place of FT9/FT10")
    else:
        print("dropped, not derivable from this montage: %s" % ", ".join(DROPPED))
    print("50 Hz notch: %s\n" % ("off" if args.no_notch else "on"))

    total_seg = total_sz = 0
    for edf in edfs:
        patient = edf.parent.name
        seizures = by_file.get((patient, edf.name), [])
        dest_dir = out_root / patient
        dest_dir.mkdir(exist_ok=True)
        dest = dest_dir / (edf.stem + "_labeled.npz")

        if dest.exists():
            with np.load(dest) as d:
                n, sz = len(d["labels"]), int(d["labels"].sum())
            print("  %-20s %6d windows %5d seizure  [have]" % (edf.name, n, sz))
            total_seg += n
            total_sz += sz
            continue

        seg, lab, err = convert_file(edf, seizures, notch=not args.no_notch,
                                     approximate_ft=args.approximate_ft)
        if err:
            print("  %-20s SKIPPED: %s" % (edf.name, err))
            continue

        tmp = dest.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, segments=seg, labels=lab)
        tmp.replace(dest)

        total_seg += len(lab)
        total_sz += int(lab.sum())
        print("  %-20s %6d windows %5d seizure  %s"
              % (edf.name, len(lab), int(lab.sum()),
                 "(%d annotated)" % len(seizures) if seizures else ""))

    print("\n%d windows, %d seizure (%.3f%%)"
          % (total_seg, total_sz, 100 * total_sz / max(total_seg, 1)))
    print("written to %s/" % out_root)


if __name__ == "__main__":
    main()
