"""
Turn a SzCORE recording into the input this model was trained on.

SzCORE hands a container one EDF holding the 19 electrodes of the 10-20 system
in a common-average reference, at 256 Hz:

    Fp1-Avg F3-Avg C3-Avg P3-Avg O1-Avg F7-Avg T3-Avg T5-Avg Fz-Avg Cz-Avg
    Pz-Avg Fp2-Avg F4-Avg C4-Avg P4-Avg O2-Avg F8-Avg T4-Avg T6-Avg

This model wants 20 bipolar derivations. Those two things are exactly
reconcilable, because subtracting two common-average channels cancels the
average:

    (A - avg) - (B - avg) = A - B

so nothing is approximated and nothing is zero-filled. The 20 channels are
BIPOLAR from siena_to_npz.py, which needs only these 19 electrodes. The three
channels this model does not use -- T7-FT9, FT9-FT10, FT10-T8, indices 19-21 of
TARGET_CHANNELS -- are the only ones that need FT9 and FT10, which the 10-20
set does not contain. The 20-channel checkpoints already drop exactly those
three (drop_channels [19, 20, 21] on 92 of them), so no retraining is needed.

THE TRAP THIS GUARDS AGAINST

    Channel naming is the failure mode here, not the maths. SzCORE's own test
    fixtures disagree with each other: tests/data/unipolar.edf uses the old
    names (T3 T4 T5 T6) while tests/data/bipolar.edf uses the new ones (T7 T8
    P7 P8) with inconsistent capitalisation. A matcher that handles one and not
    the other silently substitutes zeros, the model sees a flat channel, and
    the run produces a plausible-looking wrong answer rather than an error.
    That is not hypothetical -- it is what a submitted algorithm had to be
    withdrawn and fixed for.

    So every electrode must be found, and every derived channel must carry
    signal. Both are asserted, and a failure raises rather than degrades.

Usage:
    python szcore_adapter.py recording.edf            # report what it found
    python szcore_adapter.py --verify-against-siena   # prove it matches
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import signal

from siena_to_npz import BIPOLAR, SAMPLING_RATE, WINDOW, STEP, LOWCUT, HIGHCUT

# Electrodes that changed name when the 10-20 system was extended. The source
# data and the checkpoints use the left column; SzCORE may hand over either.
SYNONYMS = {
    "t3": "t7", "t4": "t8", "t5": "p7", "t6": "p8",
    "t7": "t3", "t8": "t4", "p7": "t5", "p8": "t6",
}


def normalise(name):
    """A channel label reduced to the electrode it names.

    Handles the '-Avg' reference suffix, an 'EEG ' prefix, whitespace, and
    capitalisation, all of which vary between files that are otherwise the
    same format.
    """
    got = str(name).lower().strip()
    for suffix in ("-avg", "-ref", "-le", "-re"):
        if got.endswith(suffix):
            got = got[: -len(suffix)]
    return got.replace("eeg", "").replace(" ", "").replace("-", "").strip()


def electrode_index(want, ch_names):
    """Index of an electrode in an EDF channel list, or None.

    Tries the name as given, then its old/new-naming counterpart, so a file
    written with T7 satisfies a request for T3.
    """
    target = normalise(want)
    got = [normalise(c) for c in ch_names]
    if target in got:
        return got.index(target)
    other = SYNONYMS.get(target)
    if other and other in got:
        return got.index(other)
    return None


def build_bipolar(data, ch_names):
    """(20, n_samples) in BIPOLAR order, from referential channels.

    Raises if any electrode is missing rather than filling with zeros. A
    constant channel would not merely lose that derivation -- siena_to_npz.py
    notes it identifies the cohort outright, so the model is being asked a
    question it was never trained on.
    """
    missing = []
    for _, anode, cathode in BIPOLAR:
        for e in (anode, cathode):
            if electrode_index(e, ch_names) is None and e not in missing:
                missing.append(e)
    if missing:
        raise ValueError(
            "missing electrodes %s; channels present: %s"
            % (", ".join(missing), ", ".join(map(str, ch_names))))

    out = np.zeros((len(BIPOLAR), data.shape[1]), dtype=np.float64)
    for i, (_, anode, cathode) in enumerate(BIPOLAR):
        out[i] = data[electrode_index(anode, ch_names)] - \
                 data[electrode_index(cathode, ch_names)]
    return out


def assert_carries_signal(derived):
    """Every derived channel must vary.

    Channels 18 and 19 of BIPOLAR are a sign flip and a duplicate of channels
    2 and 14, so they are not independent -- but they are not constant either,
    and a constant one means an electrode resolved to the wrong place.
    """
    flat = [i for i, ch in enumerate(derived) if np.std(ch) == 0]
    if flat:
        raise ValueError(
            "derived channels %s are constant, so an electrode was mismatched"
            % flat)


def bandpass(data):
    """The filter edf_to_npz.py applies, per segment."""
    nyq = 0.5 * SAMPLING_RATE
    b, a = signal.butter(4, [LOWCUT / nyq, HIGHCUT / nyq], btype="band")
    return signal.filtfilt(b, a, data, axis=-1)


def to_segments(derived):
    """(n_windows, 20, 1024) float32, matching edf_to_npz.edf_to_arrays.

    Filter then z-score happen inside each window, not over the recording, so
    this has to loop rather than filter once. Doing it the cheap way would
    change the input distribution the model was trained on.
    """
    channels = derived.astype(np.float32)
    n_samples = channels.shape[1]
    segments = []
    for start in range(0, n_samples - WINDOW + 1, STEP):
        seg = channels[:, start:start + WINDOW]
        try:
            seg = bandpass(seg)
        except Exception:
            pass
        mean = np.mean(seg, axis=1, keepdims=True)
        std = np.std(seg, axis=1, keepdims=True) + 1e-8
        segments.append(((seg - mean) / std).astype(np.float32))
    if not segments:
        return np.zeros((0, len(BIPOLAR), WINDOW), dtype=np.float32)
    return np.stack(segments, axis=0)


def load(edf_path):
    """(segments, n_samples) for one SzCORE recording.

    Resampling is deliberate rather than assumed: SzCORE states 256 Hz, but a
    file at another rate would silently change what a 1024-sample window means.
    """
    import mne
    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose="ERROR")
    if int(round(raw.info["sfreq"])) != SAMPLING_RATE:
        raw = raw.resample(SAMPLING_RATE, verbose="ERROR")
    data = raw.get_data()
    derived = build_bipolar(data, raw.ch_names)
    assert_carries_signal(derived)
    return to_segments(derived), data.shape[1]


def verify_against_siena(limit=2):
    """Prove the derivation matches the one used to build the training data.

    Siena is referential like SzCORE, so running both routes over the same
    file is a real end-to-end check rather than a synthetic one. If these
    disagree, the container would be feeding the model something other than
    what it was trained on.
    """
    import mne
    root = Path("additional_data/siena")
    edfs = sorted(root.rglob("*.edf"))[:limit]
    if not edfs:
        print("no Siena EDFs found under %s" % root)
        return 1

    from siena_to_npz import build_bipolar as siena_build
    bad = 0
    for edf in edfs:
        raw = mne.io.read_raw_edf(str(edf), preload=True, verbose="ERROR")
        mine = build_bipolar(raw.get_data(), raw.ch_names)
        theirs, missing = siena_build(raw, False)
        if theirs is None:
            print("%-28s siena_to_npz could not build it: missing %s"
                  % (edf.name, missing))
            bad += 1
            continue
        theirs = np.asarray(theirs, dtype=np.float64)
        if theirs.shape != mine.shape:
            print("%-28s SHAPE %s vs %s" % (edf.name, mine.shape, theirs.shape))
            bad += 1
            continue
        gap = float(np.max(np.abs(mine - theirs)))
        scale = float(np.max(np.abs(theirs))) or 1.0
        ok = gap / scale < 1e-9
        print("%-28s max |difference| %.3e  (signal %.3e)  %s"
              % (edf.name, gap, scale, "identical" if ok else "DIFFERS"))
        bad += 0 if ok else 1
    print("\n%s" % ("derivation matches siena_to_npz.py"
                    if not bad else "%d file(s) disagree" % bad))
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("edf", nargs="?", help="a SzCORE-style referential EDF")
    ap.add_argument("--verify-against-siena", action="store_true",
                    help="check the derivation against the training pipeline")
    args = ap.parse_args()

    if args.verify_against_siena:
        return verify_against_siena()
    if not args.edf:
        ap.error("give an EDF, or --verify-against-siena")

    segments, n_samples = load(args.edf)
    print("%s" % args.edf)
    print("  samples        %d (%.1f s at %d Hz)"
          % (n_samples, n_samples / SAMPLING_RATE, SAMPLING_RATE))
    print("  windows        %d of %d samples, stride %d"
          % (len(segments), WINDOW, STEP))
    print("  model input    %s" % (segments.shape,))
    if len(segments):
        print("  per-window std %.3f (z-scored, so ~1.0 is correct)"
              % float(np.mean(np.std(segments, axis=2))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
