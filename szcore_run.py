"""
Run the detector over one SzCORE recording and write the events file.

This is what the submitted container executes. SzCORE mounts a directory of
recordings read-only, hands over one EDF at a time, and expects a TSV of
detected events back:

    docker run -v ./data:/data -v /tmp:/output \
        -e INPUT=rec.edf -e OUTPUT=rec.tsv <image>

THE OPERATING POINT IS NOT THE ONE THIS PROJECT NORMALLY USES

    Every threshold elsewhere in this repo maximises event sensitivity at a
    tolerated false-alarm rate. SzCORE reports per-subject event F1, which
    prices a false alarm against a caught seizure, and the two objectives
    disagree sharply. retune_f1.py measures the gap on the cached
    leave-one-patient-out predictions:

        threshold 0.45  ->  F1 0.233   sensitivity 0.986   198 FA/day
        threshold 0.92  ->  F1 0.501   sensitivity 0.815    52 FA/day

    The stricter setting more than doubles F1 while catching fewer seizures.
    It was chosen on CHB-MIT and checked on Siena, where it scores 0.509
    against the 0.587 of a setting tuned on Siena itself -- so it is the
    setting that holds up on both cohorts rather than the highest single
    number. Dianalund is held out and cannot be tuned on at all, so a setting
    that transfers is worth more than one that peaks.

WHY THE MASK IS BUILT AT SAMPLE RATE AND NOT AT WINDOW RATE

    The scorer sizes its comparison from recordingDuration on the first row,
    and its duration tolerance defaults to zero. Predictions come one per 512
    samples, and a recording does not divide evenly into windows: an hour at
    256 Hz yields 1799 windows covering 3598 s, not 3600 s. Reporting the
    windowed length would disagree with the reference by two seconds, the
    scorer would raise, and the caller substitutes an all-zero mask -- so the
    recording scores zero with no error raised anywhere. The mask therefore
    spans every sample of the file, and each detected window marks the samples
    it actually covers.

Usage:
    python szcore_run.py input.edf output.tsv
    python szcore_run.py input.edf output.tsv --checkpoint checkpoints/deploy.pt
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

from szcore_adapter import load as load_recording
from siena_to_npz import SAMPLING_RATE, WINDOW, STEP

# From retune_f1.py on endtoend_lopo20siena.npz: 24 patients and 190 seizure
# events, every one predicted by a checkpoint that never trained on that
# patient, with no per-patient adaptation, which is the condition the
# container runs under.
#
# MIN_CONSEC is 11 rather than 3, and that single parameter is most of the
# score. It is how many consecutive windows must fire before a seizure is
# called: 3 windows is 6 s, 11 is 22 s. Short bursts are overwhelmingly chewing
# and movement, so requiring a longer run removes them.
#
#     3  ->  F1 0.300, sensitivity 0.687, 75.7 false alarms/day
#    11  ->  F1 0.420, sensitivity 0.460,  7.7 false alarms/day
#
# It is a trade, not a free gain: about 44 of 190 real seizures are given up.
# That is the right trade for the leaderboard metric and the wrong one for a
# clinical alarm, and the submission says so rather than leaving the reader to
# assume this is the configuration the project recommends.
#
# An earlier tuning on endtoend_lopo.npz chose 3, and was wrong twice over:
# 5 of its 24 patients used per-patient adapted models, which the container
# cannot do, and its predictions came from a different checkpoint family.
THRESHOLD = 0.92
SMOOTHING = 1
MIN_CONSEC = 11

DEFAULT_CHECKPOINT = "checkpoints/szcore_deploy.pt"
BATCH = 256


def find_runs(mask):
    """Maximal runs of True, as (start, end) exclusive."""
    if not mask.any():
        return []
    padded = np.concatenate(([False], mask, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(edges[::2], edges[1::2]))


def smooth(probs, k):
    """Moving average over k windows, edges handled by shrinking the window."""
    if k <= 1 or len(probs) < k:
        return probs
    kernel = np.ones(k, dtype=np.float64)
    num = np.convolve(probs, kernel, mode="same")
    den = np.convolve(np.ones_like(probs, dtype=np.float64), kernel, mode="same")
    return num / den


def probabilities(segments, checkpoint):
    """Seizure probability per window.

    CPU is the assumption. Whether the evaluation host has a GPU is not
    documented on the current submission page -- the figures that mention one
    belong to the 2025 challenge -- so this must not require it.
    """
    import torch
    from config import ModelConfig
    from model import create_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(checkpoint, map_location=device, weights_only=False)
    model = create_model(ModelConfig(**ck["model_config"])).to(device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()

    expected = ck["model_config"]["num_channels"]
    if segments.shape[1] != expected:
        raise ValueError(
            "checkpoint expects %d channels, the recording produced %d. The "
            "23-channel models need FT9 and FT10, which the 10-20 set SzCORE "
            "provides does not contain -- submit a 20-channel checkpoint."
            % (expected, segments.shape[1]))

    out = np.empty(len(segments), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(segments), BATCH):
            batch = torch.from_numpy(segments[i:i + BATCH]).float().to(device)
            out[i:i + len(batch)] = torch.softmax(model(batch), 1)[:, 1].cpu().numpy()
    return out


def detect(probs):
    """Per-window seizure decisions at the tuned operating point."""
    pred = smooth(probs, SMOOTHING) >= THRESHOLD
    if MIN_CONSEC > 1:
        cleaned = np.zeros_like(pred)
        for lo, hi in find_runs(pred):
            if hi - lo >= MIN_CONSEC:
                cleaned[lo:hi] = True
        pred = cleaned
    return pred


def sample_mask(window_pred, n_samples):
    """Window decisions expanded to one value per sample of the recording.

    A window covers WINDOW samples starting at its own offset, so a detected
    window marks that whole span. The mask is the length of the file, which is
    what makes the reported recordingDuration the real one.
    """
    mask = np.zeros(n_samples, dtype=bool)
    for i in np.flatnonzero(window_pred):
        start = i * STEP
        mask[start:min(start + WINDOW, n_samples)] = True
    return mask


def write_events(mask, out_path):
    """The TSV SzCORE reads back.

    epilepsy2bids writes the header from its own dataclass, which is the same
    string the repository's CI checks with a literal regex, and emits the
    single background row when nothing was detected. Hand-writing this file
    invites a header that differs by one column name.
    """
    from epilepsy2bids.annotations import Annotations

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Annotations.loadMask(mask, SAMPLING_RATE).saveTsv(str(out_path))
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", nargs="?", default=None)
    ap.add_argument("output", nargs="?", default=None)
    ap.add_argument("--checkpoint", default=os.environ.get(
        "CHECKPOINT", DEFAULT_CHECKPOINT))
    args = ap.parse_args()

    # The container is driven by environment variables; the positional form is
    # for running it by hand.
    inp = args.input or os.path.join("/data", os.environ.get("INPUT", ""))
    out = args.output or os.path.join("/output", os.environ.get("OUTPUT", ""))
    if not args.input and not os.environ.get("INPUT"):
        ap.error("give an input EDF, or set INPUT")

    segments, n_samples = load_recording(inp)
    if len(segments) == 0:
        # Shorter than one window. SzCORE guarantees at least a minute, but a
        # header-only output would be scored as an all-zero mask rather than
        # rejected, so the background row is written explicitly.
        write_events(np.zeros(n_samples, dtype=bool), out)
        print("%s: %.1f s, shorter than one window -- wrote background only"
              % (inp, n_samples / SAMPLING_RATE))
        return 0

    probs = probabilities(segments, args.checkpoint)
    pred = detect(probs)
    mask = sample_mask(pred, n_samples)
    path = write_events(mask, out)

    events = find_runs(mask)
    print("%s -> %s" % (inp, path))
    print("  %.1f s, %d windows, %d detected, %d event(s)"
          % (n_samples / SAMPLING_RATE, len(segments), int(pred.sum()),
             len(events)))
    for lo, hi in events[:10]:
        print("    %.1f s for %.1f s" % (lo / SAMPLING_RATE,
                                         (hi - lo) / SAMPLING_RATE))
    return 0


if __name__ == "__main__":
    sys.exit(main())
