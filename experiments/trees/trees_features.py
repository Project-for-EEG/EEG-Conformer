"""
Tree test, step 1: per-window features from the raw EDF. CPU only.

Rebuilds every window exactly as edf_to_npz.py does (same channel matching, float32,
per-window 0.5-50 Hz band-pass, 4 s windows, 2 s step). Features are taken from the
band-passed window BEFORE the per-window z-score. Check: re-z-scoring the rebuilt
window must reproduce the .npz segment within 1e-4.

Per window and channel, 19 features (float32), in this order:
  size  (8): rms, line length (mean |diff|), mean energy of db4 bands A5 D5 D4 D3 D2 D1
  shape (11): line length / rms, 6 relative band energies, zero crossings of x,
              zero crossings of diff(x), kurtosis, skewness
A channel that is flat zero (missing from the recording) gets NaN.

Run with cwd C:\\EEG. Writes <out>/<CASE>__<stem>.npz and <out>/features_log.txt.
Nothing in C:\\EEG is written.
"""
import argparse
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pywt
from scipy import signal, stats

sys.dont_write_bytecode = True
REPO = Path(r"C:\EEG")
sys.path.insert(0, str(REPO))
from edf_to_npz import (TARGET_CHANNELS, find_matching_channel, annotations_for,  # noqa: E402
                        SAMPLING_RATE, WINDOW, STEP, LOWCUT, HIGHCUT, mne)

PREP = REPO / "preprocessed_data"
RAW = REPO / "raw_data"
CHUNK = 256
NAMES = (["rms", "ll"] + ["E_" + b for b in ("A5", "D5", "D4", "D3", "D2", "D1")]
         + ["nll"] + ["rel_" + b for b in ("A5", "D5", "D4", "D3", "D2", "D1")]
         + ["zc", "zcd", "kurt", "skew"])
FIXED = {}


def ba():
    nyq = 0.5 * SAMPLING_RATE
    return signal.butter(4, [LOWCUT / nyq, HIGHCUT / nyq], btype="band")


def features(F):
    """F: (n, 23, 1024) float64 band-passed windows -> (n, 23, 19) float32."""
    uv = F * 1e6
    rms = np.sqrt(np.mean(uv ** 2, axis=-1))
    d = np.diff(uv, axis=-1)
    ll = np.mean(np.abs(d), axis=-1)
    coeffs = pywt.wavedec(uv, "db4", level=5, axis=-1)          # A5, D5, ..., D1
    E = np.stack([np.mean(c ** 2, axis=-1) for c in coeffs], axis=-1)
    tot = E.sum(-1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        nll = ll / rms
        rel = E / tot
        kurt = stats.kurtosis(uv, axis=-1)
        skew = stats.skew(uv, axis=-1)
    zc = np.sum(np.signbit(uv[..., 1:]) != np.signbit(uv[..., :-1]), axis=-1)
    zcd = np.sum(np.signbit(d[..., 1:]) != np.signbit(d[..., :-1]), axis=-1)
    out = np.concatenate([rms[..., None], ll[..., None], E, nll[..., None], rel,
                          zc[..., None], zcd[..., None], kurt[..., None], skew[..., None]],
                         axis=-1).astype(np.float32)
    out[rms < 1e-6] = np.nan                                      # flat / missing channel
    return out


def one(job):
    case, npz_path, edf_path, out_path = job
    t0 = time.time()
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    data = raw.get_data()
    n_samples = data.shape[1]
    channels = np.zeros((len(TARGET_CHANNELS), n_samples))
    for i, target in enumerate(TARGET_CHANNELS):
        idx = find_matching_channel(target, raw.ch_names)
        if idx is not None:
            channels[i] = data[idx]
    channels = channels.astype(np.float32)
    del data, raw
    starts = list(range(0, n_samples - WINDOW + 1, STEP))

    seizures = annotations_for(case.lower()).get(Path(edf_path).name, [])
    sample_labels = np.zeros(n_samples, dtype=np.int64)
    for s in seizures:
        lo = int(s["start"] * SAMPLING_RATE)
        hi = min(int(s["end"] * SAMPLING_RATE), n_samples)
        if hi > lo:
            sample_labels[lo:hi] = 1
    edf_lab = np.array([1 if np.mean(sample_labels[s:s + WINDOW]) > 0.5 else 0
                        for s in starts], dtype=np.int64)

    with np.load(npz_path) as d:
        ref_seg = d["segments"]
        ref_lab = d["labels"]
    b, a = ba()
    feats, worst = [], 0.0
    if len(starts) != len(ref_lab):
        return case, Path(npz_path).name, len(starts), len(ref_lab), float("nan"), -1, \
            time.time() - t0, "COUNT MISMATCH"
    for j in range(0, len(starts), CHUNK):
        st = starts[j:j + CHUNK]
        W = np.stack([channels[:, s:s + WINDOW] for s in st])        # float32
        F = signal.filtfilt(b, a, W, axis=-1)
        mean = np.mean(F, axis=-1, keepdims=True)
        std = np.std(F, axis=-1, keepdims=True) + 1e-8
        Z = ((F - mean) / std).astype(np.float32)
        worst = max(worst, float(np.max(np.abs(Z - ref_seg[j:j + CHUNK]))))
        feats.append(features(F))
    feats = np.concatenate(feats)
    np.savez(out_path, feats=feats, labels=ref_lab.astype(np.int8),
             names=np.array(NAMES))
    nlab = int(np.sum(edf_lab != ref_lab))
    return case, Path(npz_path).name, len(starts), len(ref_lab), worst, nlab, \
        time.time() - t0, "OK" if worst < 1e-4 else "MISMATCH"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--fixed-chb05-06", required=True)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = []
    for c in sorted(d.name for d in PREP.iterdir() if d.is_dir()):
        for f in sorted((PREP / c).glob("*.npz")):
            stem = f.stem[:-len("_labeled")]
            edf = RAW / c.lower() / (stem + ".edf")
            if stem == "chb05_06":
                edf = Path(args.fixed_chb05_06)
            assert edf.exists(), edf
            jobs.append((c, str(f), str(edf), str(out / ("%s__%s.npz" % (c, f.stem)))))
    if args.limit:
        jobs = jobs[:args.limit]
    print("%d recordings, %d processes" % (len(jobs), args.procs), flush=True)
    t0, lines, bad = time.time(), [], 0
    with Pool(args.procs) as pool:
        for r in pool.imap_unordered(one, jobs):
            case, name, n1, n2, worst, nlab, sec, status = r
            bad += status != "OK"
            line = ("%-6s %-26s windows %6d/%6d  max|dz| %.2e  edf-label diffs %d  %4.0f s  %s"
                    % (case, name, n1, n2, worst, nlab, sec, status))
            lines.append(line)
            print(line, flush=True)
    lines.sort()
    lines.append("%d recordings, %d not OK, %.0f min" % (len(jobs), bad, (time.time() - t0) / 60))
    print(lines[-1])
    (out / "features_log.txt").write_text("\n".join(lines) + "\n", encoding="ascii")


if __name__ == "__main__":
    main()
