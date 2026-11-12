"""Audit step 5: feature checks. (a) synthetic sines through trees_features.features();
(b) NaN / inf / range audit over every saved feats file; (c) independent recompute of all
19 features for two real recordings straight from EDF. Read-only outside this folder."""
import sys
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np

REPO = Path(r"C:\EEG")
TR = Path(r"C:\Users\user\AppData\Local\Temp\claude\36c1f29744f93fe6"
          r"\b3889703-629c-4c5a-b583-69f722743ff4\scratchpad\trees")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(TR))
import trees_features as tf                                                  # noqa: E402
from scipy import signal                                                     # noqa: E402

FEATS = TR / "feats"
NAMES = tf.NAMES
print("feature order:", list(enumerate(NAMES)))
print("SHAPE cols 8..18 ->", [NAMES[i] for i in range(8, 19)])

# (a) synthetic sines, 50 uV amplitude, in volts, after the same band-pass
t = np.arange(1024) / 256.0
b, a = tf.ba()
print("\n(a) sine tests: freq -> dominant band, rms(uV), zc, zcd, nll")
for f in (2.0, 6.0, 12.0, 24.0, 45.0):
    x = 50e-6 * np.sin(2 * np.pi * f * t + 0.3)
    W = np.tile(x, (1, 23, 1)).astype(np.float32)
    F = signal.filtfilt(b, a, W, axis=-1)
    o = tf.features(F)[0, 0]
    rel = o[9:15]
    print("  %5.1f Hz -> %s (rel %s)  rms %.2f  zc %d (expect ~%d)  zcd %d  nll %.3f"
          % (f, NAMES[9 + int(np.argmax(rel))], np.round(rel, 3), o[0], o[15], 2 * f * 4,
             o[16], o[8]))
W = np.zeros((1, 23, 1024), np.float32)
print("  flat zero channel -> all NaN:", bool(np.isnan(tf.features(signal.filtfilt(b, a, W, axis=-1))).all()))

# (b) audit over every saved file
print("\n(b) NaN audit per recording (channels fully NaN / channels partly NaN)")
tot = nan_w = part_ch = inf_n = mixed = 0
rms_all = []
for p in sorted(FEATS.glob("*.npz")):
    with np.load(p) as d:
        X = d["feats"]
    tot += X.shape[0]
    nanw = np.isnan(X).any(-1)                         # (n, 23)
    alln = np.isnan(X).all(-1)
    mixed += int((nanw & ~alln).sum())                 # some but not all of 19 NaN
    inf_n += int(np.isinf(X).sum())
    full = np.flatnonzero(nanw.all(0))
    part = np.flatnonzero(nanw.any(0) & ~nanw.all(0))
    if len(full) or len(part):
        print("  %-28s n %5d  full-NaN ch %s  partial-NaN ch %s (windows %s)"
              % (p.stem, X.shape[0], full.tolist(), part.tolist(),
                 [int(nanw[:, c].sum()) for c in part]))
    part_ch += len(part)
    rms_all.append(np.nanmedian(X[:, :, 0], axis=0))
print("  windows %d; windows-with-mixed-NaN %d; inf values %d" % (tot, mixed, inf_n))
r = np.concatenate(rms_all)
print("  per-recording median RMS (uV) across channels: 5/50/95%% = %s"
      % np.round(np.nanpercentile(r, [5, 50, 95]), 1))


# (c) independent recompute
def indep(F):
    """F (n,23,1024) band-passed volts -> (n,23,19), written separately from tf.features."""
    import pywt
    from scipy import stats
    x = F * 1e6
    n, C, L = x.shape
    out = np.full((n, C, 19), np.nan)
    for i in range(n):
        for c in range(C):
            s = x[i, c]
            rms = np.sqrt(np.mean(s * s))
            if rms < 1e-6:
                continue
            ds = s[1:] - s[:-1]
            ll = np.abs(ds).mean()
            cs = pywt.wavedec(s, pywt.Wavelet("db4"), mode="symmetric", level=5)
            names = ["A5", "D5", "D4", "D3", "D2", "D1"]
            assert len(cs) == 6
            E = np.array([np.mean(cc ** 2) for cc in cs])
            sgn = s < 0
            zc = int(np.count_nonzero(sgn[1:] ^ sgn[:-1]))
            sg2 = ds < 0
            zcd = int(np.count_nonzero(sg2[1:] ^ sg2[:-1]))
            m = s.mean()
            m2 = np.mean((s - m) ** 2)
            kurt = np.mean((s - m) ** 4) / m2 ** 2 - 3.0
            skew = np.mean((s - m) ** 3) / m2 ** 1.5
            out[i, c] = np.r_[rms, ll, E, ll / rms, E / E.sum(), zc, zcd, kurt, skew]
    return out


import mne                                                                   # noqa: E402
from edf_to_npz import TARGET_CHANNELS, find_matching_channel                # noqa: E402
targets = [("CHB01", "chb01_03"), ("CHB04", "chb04_05")]
for p in sorted(FEATS.glob("*.npz")):
    with np.load(p) as d:
        if np.isnan(d["feats"]).all(-1).all(0).any():
            targets.append(tuple(p.stem.replace("_labeled", "").split("__")))
            break
print("\n(c) independent recompute from EDF:", targets)
for case, stem in targets:
    raw = mne.io.read_raw_edf(REPO / "raw_data" / case.lower() / (stem + ".edf"),
                              preload=True, verbose=False)
    data = raw.get_data()
    ch = np.zeros((23, data.shape[1]))
    for i, tgt in enumerate(TARGET_CHANNELS):
        idx = find_matching_channel(tgt, raw.ch_names)
        if idx is not None:
            ch[i] = data[idx]
    ch = ch.astype(np.float32)
    starts = list(range(0, data.shape[1] - 1024 + 1, 512))
    pick = np.unique(np.r_[0, 1, len(starts) // 2, len(starts) - 1,
                           np.random.default_rng(0).choice(len(starts), 40, replace=False)])
    Wn = np.stack([ch[:, starts[k]:starts[k] + 1024] for k in pick])
    F = signal.filtfilt(b, a, Wn, axis=-1)
    mine = indep(F)
    with np.load(FEATS / ("%s__%s_labeled.npz" % (case, stem))) as d:
        saved = d["feats"][pick].astype(np.float64)
    both_nan = np.isnan(mine) == np.isnan(saved)
    rel = np.abs(mine - saved) / (np.abs(mine) + 1e-9)
    worst = np.nanmax(np.where(np.isnan(mine), 0, rel), axis=(0, 1))
    print("  %s %s: %d windows, NaN pattern equal %s; max rel diff per feature:"
          % (case, stem, len(pick), bool(both_nan.all())))
    print("   " + " ".join("%s %.1e" % (NAMES[k], worst[k]) for k in range(19)))
