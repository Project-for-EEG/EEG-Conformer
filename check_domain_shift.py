"""
How different does Siena look from CHB-MIT after conversion?

Merging two cohorts only helps if the model sees them as the same kind of
signal. If it can tell them apart, it can learn cohort-specific shortcuts, and
"more patients" turns into "two datasets sharing a network".

The conversion already removes the obvious differences: sampling rate, montage,
electrode naming, and the 50 Hz mains that Italian recordings carry and
American ones do not. Per-segment z-scoring removes amplitude scale. What is
left is whatever the recording equipment and the patient population contribute.

Three things are measured, all on the *converted, z-scored* windows the model
would actually receive:

  1. Band power per channel. Distinct spectra mean distinct signal, and the
     50 Hz bin says directly whether the notch did its job.
  2. Cross-channel correlation structure. Montage geometry is shared by
     construction, so a large gap here means something physical differs.
  3. A cohort classifier. The direct question: can a model trained to separate
     CHB-MIT from Siena windows do it? AUC near 0.5 means the cohorts are
     genuinely mixed. AUC near 1.0 means a seizure model could use cohort
     identity as a feature, and the merge needs care.

The classifier is the one that matters. The first two explain its answer.

Usage:
    python check_domain_shift.py                 # after siena_to_npz.py
    python check_domain_shift.py --n 4000
"""
import argparse
from pathlib import Path

import numpy as np
from scipy import signal
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

CHB = Path("preprocessed_data")
SIENA = Path("preprocessed_data_siena")
SAMPLING_RATE = 256
BANDS = [("delta", 0.5, 4), ("theta", 4, 8), ("alpha", 8, 13),
         ("beta", 13, 30), ("gamma", 30, 50), ("mains", 48, 52)]

# CHB-MIT indices 19/20/21 are the FT9/FT10 derivations Siena cannot rebuild
DROP = [19, 20, 21]


def sample_windows(root, n, rng, drop=None):
    """n random windows drawn across every patient under a root."""
    files = sorted(root.glob("*/*.npz"))
    if not files:
        raise SystemExit("no .npz under %s" % root)
    per_file = max(1, n // len(files))
    out = []
    for f in files:
        with np.load(f) as d:
            seg = d["segments"]
            if len(seg) == 0:
                continue
            take = min(per_file, len(seg))
            idx = rng.choice(len(seg), size=take, replace=False)
            X = seg[np.sort(idx)]
        if drop:
            keep = [c for c in range(X.shape[1]) if c not in drop]
            X = X[:, keep, :]
        out.append(X)
        if sum(len(o) for o in out) >= n:
            break
    X = np.concatenate(out)[:n]
    return X


def band_power(X):
    """(n, channels, bands) relative band power via Welch."""
    f, P = signal.welch(X, fs=SAMPLING_RATE, nperseg=256, axis=-1)
    total = P.sum(axis=-1, keepdims=True) + 1e-12
    feats = []
    for _, lo, hi in BANDS:
        m = (f >= lo) & (f < hi)
        feats.append(P[..., m].sum(axis=-1) / total[..., 0])
    return np.stack(feats, axis=-1)


def corr_structure(X):
    """Mean absolute cross-channel correlation per window."""
    out = np.empty(len(X))
    for i, w in enumerate(X):
        c = np.corrcoef(w)
        iu = np.triu_indices_from(c, k=1)
        out[i] = np.nanmean(np.abs(c[iu]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=3000,
                    help="windows sampled from each cohort")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--siena", default=str(SIENA),
                    help="converted Siena root to compare against")
    ap.add_argument("--keep-ft", action="store_true",
                    help="keep CHB-MIT channels 19-21 instead of dropping "
                         "them; use with a 23-channel Siena conversion")
    ap.add_argument("--per-channel", action="store_true",
                    help="also report how separable each channel is on its "
                         "own, which isolates whether the approximated "
                         "FT9/FT10 derivations are a cohort giveaway")
    args = ap.parse_args()

    siena = Path(args.siena)
    if not siena.exists():
        raise SystemExit("run siena_to_npz.py first to create %s" % siena)

    rng = np.random.RandomState(args.seed)
    print("sampling %d windows from each cohort" % args.n)
    print("  CHB-MIT %s" % CHB)
    print("  Siena   %s" % siena)
    Xc = sample_windows(CHB, args.n, rng, drop=None if args.keep_ft else DROP)
    Xs = sample_windows(siena, args.n, rng)
    print("  CHB-MIT %s   Siena %s\n" % (Xc.shape, Xs.shape))
    if Xc.shape[1:] != Xs.shape[1:]:
        raise SystemExit("shape mismatch: %s vs %s" % (Xc.shape, Xs.shape))

    bc, bs = band_power(Xc), band_power(Xs)
    print("relative band power, averaged over channels and windows")
    print("  %-7s %9s %9s %9s" % ("band", "CHB-MIT", "Siena", "ratio"))
    for i, (name, _, _) in enumerate(BANDS):
        a, b = bc[..., i].mean(), bs[..., i].mean()
        print("  %-7s %9.4f %9.4f %9.2fx" % (name, a, b, b / (a + 1e-12)))
    print("  (mains overlaps gamma; it is the 50 Hz check, not a separate band)")

    cc, cs = corr_structure(Xc[:500]), corr_structure(Xs[:500])
    print("\nmean |cross-channel correlation|")
    print("  CHB-MIT %.3f +/- %.3f" % (cc.mean(), cc.std()))
    print("  Siena   %.3f +/- %.3f" % (cs.mean(), cs.std()))

    # the direct test: separate the cohorts from band-power features
    X = np.concatenate([bc.reshape(len(bc), -1), bs.reshape(len(bs), -1)])
    y = np.concatenate([np.zeros(len(bc)), np.ones(len(bs))])
    cut = rng.rand(len(y)) < 0.7
    clf = HistGradientBoostingClassifier(max_iter=200, random_state=args.seed)
    clf.fit(X[cut], y[cut])
    auc = roc_auc_score(y[~cut], clf.predict_proba(X[~cut])[:, 1])

    print("\ncohort classifier (band power -> which dataset)")
    print("  AUC %.4f" % auc)
    if auc > 0.95:
        verdict = ("the cohorts are trivially separable. A seizure model can "
                   "use cohort identity as a feature, so train on the merge "
                   "but report on CHB-MIT patients only, and expect the gain "
                   "to come from diversity rather than transfer.")
    elif auc > 0.75:
        verdict = ("the cohorts are distinguishable but overlapping. Usable, "
                   "with the same reporting caution.")
    else:
        verdict = "the cohorts are well mixed; a plain merge is reasonable."
    print("  " + verdict)

    if args.per_channel:
        print("")
        print("per-channel separability (AUC from that channel alone)")
        print("  a channel that separates the cohorts on its own is a"
              " giveaway the model can latch onto")
        order = []
        for ch in range(bc.shape[1]):
            Xch = np.concatenate([bc[:, ch, :], bs[:, ch, :]])
            c = HistGradientBoostingClassifier(max_iter=120,
                                               random_state=args.seed)
            c.fit(Xch[cut], y[cut])
            a = roc_auc_score(y[~cut], c.predict_proba(Xch[~cut])[:, 1])
            order.append((a, ch))
        for a, ch in sorted(order, reverse=True):
            mark = "  <- approximated" if ch in (19, 20, 21) else ""
            print("  channel %2d  AUC %.4f%s" % (ch, a, mark))
        approx = [a for a, ch in order if ch in (19, 20, 21)]
        rest = [a for a, ch in order if ch not in (19, 20, 21)]
        if approx:
            print("")
            print("  approximated channels mean AUC %.4f, others %.4f"
                  % (sum(approx) / len(approx), sum(rest) / len(rest)))
            if sum(approx) / len(approx) > sum(rest) / len(rest) + 0.03:
                print("  the substituted channels are more separable than the"
                      " exact ones; they leak cohort identity")
            else:
                print("  the substitution is no more separable than the exact"
                      " channels, so it adds no new cohort signal")

    print("\nnote: a high AUC here is expected and is not a reason to abandon "
          "the merge.\nDifferent hospitals, equipment and patient ages produce "
          "different EEG.\nIt is a reason to evaluate held-out CHB-MIT "
          "patients separately from held-out\nSiena patients, so a gain on one "
          "is not hidden by a loss on the other.")


if __name__ == "__main__":
    main()
