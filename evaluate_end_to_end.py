"""
The whole pipeline, end to end, with nothing chosen using test data.

Every headline number so far has used an oracle threshold -- picked knowing the
test period. Useful for comparing models, not claimable as performance. This
runs the real thing:

    split chronologically -> calibrate on adaptation background -> decide
    base vs personalised -> apply to the test period -> measure

and reports the false-alarm rate actually achieved, not the one targeted.

Two calibration schemes are compared, because the fixed one is known to drift:

  fixed    a single threshold from the adaptation window. Measured to overshoot
           its budget by 4-9x, because the first hour is not representative of
           the hours that follow.
  rolling  transfer the *percentile* rather than the absolute threshold, and
           re-estimate it each hour from the preceding hour of the recording.
           Unsupervised -- seizures are ~1% of time, so a high percentile of
           recent probabilities tracks the background tail as it moves.

Only patients the base model fails on are personalised, which is both better
(0.747 vs 0.693 at 2 FA/hr in the oracle study) and faster.

Usage:
    python evaluate_end_to_end.py --target-fa 2.0
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from config import ModelConfig
from model import create_model
from evaluate_temporal import find_runs, smooth, predict_file, SECONDS_PER_WINDOW
from calibrate_per_patient import pick_threshold, detect
from finetune_per_patient import (patient_files, find_split, adapt_background_mask,
                                  adapt_background_probs, load_adapt, finetune,
                                  test_blocks)
from selective_personalise import adaptation_window, base_handles_patient

CKPT = Path("checkpoints")
WINDOWS_PER_HOUR = int(3600 / SECONDS_PER_WINDOW)


def apply_min_consec(pred, min_consec):
    if min_consec <= 1:
        return pred
    cleaned = np.zeros_like(pred)
    for lo, hi in find_runs(pred):
        if hi - lo >= min_consec:
            cleaned[lo:hi] = True
    return cleaned


def rolling_thresholds(probs, pct, t0, chunk):
    """Per-window thresholds: hour k uses the pct-th percentile of hour k-1.

    Strictly causal -- a window is never thresholded using anything that comes
    after it. The first chunk falls back to the adaptation-derived threshold,
    since there is no prior hour yet.
    """
    th = np.full(len(probs), t0, dtype=float)
    for start in range(chunk, len(probs), chunk):
        prev = probs[start - chunk:start]
        if len(prev):
            th[start:start + chunk] = np.percentile(prev, pct)
    return th


def score(blocks, thresh, k, m, rolling=None):
    """Event sensitivity / FA per hour. thresh is scalar, or per-window when
    rolling is given."""
    ev_tot = ev_hit = fa = windows = 0
    flagged = 0
    for probs, labels in blocks:
        if rolling is None:
            pred = detect(probs, thresh, k, m)
        else:
            pct, t0, chunk = rolling
            th = rolling_thresholds(probs, pct, t0, chunk)
            pred = apply_min_consec(smooth(probs, k) >= th, m)
        truth = labels.astype(bool)
        windows += len(pred)
        flagged += int(pred.sum())
        events = find_runs(truth)
        ev_tot += len(events)
        ev_hit += sum(1 for lo, hi in events if pred[lo:hi].any())
        for lo, hi in find_runs(pred):
            if not truth[lo:hi].any():
                fa += 1
    hours = windows * SECONDS_PER_WINDOW / 3600
    return {"events": ev_tot, "caught": ev_hit,
            "sens": ev_hit / ev_tot if ev_tot else 0.0,
            "fa_hr": fa / hours if hours else 0.0,
            "flagged": flagged / windows if windows else 0.0,
            "hours": hours}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target-fa", type=float, default=2.0)
    ap.add_argument("--ckpt-glob", type=str, default="fold*.pt",
                    help="which fold checkpoints to evaluate, e.g. "
                         "natural_fold*.pt for the natural-prior models")
    ap.add_argument("--seizures-for-adapt", type=int, default=1)
    ap.add_argument("--calib-holdout", type=float, default=0.3)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--smooth", type=int, default=5)
    ap.add_argument("--min-consec", type=int, default=3)
    ap.add_argument("--roll-hours", type=float, default=1.0)
    ap.add_argument("--cache", type=str, default="endtoend_probs.npz",
                    help="dump chosen-model test blocks + calibration "
                         "background so operating points can be swept offline")
    args = ap.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)
    K, M = args.smooth, args.min_consec
    chunk = int(args.roll_hours * WINDOWS_PER_HOUR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    owner = {}
    for cp in sorted(CKPT.glob(args.ckpt_glob)):
        ck = torch.load(cp, map_location="cpu", weights_only=False)
        for p in ck["val_patients"]:
            owner[p] = cp

    print(f"target {args.target_fa} FA/hr | selective personalisation | "
          f"rolling window {args.roll_hours} h")
    print("nothing below is chosen using test data\n")
    print(f"{'patient':8s} {'ev':>4s} {'model':>6s} {'fixed sens':>11s} "
          f"{'fixedFA':>8s} {'roll sens':>10s} {'rollFA':>8s}")

    agg = {"ev": 0, "fx_hit": 0, "rl_hit": 0,
           "fx_fa": 0.0, "rl_fa": 0.0, "hours": 0.0}
    n_tuned = 0
    cache = {}

    for patient in sorted(owner):
        files = patient_files(patient)
        split = find_split(files, args.seizures_for_adapt)
        if split is None:
            continue

        ck = torch.load(owner[patient], map_location=device, weights_only=False)

        def fresh():
            m = create_model(ModelConfig(**ck["model_config"])).to(device)
            m.load_state_dict(ck["model_state_dict"])
            m.eval()
            return m

        rng = np.random.default_rng(42)
        masks = adapt_background_mask(files, split, args.calib_holdout, rng)

        base = fresh()
        bg = adapt_background_probs(base, device, files, split, masks)
        t = pick_threshold(bg, args.target_fa, K, M)
        ap_probs, ap_labels = adaptation_window(base, device, files, split)
        keep_base = base_handles_patient(ap_probs, ap_labels, t, K, M)

        if keep_base:
            model = base
        else:
            del base
            model = finetune(fresh(), *load_adapt(files, split, masks),
                             device, args.epochs, args.lr)
            bg = adapt_background_probs(model, device, files, split, masks)
            t = pick_threshold(bg, args.target_fa, K, M)
            n_tuned += 1

        blocks = test_blocks(model, device, files, split)
        del model
        torch.cuda.empty_cache()
        if not blocks:
            continue

        cache[patient] = {"blocks": blocks, "bg": bg, "t": t,
                          "model": "base" if keep_base else "tuned"}
        pct = 100.0 * float((bg < t).mean())
        s_fix = score(blocks, t, K, M)
        s_rol = score(blocks, t, K, M, rolling=(pct, t, chunk))

        agg["ev"] += s_fix["events"]
        agg["fx_hit"] += s_fix["caught"]
        agg["rl_hit"] += s_rol["caught"]
        agg["fx_fa"] += s_fix["fa_hr"] * s_fix["hours"]
        agg["rl_fa"] += s_rol["fa_hr"] * s_rol["hours"]
        agg["hours"] += s_fix["hours"]

        print(f"{patient:8s} {s_fix['events']:4d} "
              f"{('base' if keep_base else 'tuned'):>6s} "
              f"{s_fix['sens']:11.2f} {s_fix['fa_hr']:8.2f} "
              f"{s_rol['sens']:10.2f} {s_rol['fa_hr']:8.2f}")

    np.savez_compressed(args.cache, data=np.array(cache, dtype=object))
    print()
    print(f"cached -> {args.cache}")

    e, h = agg["ev"], agg["hours"]
    print(f"\npersonalised {n_tuned} of {len(owner)} patients")
    print(f"tested on {h:.0f} h, {e} events, target {args.target_fa} FA/hr\n")
    print(f"  fixed threshold   : {agg['fx_hit']}/{e} = {agg['fx_hit']/e:.3f} "
          f"sensitivity at {agg['fx_fa']/h:.2f} FA/hr")
    print(f"  rolling percentile: {agg['rl_hit']}/{e} = {agg['rl_hit']/e:.3f} "
          f"sensitivity at {agg['rl_fa']/h:.2f} FA/hr")


if __name__ == "__main__":
    main()
