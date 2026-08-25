"""
Selective personalisation: fine-tune only the patients the base model fails on.

Blanket fine-tuning is not the right default. Measured at 2 FA/hr with oracle
thresholds, fine-tuning every patient scores 0.693 while choosing the better
model per patient scores 0.819 -- because adaptation actively damages patients
the base model already handled (CHB24 and CHB08 both fell from 1.00 to 0.00).

The choice can be made without peeking at the test period. The base model never
saw the adaptation window, so running it there is legitimate: if it already
detects the patient's own first seizure, keep it; if it misses, personalise.

Reports the honest rule against three references:
  * base everywhere       (no personalisation)
  * tuned everywhere      (blanket personalisation)
  * oracle per patient    (upper bound -- picks with test knowledge)

The gap between the honest rule and the oracle is how much the 1-seizure
decision signal actually costs.

Requires finetune_probs.npz from finetune_per_patient.py.
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from config import ModelConfig
from model import create_model
from evaluate_temporal import find_runs, predict_file
from calibrate_per_patient import score_blocks, pick_threshold, detect
from finetune_per_patient import (patient_files, find_split,
                                  adapt_background_mask, adapt_background_probs)

CACHE = Path("finetune_probs.npz")
CKPT = Path("checkpoints")


def adaptation_window(model, device, files, split):
    """(probs, labels) over the adaptation window only."""
    fi, wi = split
    P, L = [], []
    for i, f in enumerate(files[:fi + 1]):
        probs, lab = predict_file(model, device, f)
        if i == fi:
            probs, lab = probs[:wi], lab[:wi]
        P.append(probs)
        L.append(lab)
    return np.concatenate(P), np.concatenate(L)


def base_handles_patient(probs, labels, thresh, k, m):
    """Does the base model detect the patient's own adaptation seizure?

    One seizure is a thin signal, but it is the only labelled example available
    at decision time, and it is honest -- the base model never trained on it.
    """
    pred = detect(probs, thresh, k, m)
    events = find_runs(labels.astype(bool))
    if not events:
        return True          # nothing to judge on; leave the base model alone
    return all(pred[lo:hi].any() for lo, hi in events)


def best_at(blocks, fa_cap, k, m, flag_cap=0.10):
    best = 0.0
    for t in np.linspace(0.0005, 0.9999, 400):
        s = score_blocks(blocks, t, k, m)
        if s["fa_hr"] <= fa_cap and s["flagged"] <= flag_cap:
            best = max(best, s["sens"])
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target-fa", type=float, default=2.0)
    ap.add_argument("--seizures-for-adapt", type=int, default=1)
    ap.add_argument("--calib-holdout", type=float, default=0.3)
    ap.add_argument("--smooth", type=int, default=5)
    ap.add_argument("--min-consec", type=int, default=3)
    args = ap.parse_args()
    K, M = args.smooth, args.min_consec

    if not CACHE.exists():
        print(f"{CACHE} not found -- run finetune_per_patient.py first.")
        return
    cache = np.load(CACHE, allow_pickle=True)["data"].item()

    owner = {}
    for cp in sorted(CKPT.glob("fold*.pt")):
        ck = torch.load(cp, map_location="cpu", weights_only=False)
        for p in ck["val_patients"]:
            owner[p] = cp

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"decision rule: keep base if it detects the patient's own first "
          f"seizure, else personalise")
    print(f"scored at {args.target_fa} FA/hr with oracle thresholds\n")
    print(f"{'patient':8s} {'ev':>4s} {'chose':>7s} {'base':>6s} {'tuned':>6s} "
          f"{'picked':>7s} {'oracle':>7s}")

    ev_tot = 0
    tot = {"base": 0.0, "tuned": 0.0, "picked": 0.0, "oracle": 0.0}
    right = wrong = 0

    for pat in sorted(cache):
        d = cache[pat]
        ev = score_blocks(d["base"], 0.5, K, M)["events"]
        if ev == 0:
            continue

        files = patient_files(pat)
        split = find_split(files, args.seizures_for_adapt)
        ck = torch.load(owner[pat], map_location=device, weights_only=False)
        base = create_model(ModelConfig(**ck["model_config"])).to(device)
        base.load_state_dict(ck["model_state_dict"])
        base.eval()

        rng = np.random.default_rng(42)
        masks = adapt_background_mask(files, split, args.calib_holdout, rng)
        t_adapt = pick_threshold(
            adapt_background_probs(base, device, files, split, masks),
            args.target_fa, K, M)
        ap_probs, ap_labels = adaptation_window(base, device, files, split)
        keep_base = base_handles_patient(ap_probs, ap_labels, t_adapt, K, M)
        del base
        torch.cuda.empty_cache()

        s_base = best_at(d["base"], args.target_fa, K, M)
        s_tuned = best_at(d["tuned"], args.target_fa, K, M)
        picked = s_base if keep_base else s_tuned
        oracle = max(s_base, s_tuned)

        ev_tot += ev
        tot["base"] += s_base * ev
        tot["tuned"] += s_tuned * ev
        tot["picked"] += picked * ev
        tot["oracle"] += oracle * ev
        right += ev if abs(picked - oracle) < 1e-9 else 0
        wrong += ev if abs(picked - oracle) >= 1e-9 else 0

        flag = "" if abs(picked - oracle) < 1e-9 else "   <-- wrong pick"
        print(f"{pat:8s} {ev:4d} {('base' if keep_base else 'tuned'):>7s} "
              f"{s_base:6.2f} {s_tuned:6.2f} {picked:7.2f} {oracle:7.2f}{flag}")

    print(f"\nAt {args.target_fa} FA/hr over {ev_tot} events:")
    for name in ("base", "tuned", "picked", "oracle"):
        label = {"base": "base everywhere    ",
                 "tuned": "tuned everywhere   ",
                 "picked": "SELECTIVE (honest) ",
                 "oracle": "oracle upper bound "}[name]
        print(f"  {label}: {tot[name]/ev_tot:.3f}")
    print(f"\n  decision was right for {right}/{ev_tot} events "
          f"({100*right/ev_tot:.0f}%)")
    print(f"  cost of the honest rule vs oracle: "
          f"{(tot['picked'] - tot['oracle'])/ev_tot:+.3f}")


if __name__ == "__main__":
    main()
