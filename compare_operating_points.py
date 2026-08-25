"""
Fair comparison: per-patient calibrated thresholds vs the best single global one,
read off at MATCHED false-alarm rates.

The earlier runs were not comparable -- per-patient calibration overshot its
budget (4.0 FA/hr against a 1.0 target), so its higher sensitivity was partly
just spending more alarms. This sweeps the global threshold across its whole
range and reports sensitivity at the FA rates per-patient calibration actually
achieved, so like is compared with like.

Probabilities are computed once and cached, since inference over all 24 patients
dominates the runtime.
"""
import numpy as np, torch
from pathlib import Path
from config import ModelConfig
from model import create_model
from evaluate_temporal import find_runs, predict_file, SECONDS_PER_WINDOW
from calibrate_per_patient import (detect, score_blocks, pick_threshold,
                                   split_calibration, WINDOWS_PER_HOUR)

CACHE = Path("scratch_probs.npz")
K, M, CAL_H = 5, 3, 1.0

def gather():
    if CACHE.exists():
        z = np.load(CACHE, allow_pickle=True)
        return z["data"].item()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = {}
    for ck_path in sorted(Path("checkpoints").glob("fold*.pt")):
        ck = torch.load(ck_path, map_location=device, weights_only=False)
        model = create_model(ModelConfig(**ck["model_config"])).to(device)
        model.load_state_dict(ck["model_state_dict"]); model.eval()
        for pat in ck["val_patients"]:
            data[pat] = [predict_file(model, device, f)
                         for f in sorted((Path("preprocessed_data")/pat).glob("*.npz"))]
        del model; torch.cuda.empty_cache()
    np.savez_compressed(CACHE, data=np.array(data, dtype=object))
    return data

data = gather()
cal_w = int(CAL_H * WINDOWS_PER_HOUR)
splits = {p: split_calibration(v, cal_w) for p, v in data.items()}
all_blocks = [b for p in splits for b in splits[p][1]]

print("GLOBAL THRESHOLD SWEEP (same held-out data)\n")
print(f"  {'thresh':>8s} {'sens':>7s} {'FA/hr':>8s} {'%flagged':>9s}")
curve = []
for t in [0.5, 0.9, 0.99, 0.995, 0.999, 0.9995, 0.9999]:
    s = score_blocks(all_blocks, t, K, M)
    curve.append((s["fa_hr"], s["sens"], t))
    print(f"  {t:8.4f} {s['sens']:7.3f} {s['fa_hr']:8.2f} {100*s['flagged']:8.2f}%")

print("\nPER-PATIENT CALIBRATION at several budgets\n")
print(f"  {'target':>8s} {'sens':>7s} {'FA/hr':>8s}")
pp = []
for target in (0.1, 0.25, 0.5, 1.0, 2.0):
    hit = tot = 0; fa_w = hrs = 0.0
    for p, (cal, test) in splits.items():
        if len(cal) < cal_w // 2 or not test: continue
        t = pick_threshold(cal, target, K, M)
        s = score_blocks(test, t, K, M)
        hit += s["caught"]; tot += s["events"]
        fa_w += s["fa_hr"] * s["hours"]; hrs += s["hours"]
    pp.append((fa_w/hrs, hit/tot, target))
    print(f"  {target:8.2f} {hit/tot:7.3f} {fa_w/hrs:8.2f}")

print("\nMATCHED-FA COMPARISON\n")
print(f"  {'FA/hr':>8s} {'per-patient':>12s} {'global':>8s} {'gain':>8s}")
for fa, sens, _ in pp:
    g = np.interp(fa, [c[0] for c in sorted(curve)], [c[1] for c in sorted(curve)])
    print(f"  {fa:8.2f} {sens:12.3f} {g:8.3f} {sens-g:+8.3f}")
