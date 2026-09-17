"""
Score the patient-count curve on events, not on window AUC.

helsinki_curve.py trains four models on 8, 16, 32 and 67 Helsinki patients
against a fixed held-out test set, and reports window-level AUC. That metric
has misread three decisions in this project: it called dropping three channels
free when the event cost was 0.109, called the Siena merge harmful when it
helped, and called CBraMod equal when it catches 17 fewer seizures.

So the curve gets scored the way a detector is judged: how many seizure events
are caught, and how often it alarms on background.

Each model is swept across thresholds and read at MATCHED false-alarm rates,
because the four models are not calibrated to each other and comparing them at
a shared 0.5 would compare four different operating points rather than four
training set sizes.

The flagged-time bound is applied throughout. Without it a threshold low enough
to alarm continuously merges every prediction into one run per recording, which
overlaps every seizure and books almost no separate false alarms -- scoring
perfect sensitivity while being useless. That trap has caught this project
twice.

Usage:
    python helsinki_curve_events.py
"""
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from config import ModelConfig
from model import create_model
from evaluate_temporal import find_runs, SECONDS_PER_WINDOW
from calibrate_per_patient import detect
from helsinki_curve import split_patients

ROOT = Path("preprocessed_data_helsinki20")
CKPT = Path("checkpoints")
SIZES = [8, 16, 32, 67]
K, M = 5, 3
MAX_FLAGGED = 0.40      # neonates are 29.7% seizure here, so the bound is
                        # looser than CHB-MIT's 0.20 and still well under a
                        # detector that simply alarms continuously


def probs_for(model, device, patient, batch=256):
    """(probs, labels) per recording, in time order."""
    out = []
    for f in sorted((ROOT / patient).glob("*.npz")):
        with np.load(f) as d:
            X, y = d["segments"], d["labels"]
        p = np.empty(len(X), dtype=np.float32)
        with torch.no_grad():
            for i in range(0, len(X), batch):
                t = torch.from_numpy(X[i:i + batch]).float().to(device)
                p[i:i + len(t)] = torch.softmax(model(t), 1)[:, 1].cpu().numpy()
        out.append((p, y))
    return out


def score(blocks, thresh):
    ev = hit = fa = win = flag = 0
    for probs, labels in blocks:
        pred = detect(probs, thresh, K, M)
        truth = labels.astype(bool)
        win += len(pred)
        flag += int(pred.sum())
        events = find_runs(truth)
        ev += len(events)
        hit += sum(1 for lo, hi in events if pred[lo:hi].any())
        for lo, hi in find_runs(pred):
            if not truth[lo:hi].any():
                fa += 1
    hours = win * SECONDS_PER_WINDOW / 3600
    return ev, hit, fa / hours, flag / win


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test, _, _ = split_patients(42)
    print("scoring on %d held-out Helsinki babies, never trained on\n"
          % len(test))

    curves = {}
    for n in SIZES:
        path = CKPT / ("helscurve_%d.pt" % n)
        if not path.exists():
            print("missing %s" % path.name)
            continue
        ck = torch.load(path, map_location=device, weights_only=False)
        model = create_model(ModelConfig(**ck["model_config"])).to(device)
        model.load_state_dict(ck["model_state_dict"])
        model.eval()

        blocks = []
        for p in test:
            blocks.extend(probs_for(model, device, p))

        pts = []
        for t in np.linspace(0.02, 0.99, 50):
            ev, hit, fa_hr, flagged = score(blocks, t)
            if flagged <= MAX_FLAGGED and ev:
                pts.append((fa_hr, hit / ev, t))
        curves[n] = sorted(pts)
        best = max(pts, key=lambda x: x[1]) if pts else None
        print("%2d patients (%2d with seizures): %d events, best sensitivity "
              "%.3f at %.1f FA/h"
              % (n, ck["n_train_with_seizures"],
                 score(blocks, 0.5)[0], best[1] if best else float("nan"),
                 best[0] if best else float("nan")))
        del model
        torch.cuda.empty_cache()

    def at(pts, target):
        ok = [p for p in pts if p[0] <= target]
        return max(ok, key=lambda p: p[1])[1] if ok else None

    print("\nevent sensitivity at matched false-alarm rates")
    targets = [2, 5, 10, 20, 40]
    print("%-10s " % "patients" + " ".join("%8s" % ("%d FA/h" % t)
                                           for t in targets))
    rows = []
    for n in SIZES:
        if n not in curves:
            continue
        vals = [at(curves[n], t) for t in targets]
        rows.append((n, vals))
        print("%-10d " % n + " ".join(
            "%8s" % ("%.3f" % v if v is not None else "-") for v in vals))

    print()
    for i, t in enumerate(targets):
        seq = [(n, v[i]) for n, v in rows if v[i] is not None]
        if len(seq) >= 2:
            first, last = seq[0], seq[-1]
            print("  at %2d FA/h: %d patients %.3f -> %d patients %.3f  (%+.3f)"
                  % (t, first[0], first[1], last[0], last[1],
                     last[1] - first[1]))

    out = Path("cv_results") / ("helsinki_curve_events_%s.json"
                                % datetime.now().strftime("%Y%m%d_%H%M%S"))
    out.write_text(json.dumps(
        {"test_patients": test,
         "curves": {str(n): [[float(a), float(b), float(c)] for a, b, c in v]
                    for n, v in curves.items()}}, indent=2))
    print("\nsaved %s" % out)


if __name__ == "__main__":
    main()
