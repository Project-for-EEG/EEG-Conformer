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

        # A 50-point linear grid was too coarse and silently misreported this
        # curve. The 8-patient model had no operating point between 4.31 and
        # 10.80 FA/h, so its "5 FA/h" and "10 FA/h" readings were the same
        # point twice, and the 8-to-67 gain at 10 FA/h came out as +0.405 when
        # it was really comparing 4.3 FA/h against 8.2. The alarm rate is a step
        # function of the threshold and every step sits at some observed
        # probability, so the grid is drawn from the data as well: quantiles of
        # the probabilities concentrate points exactly where the rate moves.
        probs_all = np.concatenate([b[0] for b in blocks])
        grid = np.unique(np.concatenate([
            np.linspace(0.02, 0.99, 200),
            np.quantile(probs_all, np.linspace(0.50, 0.9999, 400)),
        ]))
        pts = []
        for t in grid:
            ev, hit, fa_hr, flagged = score(blocks, float(t))
            if flagged <= MAX_FLAGGED and ev:
                pts.append((fa_hr, hit / ev, float(t)))
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
        """(sensitivity, the rate actually achieved) under a budget.

        The achieved rate is returned and printed, not just the sensitivity.
        Hiding it is what let a comparison between 4.3 and 8.2 FA/h be reported
        as if both sat at 10.
        """
        ok = [p for p in pts if p[0] <= target]
        if not ok:
            return None, None
        best = max(ok, key=lambda p: p[1])
        return best[1], best[0]

    print("\nevent sensitivity under a false-alarm BUDGET")
    print("each cell is sensitivity and the rate actually achieved, because a")
    print("budget is a ceiling and two models under the same ceiling can sit at")
    print("very different rates")
    targets = [2, 5, 10, 20, 40]
    print("\n%-10s " % "patients" + " ".join("%16s" % ("<= %d FA/h" % t)
                                            for t in targets))
    rows = []
    for n in SIZES:
        if n not in curves:
            continue
        vals = [at(curves[n], t) for t in targets]
        rows.append((n, vals))
        print("%-10d " % n + " ".join(
            "%16s" % ("%.3f @ %.1f" % (s, a) if s is not None else "-")
            for s, a in vals))

    print()
    for i, t in enumerate(targets):
        seq = [(n, v[i]) for n, v in rows if v[i][0] is not None]
        if len(seq) < 2:
            continue
        (n0, (s0, a0)), (n1, (s1, a1)) = seq[0], seq[-1]
        # Only call it a matched comparison when the two achieved rates are
        # close. Otherwise the difference mixes training size with budget.
        matched = abs(a1 - a0) <= 0.25 * max(a0, a1, 1e-9)
        print("  under %2d FA/h: %d patients %.3f @ %.1f -> %d patients "
              "%.3f @ %.1f  (%+.3f)%s"
              % (t, n0, s0, a0, n1, s1, a1, s1 - s0,
                 "" if matched else "   NOT MATCHED, rates differ by %.0f%%"
                 % (100 * abs(a1 - a0) / max(a0, a1))))

    out = Path("cv_results") / ("helsinki_curve_events_%s.json"
                                % datetime.now().strftime("%Y%m%d_%H%M%S"))
    out.write_text(json.dumps(
        {"test_patients": test,
         "curves": {str(n): [[float(a), float(b), float(c)] for a, b, c in v]
                    for n, v in curves.items()}}, indent=2))
    print("\nsaved %s" % out)


if __name__ == "__main__":
    main()
