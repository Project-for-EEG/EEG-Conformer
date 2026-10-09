"""
The detector against the human experts, on the same babies, scored the same way.

human_ceiling.py found that Helsinki's three expert annotators, each scored
against the other two under SzCORE event rules, produce 6-26 false alarms a
day. That is a ceiling worth knowing, but on its own it compares nothing: the
detector's headline 124 false alarms a day is a CHB-MIT figure, children in an
epilepsy monitoring unit, and the experts were reading newborns in a NICU. A
human number from one cohort against a model number from another says nothing
about either.

So this puts both on one footing.

    babies      only the 46 seizure-bearing babies the 4-fold CV models were
                tested on. Each is scored by the fold model that never saw it.
                The 33 background-only babies are in every fold's training set,
                so the model cannot be fairly scored on them, and the humans
                are restricted to the same 46 to match.
    reference   each PAIR of experts agreeing (A&B, A&C, B&C). A held-out human
                is scored against the other two; the model, having no held-out
                expert, is scored against all three pairs and averaged, so it
                faces exactly the references the humans faced.
    scoring     one score() function for both, so they cannot drift apart.
                SzCORE rules: detections merged under 90 s, references extended
                30 s before and 60 s after, events split at 300 s.
    resolution  one second. The experts scored per second; a model window i
                covers seconds [2i, 2i+4), and a second is detected if any
                window covering it fired.

The humans have no threshold, so each is one point. The model is a curve, and
the comparison that means something is the model's false-alarm rate at the
sensitivity the humans actually achieve.

Usage:
    python human_vs_model.py
    python human_vs_model.py --out human_vs_model.txt
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from config import ModelConfig
from model import create_model
from helsinki_curve import cv_folds
from calibrate_per_patient import detect

ANNOT = Path("additional_data/helsinki")
FILES = ["annotations_2017_A_fixed.csv", "annotations_2017_B.csv",
         "annotations_2017_C.csv"]
ROOT = Path("preprocessed_data_helsinki20")
CKPT = Path("checkpoints")

# SzCORE event rules, in seconds -- the constants score_szcore.py uses.
MERGE, PRE, POST, SPLIT = 90, 30, 60, 300
SMOOTH_K, MIN_CONSEC = 5, 3      # the detection rule the curve was scored with
MAX_FLAGGED = 0.40               # the flagged-time bound helsinki_curve_events uses


def load_annotator(path):
    """{baby_number: per-second 0/1 array}, trimmed at the first blank."""
    rows = list(csv.reader(path.open()))
    header = rows[0]
    out = {}
    for j, name in enumerate(header):
        vals = []
        for r in rows[1:]:
            c = r[j].strip() if j < len(r) else ""
            if c == "":
                break
            vals.append(int(float(c)))
        out[int(name)] = np.array(vals, dtype=np.int8)
    return out


def runs(mask):
    m = np.concatenate(([0], mask.astype(np.int8), [0]))
    d = np.diff(m)
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def merge(events, gap):
    out = []
    for lo, hi in events:
        if out and lo - out[-1][1] < gap:
            out[-1] = (out[-1][0], hi)
        else:
            out.append((lo, hi))
    return out


def split(events, n):
    out = []
    for lo, hi in events:
        while hi - lo > n:
            out.append((lo, lo + n))
            lo += n
        out.append((lo, hi))
    return out


def score(det_mask, ref_mask):
    """(hits, reference events, false alarms) under SzCORE event rules."""
    det = split(merge(runs(det_mask), MERGE), SPLIT)
    ref = split(runs(ref_mask), SPLIT)
    ext = [(max(0, lo - PRE), hi + POST) for lo, hi in ref]
    hit = sum(1 for lo, hi in ext if any(d0 < hi and d1 > lo for d0, d1 in det))
    fa = sum(1 for d0, d1 in det if not any(d0 < hi and d1 > lo for lo, hi in ext))
    return hit, len(ref), fa


def window_probs(model, device, baby, batch=256):
    """Per-window seizure probability for one baby's recording."""
    f = sorted((ROOT / baby).glob("*.npz"))
    if len(f) != 1:
        raise ValueError("%s: expected one recording, found %d" % (baby, len(f)))
    with np.load(f[0]) as d:
        X = d["segments"]
    out = np.empty(len(X), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(X), batch):
            t = torch.from_numpy(X[i:i + batch]).float().to(device)
            out[i:i + len(t)] = torch.softmax(model(t), 1)[:, 1].cpu().numpy()
    return out


def to_seconds(window_pred, n_seconds):
    """Window decisions to one value per second; window i covers [2i, 2i+4)."""
    sec = np.zeros(n_seconds, dtype=np.int8)
    for i in np.flatnonzero(window_pred):
        sec[2 * i:min(2 * i + 4, n_seconds)] = 1
    return sec


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-folds", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    A, B, C = (load_annotator(ANNOT / f) for f in FILES)
    experts = {"A": A, "B": B, "C": C}
    pairs = {"A&B": ("A", "B"), "A&C": ("A", "C"), "B&C": ("B", "C")}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    folds = cv_folds(args.n_folds, args.seed)

    # model probabilities, each baby from the fold model that never saw it
    probs, length = {}, {}
    for k, test in enumerate(folds):
        ck = torch.load(CKPT / ("helscv%d_67.pt" % k), map_location=device,
                        weights_only=False)
        model = create_model(ModelConfig(**ck["model_config"])).to(device)
        model.load_state_dict(ck["model_state_dict"])
        model.eval()
        overlap = set(test) & set(ck.get("train_patients", []))
        if overlap:
            raise AssertionError("fold %d model trained on its test babies: %s"
                                 % (k, sorted(overlap)))
        for baby in test:
            n = int(baby.replace("HEL", ""))
            length[baby] = min(len(A[n]), len(B[n]), len(C[n]))
            probs[baby] = window_probs(model, device, baby)
        del model
        torch.cuda.empty_cache()

    babies = sorted(probs)
    seconds = sum(length.values())
    days = seconds / 86400.0
    lines = []

    def say(s=""):
        print(s)
        lines.append(s)

    say("humans against the detector, Helsinki, same babies, same scoring")
    say("%d babies (every seizure-bearing baby, each scored by the fold model that"
        % len(babies))
    say("never saw it), %.1f h. SzCORE event rules at 1 s resolution." % (seconds / 3600))
    say()

    # humans: each expert held out against the other two agreeing
    say("HUMANS -- each expert scored against the other two agreeing")
    humans = []
    for held, (p, q) in [("A", ("B", "C")), ("B", ("A", "C")), ("C", ("A", "B"))]:
        H = R = F = 0
        for baby in babies:
            n = int(baby.replace("HEL", ""))
            L = length[baby]
            ref = experts[p][n][:L] & experts[q][n][:L]
            h, r, f = score(experts[held][n][:L], ref)
            H += h
            R += r
            F += f
        sens, fad = H / R, F / days
        humans.append((held, sens, fad))
        say("  expert %s vs %s&%s   sensitivity %.3f   %6.1f false alarms/day"
            % (held, p, q, sens, fad))
    say()

    # model: swept, against all three pairs, averaged
    allp = np.concatenate(list(probs.values()))
    grid = np.unique(np.concatenate([np.linspace(0.02, 0.99, 120),
                                     np.quantile(allp, np.linspace(0.5, 0.9999, 240))]))
    curve = []
    for t in grid:
        sens_pairs, fa_pairs = [], []
        flagged = total = 0
        dets = {}
        for baby in babies:
            wp = detect(probs[baby].astype(np.float64), float(t), SMOOTH_K, MIN_CONSEC)
            dets[baby] = to_seconds(wp, length[baby])
            flagged += int(wp.sum())
            total += len(wp)
        if flagged / total > MAX_FLAGGED:
            continue
        for name, (p, q) in pairs.items():
            H = R = F = 0
            for baby in babies:
                n = int(baby.replace("HEL", ""))
                L = length[baby]
                h, r, f = score(dets[baby], experts[p][n][:L] & experts[q][n][:L])
                H += h
                R += r
                F += f
            sens_pairs.append(H / R)
            fa_pairs.append(F / days)
        curve.append((float(np.mean(fa_pairs)), float(np.mean(sens_pairs)), float(t)))
    curve.sort()

    say("DETECTOR -- the fold models, scored against each expert pair and averaged")
    say("  %d operating points survive the flagged-time bound" % len(curve))
    for cap in (2, 5, 10, 25, 50, 100, 250):
        ok = [c for c in curve if c[0] <= cap]
        if ok:
            fa, se, t = max(ok, key=lambda c: c[1])
            say("  <= %3d FA/day   sensitivity %.3f   (achieved %6.1f FA/day)" % (cap, se, fa))
    say()

    say("THE COMPARISON -- what the detector needs to match each human's sensitivity")
    for held, hs, hf in humans:
        ok = [c for c in curve if c[1] >= hs]
        if ok:
            fa, se, t = min(ok, key=lambda c: c[0])
            say("  expert %s: %.3f at %5.1f FA/day | detector reaches %.3f at %6.1f FA/day (%.0fx)"
                % (held, hs, hf, se, fa, fa / hf if hf else float("inf")))
        else:
            best = max(curve, key=lambda c: c[1]) if curve else None
            say("  expert %s: %.3f at %5.1f FA/day | detector NEVER reaches it -- best %.3f at %.1f FA/day"
                % (held, hs, hf, best[1] if best else float("nan"),
                   best[0] if best else float("nan")))
    say()
    say("Restricted to babies the model was never trained on. The human figures here")
    say("are therefore on 46 babies, not the 79 in human_ceiling.py, and will differ.")

    if args.out:
        Path(args.out).write_text("\n".join(lines) + "\n", encoding="ascii")
        print("\nsaved %s" % args.out)


if __name__ == "__main__":
    main()
