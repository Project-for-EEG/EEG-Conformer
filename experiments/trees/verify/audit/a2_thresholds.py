"""Audit step 2: thresholds_for() vs pick_threshold() once per target, on the real A0
calibration backgrounds of all 23 folds, rebuilt independently from the cached logits.
Also compares to the saved thresholds_a.json (A0, all 14 targets). Read-only."""
import json
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np

REPO = Path(r"C:\EEG")
TR = Path(r"C:\Users\user\AppData\Local\Temp\claude\36c1f29744f93fe6"
          r"\b3889703-629c-4c5a-b583-69f722743ff4\scratchpad\trees")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(TR))
from threadpoolctl import threadpool_limits                          # noqa: E402
from finetune_per_patient import patient_files, find_split, adapt_background_mask  # noqa
from calibrate_per_patient import pick_threshold                      # noqa: E402
import trees_train                                                    # noqa: E402

OLD = Path(r"C:\Users\user\AppData\Local\Temp\claude\36c1f29744f93fe6"
           r"\f241281f-add4-41ac-9210-f47dab0b83b0\scratchpad")
PILOT = ["CHB12", "CHB15", "CHB24", "CHB13", "CHB06", "CHB16"]


def main():
    saved = json.loads((TR / "out" / "thresholds_a.json").read_text())["A0"]
    folds = sorted(p.stem[5:] for p in (REPO / "checkpoints" / "lopo").glob("lopo_*.pt"))
    cases = sorted(d.name for d in (REPO / "preprocessed_data").iterdir() if d.is_dir())
    files = {c: patient_files(c) for c in cases}
    splits = {c: find_split(files[c], 1) for c in cases}
    worst_eq, worst_saved, n_cmp, n_one = 0.0, 0.0, 0, 0
    diffs = []
    t0 = time.time()
    for name in folds:
        held = name.split("_")
        train = [c for c in cases if c not in held]
        rng = np.random.default_rng(42)
        pooled = []
        for q in train:
            masks = adapt_background_mask(files[q], splits[q], 0.3, rng)
            fi, wi = splits[q]
            res, allb = [], []
            for i, f in enumerate(files[q][:fi + 1]):
                cdir = OLD / ("pilot_cache" if name in PILOT else "pilot_cache2")
                with np.load(cdir / ("%s__%s.npz" % (q, f.stem))) as d:
                    lg = d["logit_" + name]
                    lab = d["labels"]
                p = 1.0 / (1.0 + np.exp(-np.asarray(lg, np.float64)))
                if i == fi:
                    p, lab = p[:wi], lab[:wi]
                bg = lab == 0
                res.append(p[masks[i] & bg])
                allb.append(p[bg])
            h = np.concatenate(res)
            pooled.append(np.asarray(h if len(h) >= 200 else np.concatenate(allb), np.float32))
        bg = np.concatenate(pooled)
        tf = trees_train.thresholds_for(bg)
        for tg in trees_train.TARGETS:
            pt = pick_threshold(bg, tg, 5, 3)
            worst_eq = max(worst_eq, abs(pt - tf[tg]))
            worst_saved = max(worst_saved, abs(pt - saved[held[0]][str(tg)]))
            n_cmp += 1
            n_one += pt == 1.0
            if pt != tf[tg]:
                diffs.append((name, tg, pt, tf[tg]))
        print("%-12s bg %6d  t@0.05 %.4f  t@2 %.4f  t@5 %.4f  [%.0f s]"
              % (name, len(bg), tf[0.05], tf[2.0], tf[5.0], time.time() - t0), flush=True)
    print("compared %d (fold, target) pairs; thresholds equal to 1.0: %d" % (n_cmp, n_one))
    print("max |pick_threshold - thresholds_for| = %.3e; exact mismatches %d %s"
          % (worst_eq, len(diffs), diffs[:5]))
    print("max |pick_threshold - saved thresholds_a.json A0| = %.3e" % worst_saved)


if __name__ == "__main__":
    with threadpool_limits(1):
        main()
