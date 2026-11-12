"""critic step C (Q4): why the A0 threshold transfers badly. For every fold model, FA/h at the
fold's deployable thresholds on four background sets, with the SAME run counter
(calibrate_per_patient.fa_per_hour_on_background):
  seen-scattered   = the calibration set itself (training patients, reserved 30%, rng 42, scattered)
  seen-contiguous  = the same training patients' whole pre-split background, scored as contiguous runs
  unseen-scattered = held-out patient's pre-split background, random 30% scatter (rng 42)
  unseen-contig    = held-out patient's pre-split background, contiguous
  test-contig      = held-out patient's post-split background, contiguous (same counter)
Also re-derives the thresholds from the scatter and checks them against out/thresholds_a.json.
Also: extended targets below 0.05 FA/h, to see whether any deployable A0 point reaches 25/day.
Run with cwd C:\\EEG. CPU, numpy only."""
import sys
sys.dont_write_bytecode = True
import json
from pathlib import Path
import numpy as np
sys.path.insert(0, r"C:\EEG")
from finetune_per_patient import patient_files, find_split, adapt_background_mask  # noqa: E402
from calibrate_per_patient import fa_per_hour_on_background                         # noqa: E402
from score_official import count                                                    # noqa: E402
from evaluate_temporal import find_runs                                             # noqa: E402

TREES = Path(r"C:\Users\user\AppData\Local\Temp\claude\36c1f29744f93fe6"
             r"\b3889703-629c-4c5a-b583-69f722743ff4\scratchpad\trees")
ME = TREES / "verify" / "critic"
OLD = Path(r"C:\Users\user\AppData\Local\Temp\claude\36c1f29744f93fe6"
           r"\f241281f-add4-41ac-9210-f47dab0b83b0\scratchpad")
CACHE1, CACHE2 = OLD / "pilot_cache", OLD / "pilot_cache2"
PILOT = ["CHB12", "CHB15", "CHB24", "CHB13", "CHB06", "CHB16"]
REPO = Path(r"C:\EEG")
TARGETS = (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0)
EXTRA = (0.002, 0.005, 0.01, 0.02, 0.03)
SHOW = (0.05, 0.2, 0.5, 2.0, 5.0)
K, NR = 5, 3


def sig(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, np.float64)))


def thresholds_for(bg, targets):
    grid = np.unique(np.concatenate([np.linspace(0.001, 0.999, 400),
                                     np.quantile(bg, np.linspace(0.90, 0.99999, 200))]))
    grid.sort()
    out, todo = {}, sorted(targets)
    for t in grid:
        if not todo:
            break
        fa, fl = fa_per_hour_on_background(bg, t, K, NR)
        for tg in list(todo):
            if fa <= tg and fl <= 0.02:
                out[tg] = float(t)
                todo.remove(tg)
    for tg in todo:
        out[tg] = 1.0
    return out


def contig_segments(p, y):
    """contiguous background stretches (>= 10 windows) of one file region."""
    segs = []
    bg = (y == 0)
    for lo, hi in find_runs(bg):
        if hi - lo >= 10:
            segs.append(p[lo:hi])
    return segs


def fa_contig(segs, t):
    runs, n = 0, 0
    for s in segs:
        fa, _ = fa_per_hour_on_background(s, t, K, NR)
        runs += fa * len(s) * 2.0 / 3600.0
        n += len(s)
    return runs, n * 2.0 / 3600.0


def main():
    folds = {}
    for p in sorted((REPO / "checkpoints" / "lopo").glob("lopo_*.pt")):
        name = p.stem[len("lopo_"):]
        folds[name] = name.split("_")
    cases = sorted(d.name for d in (REPO / "preprocessed_data").iterdir() if d.is_dir())
    files, splits, Y, LG = {}, {}, {}, {}
    for c in cases:
        files[c] = patient_files(c)
        splits[c] = find_split(files[c], 1)
        Y[c], LG[c] = [], []
        for f in files[c]:
            lg = {}
            with np.load(CACHE1 / ("%s__%s.npz" % (c, f.stem))) as d:
                y = d["labels"].astype(np.int8)
                for p in PILOT:
                    lg[p] = d["logit_" + p]
            with np.load(CACHE2 / ("%s__%s.npz" % (c, f.stem))) as d:
                for p in folds:
                    if p not in PILOT:
                        lg[p] = d["logit_" + p]
            Y[c].append(y)
            LG[c].append(lg)
    saved = json.loads((TREES / "out" / "thresholds_a.json").read_text())["A0"]

    def pre(c, per_file):
        fi, wi = splits[c]
        out = []
        for i in range(fi + 1):
            p, y = per_file[i], Y[c][i]
            if i == fi:
                p, y = p[:wi], y[:wi]
            out.append((p, y))
        return out

    def post(c, per_file):
        fi, wi = splits[c]
        out = []
        for i, (p, y) in enumerate(zip(per_file, Y[c])):
            if i < fi:
                continue
            if i == fi:
                p, y = p[wi:], y[wi:]
            if len(y) >= 10:
                out.append((p, y))
        return out

    rows = []
    agg = {k: np.zeros((len(SHOW), 2)) for k in ("unseen_sc", "unseen_ct", "test_ct", "test_off")}
    seen = {k: [] for k in ("seen_sc", "seen_ct")}
    worst = 0.0
    ext = {}
    percase = {}
    for name, held in folds.items():
        train_cases = [c for c in cases if c not in held]
        per = {c: [sig(lg[name]) for lg in LG[c]] for c in cases}
        rng = np.random.default_rng(42)
        masks = {q: adapt_background_mask(files[q], splits[q], 0.3, rng) for q in train_cases}
        pooled = []
        for q in train_cases:
            res, ev = [], []
            for i, (p, y) in enumerate(pre(q, per[q])):
                res.append(p[masks[q][i] & (y == 0)])
                ev.append(p[y == 0])
            h = np.concatenate(res)
            pooled.append((h if len(h) >= 200 else np.concatenate(ev)).astype(np.float32))
        cal = np.concatenate(pooled)
        thr = thresholds_for(cal, TARGETS + EXTRA)
        for tg in TARGETS:
            worst = max(worst, abs(thr[tg] - saved[held[0]][str(tg)]))
        ext[name] = thr
        seen_segs = [s for q in train_cases for p, y in pre(q, per[q]) for s in contig_segments(p, y)]
        r2 = np.random.default_rng(42)
        for c in held:
            pr = pre(c, per[c])
            mk = adapt_background_mask(files[c], splits[c], 0.3, r2)
            unseen_sc = np.concatenate([p[mk[i] & (y == 0)] for i, (p, y) in enumerate(pr)])
            unseen_segs = [s for p, y in pr for s in contig_segments(p, y)]
            test_segs = [s for p, y in post(c, per[c]) for s in contig_segments(p, y)]
            blocks = [(np.asarray(p, np.float64), y) for p, y in post(c, per[c])]
            percase[c] = []
            for j, tg in enumerate(SHOW):
                t = thr[tg]
                fa, _ = fa_per_hour_on_background(unseen_sc, t, K, NR)
                hrs = len(unseen_sc) * 2.0 / 3600
                agg["unseen_sc"][j] += (fa * hrs, hrs)
                agg["unseen_ct"][j] += fa_contig(unseen_segs, t)
                agg["test_ct"][j] += fa_contig(test_segs, t)
                _, fp, _, sec = count(blocks, t, K, NR)
                agg["test_off"][j] += (fp, sec / 3600)
                u = fa_contig(unseen_segs, t)
                percase[c].append((u[0] / u[1] if u[1] else np.nan, fp / (sec / 3600)))
        sc = [fa_per_hour_on_background(cal, thr[tg], K, NR)[0] for tg in SHOW]
        ct = [np.divide(*fa_contig(seen_segs, thr[tg])) for tg in SHOW]
        seen["seen_sc"].append(sc)
        seen["seen_ct"].append(ct)
        rows.append("  %-12s cal %6d win | seen scattered %s | seen contiguous %s"
                    % (name, len(cal), " ".join("%.3f" % x for x in sc),
                       " ".join("%.2f" % x for x in ct)))
        print(rows[-1], flush=True)
    L = ["A0 THRESHOLD TRANSFER (FA/h at the fold's deployable threshold, same run counter)",
         "threshold re-derivation vs saved thresholds_a.json: max |dt| %.2e" % worst,
         "targets: " + " ".join("%g" % t for t in SHOW)]
    L += rows
    L.append("  seen scattered  (median over folds): " +
             " ".join("%.3f" % x for x in np.median(seen["seen_sc"], 0)))
    L.append("  seen contiguous (median over folds): " +
             " ".join("%.3f" % x for x in np.median(seen["seen_ct"], 0)))
    for k in agg:
        L.append("  %-16s(pooled over 24 cases): %s" % (k, " ".join(
            "%.3f" % (agg[k][j, 0] / agg[k][j, 1]) for j in range(len(SHOW)))))
    L.append("  hours: unseen pre-split bg %.1f, test bg %.1f"
             % (agg["unseen_ct"][0, 1], agg["test_ct"][0, 1]))
    L.append("  per case unseen pre-split contiguous FA/h | test official FA/h, at target 0.05 and 2:")
    L.append("    " + " ".join("%s:%.2f|%.2f,%.2f|%.2f" % (c, v[0][0], v[0][1], v[3][0], v[3][1])
                               for c, v in sorted(percase.items())))
    arr = np.array([[v[j][0] for j in range(len(SHOW))] for v in percase.values()])
    arr2 = np.array([[v[j][1] for j in range(len(SHOW))] for v in percase.values()])
    L.append("  median over cases: unseen pre-split contiguous %s | test official %s" % (
        " ".join("%.2f" % x for x in np.nanmedian(arr, 0)), " ".join("%.2f" % x for x in np.median(arr2, 0))))
    L.append("  cases with unseen pre-split contiguous FA/h > 1 at target 0.05: %d of %d"
             % (int((arr[:, 0] > 1).sum()), int(np.isfinite(arr[:, 0]).sum())))
    (ME / "a3_transfer_A0.txt").write_text("\n".join(L) + "\n", encoding="ascii")
    np.save(ME / "a3_A0_ext_thresholds.npy", np.array(ext, dtype=object), allow_pickle=True)
    print("\n".join(L[-8:]))


if __name__ == "__main__":
    main()
