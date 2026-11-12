"""critic: analyse one refit (verify/critic/refit_<arm>_<fold>.npy) for Q4 (threshold transfer and
in-sample memorisation) and, for Trel, Q3 (gain normalisation). Also runs A0 on the same fold.
Run with cwd C:\\EEG. Usage: python r_probe.py --file refit_Tamp_s0_CHB06.npy"""
import argparse
import json
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
sys.path.insert(0, r"C:\EEG")
from finetune_per_patient import patient_files, find_split, adapt_background_mask  # noqa: E402
from calibrate_per_patient import fa_per_hour_on_background                         # noqa: E402
from score_official import count                                                    # noqa: E402
from evaluate_temporal import find_runs, smooth                                     # noqa: E402

TREES = Path(r"C:\Users\user\AppData\Local\Temp\claude\36c1f29744f93fe6"
             r"\b3889703-629c-4c5a-b583-69f722743ff4\scratchpad\trees")
ME, OUT = TREES / "verify" / "critic", TREES / "out"
OLD = Path(r"C:\Users\user\AppData\Local\Temp\claude\36c1f29744f93fe6"
           r"\f241281f-add4-41ac-9210-f47dab0b83b0\scratchpad")
PILOT = ["CHB12", "CHB15", "CHB24", "CHB13", "CHB06", "CHB16"]
TARGETS = (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0)
SHOW = (0.05, 0.2, 0.5, 2.0, 5.0)
K, NR = 5, 3


def thresholds_for(bg, targets=TARGETS):
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


MARGIN = 60   # windows (2 min) excluded on each side of every labelled seizure


def segs_of(p, y):
    bad = (y == 1).copy()
    for lo, hi in find_runs(y == 1):
        bad[max(0, lo - MARGIN):hi + MARGIN] = True
    return [p[lo:hi] for lo, hi in find_runs(~bad) if hi - lo >= 10]


def fa_contig(segs, t):
    runs = hrs = 0.0
    for s in segs:
        fa, _ = fa_per_hour_on_background(s, t, K, NR)
        h = len(s) * 2.0 / 3600
        runs += fa * h
        hrs += h
    return runs / hrs if hrs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    a = ap.parse_args()
    R = np.load(ME / a.file, allow_pickle=True).item()
    fold, arm, probs, intrain = R["fold"], R["arm"], R["probs"], R["intrain"]
    held = fold.split("_")
    cases = sorted(probs)
    train_cases = [c for c in cases if c not in held]
    files = {c: patient_files(c) for c in cases}
    splits = {c: find_split(files[c], 1) for c in cases}
    Y = {c: [np.load(f)["labels"].astype(np.int8) for f in files[c]] for c in cases}
    L = ["REFIT PROBE: arm %s, fold %s (contiguous sets exclude +-%d windows around seizures)" % (arm, fold, MARGIN)]

    a0 = {}
    for c in cases:
        a0[c] = []
        for f in files[c]:
            src = OLD / ("pilot_cache" if fold in PILOT else "pilot_cache2") / ("%s__%s.npz" % (c, f.stem))
            with np.load(src) as d:
                a0[c].append(1.0 / (1.0 + np.exp(-d["logit_" + fold].astype(np.float64))))

    def pre(c, per):
        fi, wi = splits[c]
        return [((per[i][:wi], Y[c][i][:wi]) if i == fi else (per[i], Y[c][i])) for i in range(fi + 1)]

    def post(c, per):
        fi, wi = splits[c]
        out = []
        for i in range(fi, len(per)):
            p, y = (per[i][wi:], Y[c][i][wi:]) if i == fi else (per[i], Y[c][i])
            if len(y) >= 10:
                out.append((np.asarray(p, np.float64), y))
        return out

    saved = {}
    for t in "abc":
        saved.update(json.loads((OUT / ("thresholds_%s.json" % t)).read_text()))
    Psaved = np.load(OUT / "preds.npy", allow_pickle=True).item()
    if arm in Psaved:
        d = max(float(np.max(np.abs(p1 - np.asarray(p2)))) for c in held
                for (p1, _), (p2, _) in zip(post(c, probs[c]), Psaved[arm][c]))
        L.append("replica check: max |p_refit - p_saved| on held-out test windows %.2e" % d)

    for model, per_all in ((arm, probs), ("A0", a0)):
        rng = np.random.default_rng(42)
        masks = {q: adapt_background_mask(files[q], splits[q], 0.3, rng) for q in train_cases}
        cal, cal_in, nbr = [], [], []
        for q in train_cases:
            nres = sum(int((masks[q][i] & (y == 0)).sum()) for i, (p, y) in enumerate(pre(q, per_all[q])))
            for i, (p, y) in enumerate(pre(q, per_all[q])):
                sel = (masks[q][i] & (y == 0)) if nres >= 200 else (y == 0)
                cal.append(np.asarray(p, np.float32)[sel].astype(np.float64))
                if model == arm:
                    it = intrain[q][i][:len(y)]
                    cal_in.append(it[sel])
                    near = it.copy()
                    for s in (1, 2):
                        near[s:] |= it[:-s]
                        near[:-s] |= it[s:]
                    nbr.append(near[sel] & ~it[sel])
        cal = np.concatenate(cal).astype(np.float32)
        thr = thresholds_for(cal)
        if model in saved:
            dt = max(abs(thr[tg] - saved[model][held[0]][str(tg)]) for tg in TARGETS)
            L.append("[%s] thresholds re-derived vs saved: max |dt| %.2e" % (model, dt))
        L.append("[%s] calibration windows %d; thresholds at %s: %s" % (
            model, len(cal), SHOW, " ".join("%.4f" % thr[t] for t in SHOW)))
        if model == arm:
            cin, cnb = np.concatenate(cal_in), np.concatenate(nbr)
            L.append("[%s] calibration windows that were training windows: %.1f%%; out-of-train "
                     "windows within +-2 of a training window: %.1f%%" % (model, 100 * cin.mean(),
                                                                          100 * cnb.mean()))
            for lab, m in (("in-train", cin), ("out, near", cnb), ("out, far", ~cin & ~cnb)):
                L.append("    raw score quantiles 99/99.9/99.99%% %-10s %.4f %.4f %.4f  mean %.4f"
                         % ((lab,) + tuple(np.quantile(cal[m], [0.99, 0.999, 0.9999])) + (cal[m].mean(),)))
            thr_out = thresholds_for(cal[~cin])
            thr_far = thresholds_for(cal[~cin & ~cnb])
            L.append("[%s] thresholds from out-of-train calibration only: %s | from far-only: %s"
                     % (model, " ".join("%.4f" % thr_out[t] for t in SHOW),
                        " ".join("%.4f" % thr_far[t] for t in SHOW)))
        seen_pre = [s for q in train_cases for p, y in pre(q, per_all[q]) for s in segs_of(np.asarray(p, np.float64), y)]
        seen_post = [s for q in train_cases for p, y in post(q, per_all[q]) for s in segs_of(p, y)]
        L.append("[%s] FA/h at targets %s on:" % (model, SHOW))
        L.append("    seen patients, calibration (scattered)  %s" % " ".join(
            "%7.3f" % fa_per_hour_on_background(cal, thr[t], K, NR)[0] for t in SHOW))
        L.append("    seen patients, pre-split contiguous     %s" % " ".join(
            "%7.3f" % fa_contig(seen_pre, thr[t]) for t in SHOW))
        L.append("    seen patients, post-split contiguous    %s" % " ".join(
            "%7.3f" % fa_contig(seen_post, thr[t]) for t in SHOW))
        per_q = {q: fa_contig([s for p, y in pre(q, per_all[q]) for s in segs_of(np.asarray(p, np.float64), y)],
                              thr[2.0]) for q in train_cases}
        v = np.array([x for x in per_q.values() if np.isfinite(x)])
        L.append("    seen patients, per-patient pre-split contiguous FA/h at target 2: median %.2f, "
                 "range %.2f-%.2f, top %s" % (np.median(v), v.min(), v.max(), " ".join(
                     "%s:%.1f" % kv for kv in sorted(per_q.items(), key=lambda kv: -kv[1])[:4])))
        for c in held:
            r2 = np.random.default_rng(42)
            pr = pre(c, per_all[c])
            mk = adapt_background_mask(files[c], splits[c], 0.3, r2)
            sc = np.concatenate([np.asarray(p, np.float64)[mk[i] & (y == 0)] for i, (p, y) in enumerate(pr)])
            us = [s for p, y in pr for s in segs_of(np.asarray(p, np.float64), y)]
            ts = [s for p, y in post(c, per_all[c]) for s in segs_of(p, y)]
            bl = post(c, per_all[c])
            L.append("    unseen %s pre-split scattered        %s" % (c, " ".join(
                "%7.3f" % fa_per_hour_on_background(sc, thr[t], K, NR)[0] for t in SHOW)))
            L.append("    unseen %s pre-split contiguous       %s" % (c, " ".join(
                "%7.3f" % fa_contig(us, thr[t]) for t in SHOW)))
            L.append("    unseen %s test contiguous            %s" % (c, " ".join(
                "%7.3f" % fa_contig(ts, thr[t]) for t in SHOW)))
            res = [count(bl, thr[t], K, NR) for t in TARGETS]
            L.append("    unseen %s test official FA/h         %s" % (c, " ".join(
                "%7.3f" % (r[1] / (r[3] / 3600)) for r, t in zip(res, TARGETS) if t in SHOW)))
            L.append("    unseen %s test caught/ref at 14 targets: %s (ref %d, %.1f h)" % (
                c, " ".join(str(r[0]) for r in res), res[0][2], res[0][3] / 3600))
            L.append("    unseen %s test FA at 14 targets:         %s" % (c, " ".join(str(r[1]) for r in res)))
            yy = np.concatenate([y for _, y in bl])
            pp = np.concatenate([smooth(p, K) for p, _ in bl])
            L.append("    unseen %s test window AUC (k=5) %.3f" % (c, roc_auc_score(yy, pp)))
            if model == arm:
                for lab, th in (("out-of-train-calibrated", thr_out), ("far-calibrated", thr_far)):
                    res = [count(bl, th[t], K, NR) for t in TARGETS]
                    L.append("    unseen %s test with %s thresholds: caught %s | FA %s" % (
                        c, lab, " ".join(str(r[0]) for r in res), " ".join(str(r[1]) for r in res)))
    out = ME / ("probe_%s_%s.txt" % (arm, fold))
    out.write_text("\n".join(L) + "\n", encoding="ascii")
    print("\n".join(L))


if __name__ == "__main__":
    main()
