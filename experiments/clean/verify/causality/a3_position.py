"""
Audit task 3: could Trel look better for a trivial, file-boundary reason?

Rebuilds each test block's file stem and its offset inside the file (split file starts at wi),
then, for Tamp_s0..2 and Trel_s0..2 at targets 2.0 and 0.05 (thresholds as stored), uses the
official timescoring EventScoring per block to list every false-alarm event (start window) and
every reference event (onset window, caught or not). Bins by position inside the RECORDING FILE:
[0,30) first minute, [30,150) 1-5 min, [150,300) 5-10 min, [300,end) median fully formed.
Also: file lengths by seizure content, Trel's probability at file positions 0..4, seizures whose
300-window history holds an earlier seizure, NaN cells, and per-case README-point numbers.
155-event set (chb12_27/28/29 removed) unless stated.
"""
import json
import os
import sys
import warnings

sys.dont_write_bytecode = True
import numpy as np  # noqa: E402

sys.path.insert(0, r"C:\EEG")
from finetune_per_patient import patient_files, find_split  # noqa: E402
from score_official import detect, count, MASK_FS  # noqa: E402
from evaluate_temporal import find_runs  # noqa: E402
from timescoring import scoring  # noqa: E402
from timescoring.annotations import Annotation  # noqa: E402

OLD = r"C:\Users\user\AppData\Local\Temp\claude\36c1f29744f93fe6\b3889703-629c-4c5a-b583-69f722743ff4\scratchpad\trees"
NEW = r"C:\Users\user\AppData\Local\Temp\claude\36c1f29744f93fe6\c7848ca6-bfef-414b-a321-d6686c1a43f8\scratchpad\clean"
EXCL = {"chb12_27_labeled", "chb12_28_labeled", "chb12_29_labeled"}
BINS = [0, 30, 150, 300, 10 ** 9]
BNAME = ["[0,30)", "[30,150)", "[150,300)", "[300,end)"]


def load(path_npy, path_json, arm):
    P = np.load(path_npy, allow_pickle=True).item()[arm]
    T = json.load(open(path_json))[arm]
    return P, T


ARMS = {}
for k, t in enumerate("abc"):
    ARMS["Tamp_s%d" % k] = load(os.path.join(OLD, "out", "preds_%s.npy" % t),
                                os.path.join(OLD, "out", "thresholds_%s.json" % t), "Tamp_s%d" % k)
    ARMS["Trel_s%d" % k] = load(os.path.join(NEW, "trees_out", "preds_rel%d.npy" % k),
                                os.path.join(NEW, "trees_out", "thresholds_rel%d.json" % k),
                                "Trel_s%d" % k)
cases = sorted(ARMS["Tamp_s0"][0])

# ---- block map: stem, offset in file, file length
MAP = {}
for c in cases:
    files = patient_files(c)
    fi, wi = find_split(files, 1)
    out = []
    for i, f in enumerate(files):
        if i < fi:
            continue
        with np.load(os.path.join(OLD, "feats", "%s__%s.npz" % (c, f.stem))) as d:
            lab = d["labels"]
        lo = wi if i == fi else 0
        if len(lab) - lo >= 10:
            out.append({"stem": f.stem, "lo": lo, "n_file": len(lab), "labels": lab[lo:]})
    for a in ARMS:
        assert len(out) == len(ARMS[a][0][c])
        for b, (_, y) in zip(out, ARMS[a][0][c]):
            assert np.array_equal(np.asarray(b["labels"], np.int64), np.asarray(y, np.int64))
    MAP[c] = out
keep = {c: [b["stem"] not in EXCL for b in MAP[c]] for c in cases}


def events(probs, labels, thr):
    """Official scoring of one block, plus positions of FP events and of ref events/caught."""
    pred = detect(np.asarray(probs, np.float64), thr, 5, 3)
    truth = np.asarray(labels).astype(bool)
    s = scoring.EventScoring(Annotation(truth, MASK_FS), Annotation(pred, MASK_FS))
    fs = s.fs
    fp_starts = [ev[0] for ev in s.hyp.events
                 if np.all(~s.tpMask[round(ev[0] * fs):round(ev[1] * fs)])]
    assert len(fp_starts) == s.fp
    # replicate the TP loop exactly: over the (possibly merged) extended reference events
    prm = scoring.EventScoring.Parameters()
    ext = scoring.EventScoring._extendEvents(s.ref, prm.toleranceStart, prm.toleranceEnd)
    caught = []
    for ev in ext.events:
        ov = (np.sum(s.hyp.mask[round(ev[0] * fs):round(ev[1] * fs)]) / fs) / (ev[1] - ev[0])
        inside = [r[0] for r in s.ref.events if ev[0] <= r[0] <= ev[1]]
        caught.append((inside[0] if inside else ev[0], bool(ov > prm.minOverlap + 1e-6)))
    assert sum(c for _, c in caught) == s.tp, (sum(c for _, c in caught), s.tp)
    refs = [r[0] for r in s.ref.events]
    return s.tp, s.fp, s.refTrue, fp_starts, caught, refs


def binof(pos):
    return int(np.searchsorted(BINS, pos, side="right") - 1)


def main():
    # ---- structure of the test set
    hours = np.zeros(4)
    ev_bins = np.zeros(4, int)
    flen_sz, flen_bg = [], []
    hist_contam = 0
    n_events = 0
    for c in cases:
        for b, k in zip(MAP[c], keep[c]):
            if not k:
                continue
            pos = b["lo"] + np.arange(len(b["labels"]))
            for j in range(4):
                hours[j] += np.sum((pos >= BINS[j]) & (pos < BINS[j + 1])) * 2 / 3600
            with np.load(os.path.join(OLD, "feats", "%s__%s.npz" % (c, b["stem"]))) as d:
                full = d["labels"].astype(bool)
            runs = find_runs(full)
            (flen_sz if len(runs) else flen_bg).append(b["n_file"])
            for lo_, hi_ in runs:
                if lo_ < b["lo"]:
                    continue
                n_events += 1
                ev_bins[binof(lo_)] += 1
                if np.any(full[max(0, lo_ - 299):lo_]):
                    hist_contam += 1
    print("TEST SET (155 events): %d onsets counted from labels; test hours by position in file:"
          % n_events)
    for j in range(4):
        print("   %-10s %7.1f h (%4.1f%%)   onsets %3d (%4.1f%%)"
              % (BNAME[j], hours[j], 100 * hours[j] / hours.sum(), ev_bins[j],
                 100 * ev_bins[j] / ev_bins.sum()))
    print("   onsets whose 300-window history already holds seizure windows: %d" % hist_contam)
    fs_, fb_ = np.array(flen_sz), np.array(flen_bg)
    print("   test files WITH seizure: %d, length windows median %d, min %d, max %d, <300: %d"
          % (len(fs_), np.median(fs_), fs_.min(), fs_.max(), int(np.sum(fs_ < 300))))
    print("   test files WITHOUT seizure: %d, length median %d, min %d, max %d, <300: %d"
          % (len(fb_), np.median(fb_), fb_.min(), fb_.max(), int(np.sum(fb_ < 300))))

    # ---- per arm: FP and catches by bin
    res = {}
    for tg in ("2.0", "0.05"):
        print("\nTARGET %s FA/h  (FP events by start position; catches by onset position)" % tg)
        for a in ARMS:
            P, T = ARMS[a]
            fpb = np.zeros(4, int)
            tpb = np.zeros(4, int)
            refb = np.zeros(4, int)
            TP = FP = REF = 0
            sec = 0.0
            per_case = {}
            for c in cases:
                blocks = [(p, y) for (p, y), k in zip(P[c], keep[c]) if k]
                bm = [b for b, k in zip(MAP[c], keep[c]) if k]
                thr = float(T[c][tg])
                ctp = cfp = cref = 0
                for (p, y), b in zip(blocks, bm):
                    tp, fp, ref, fps, caught, refs = events(p, y, thr)
                    ctp, cfp, cref = ctp + tp, cfp + fp, cref + ref
                    for st in fps:
                        fpb[binof(b["lo"] + st * MASK_FS)] += 1
                    for st, cg in caught:
                        tpb[binof(b["lo"] + st * MASK_FS)] += cg
                    for st in refs:
                        refb[binof(b["lo"] + st * MASK_FS)] += 1
                    sec += len(y) * 2.0
                o = count(blocks, thr, 5, 3)
                assert (o[0], o[1], o[2]) == (ctp, cfp, cref), (a, c, o, ctp, cfp, cref)
                per_case[c] = (ctp, cref, cfp, o[3])
                TP, FP, REF = TP + ctp, FP + cfp, REF + cref
            res[(tg, a)] = dict(fpb=fpb, tpb=tpb, refb=refb, per_case=per_case)
            print("  %-8s caught %3d/%3d  FA %5d = %6.1f/day | FA/day by bin %s | caught/events by bin %s"
                  % (a, TP, REF, FP, FP / (sec / 86400),
                     " ".join("%6.1f" % (fpb[j] / hours[j] * 24) for j in range(4)),
                     " ".join("%d/%d" % (tpb[j], refb[j]) for j in range(4))))
        for g in ("Tamp", "Trel"):
            fpb = np.mean([res[(tg, "%s_s%d" % (g, k))]["fpb"] for k in range(3)], 0)
            tpb = np.mean([res[(tg, "%s_s%d" % (g, k))]["tpb"] for k in range(3)], 0)
            res[(tg, g)] = (fpb, tpb)
        dfp = res[(tg, "Tamp")][0] - res[(tg, "Trel")][0]
        print("  seed-mean FA removed by Trel (Tamp - Trel) per bin: %s  total %.1f; share by bin %s;"
              " share of test hours by bin %s"
              % (" ".join("%.1f" % v for v in dfp), dfp.sum(),
                 " ".join("%.1f%%" % (100 * v / dfp.sum()) for v in dfp),
                 " ".join("%.1f%%" % (100 * h / hours.sum()) for h in hours)))
        # steady state only: FA/day and sensitivity using only windows >= 300 of each file
        for g in ("Tamp", "Trel"):
            fpb, tpb = res[(tg, g)]
            refb = res[(tg, g + "_s0")]["refb"]
            print("  %s steady state (file position >= 300): %.1f FA/day, caught %.1f/%d (%.3f);"
                  "  first 300 windows: %.1f FA/day, caught %.1f/%d"
                  % (g, fpb[3] / hours[3] * 24, tpb[3], refb[3], tpb[3] / refb[3],
                     fpb[:3].sum() / hours[:3].sum() * 24, tpb[:3].sum(), refb[:3].sum()))

    # ---- per case at the README point, seed 0
    print("\nPER CASE, target 2.0, seed 0, 155 set: caught/events, FA/day  (Tamp_s0 | Trel_s0)")
    for c in cases:
        a = res[("2.0", "Tamp_s0")]["per_case"][c]
        b = res[("2.0", "Trel_s0")]["per_case"][c]
        print("  %-6s %3d/%3d %7.1f/day | %3d/%3d %7.1f/day   dFA/day %+7.1f  test h %.1f"
              % (c, a[0], a[1], a[2] / (a[3] / 86400), b[0], b[1], b[2] / (b[3] / 86400),
                 b[2] / (b[3] / 86400) - a[2] / (a[3] / 86400), a[3] / 3600))

    # ---- Trel probability near the file start (test background, seed 0)
    P = ARMS["Trel_s0"][0]
    Pa = ARMS["Tamp_s0"][0]
    near = {j: [] for j in range(5)}
    neara = {j: [] for j in range(5)}
    later, latera = [], []
    for c in cases:
        for (p, y), (pa, _), b, k in zip(P[c], Pa[c], MAP[c], keep[c]):
            if not k:
                continue
            p, pa, y = np.asarray(p), np.asarray(pa), np.asarray(y)
            for j in range(5):
                if b["lo"] == 0 and y[j] == 0:
                    near[j].append(p[j])
                    neara[j].append(pa[j])
            pos = b["lo"] + np.arange(len(y))
            m = (pos >= 300) & (y == 0)
            later.append(p[m])
            latera.append(pa[m])
    later, latera = np.concatenate(later), np.concatenate(latera)
    print("\nRAW window probability on test background, by position in file (blocks that start at"
          " file start), seed 0:")
    for j in range(5):
        print("   window %d: n=%3d  Trel mean %.4f, frac>0.2 %.3f | Tamp mean %.4f, frac>0.2 %.3f"
              % (j, len(near[j]), np.mean(near[j]), np.mean(np.array(near[j]) > 0.2),
                 np.mean(neara[j]), np.mean(np.array(neara[j]) > 0.2)))
    print("   pos>=300: n=%d  Trel mean %.4f, frac>0.2 %.4f | Tamp mean %.4f, frac>0.2 %.4f"
          % (len(later), later.mean(), np.mean(later > 0.2), latera.mean(), np.mean(latera > 0.2)))


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
