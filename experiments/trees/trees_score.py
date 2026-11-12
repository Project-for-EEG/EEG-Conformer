"""
Tree test, step 3: deployable curves, the decision in tree_rule.txt, and the secondary
views. Reads <out>/preds.npy and <out>/thresholds.json from trees_train.py. CPU only.
Writes <out>/score_results.txt, score_results.json and oracle tables.
"""
import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score

sys.dont_write_bytecode = True
REPO = Path(r"C:\EEG")
sys.path.insert(0, str(REPO))
from score_official import count, score_all      # noqa: E402
from evaluate_temporal import smooth             # noqa: E402

TARGETS = ("0.05", "0.1", "0.15", "0.2", "0.3", "0.4", "0.5", "0.75", "1.0", "1.5", "2.0",
           "3.0", "4.0", "5.0")
SEEDS = (0, 1, 2)
GROUPS = {"A0": ["A0"], "Tamp": ["Tamp_s%d" % s for s in SEEDS],
          "Tfree": ["Tfree_s%d" % s for s in SEEDS], "Tall": ["Tall"]}
BUDGETS = (5, 10, 25, 50, 100)
K, N_RUN, NBOOT = 5, 3, 2000


def as_data(P):
    return {c: {"blocks": [(np.asarray(p, np.float64), y) for p, y in P[c]]} for c in P}


def deploy_table(P, T, cases):
    """Per case and target: tp, fp; plus ref and sec per case."""
    tp = np.zeros((len(cases), len(TARGETS)), np.int64)
    fp = np.zeros_like(tp)
    ref = np.zeros(len(cases), np.int64)
    sec = np.zeros(len(cases))
    for i, c in enumerate(cases):
        blocks = [(np.asarray(p, np.float64), y) for p, y in P[c]]
        for j, tg in enumerate(TARGETS):
            a, b, r, s = count(blocks, float(T[c][tg]), K, N_RUN)
            tp[i, j], fp[i, j] = a, b
            ref[i], sec[i] = r, s
    return {"tp": tp, "fp": fp, "ref": ref, "sec": sec}


def readout(D, b, w=None):
    """Best pooled sensitivity among targets whose pooled achieved FA/day <= b."""
    w = np.ones(len(D["ref"])) if w is None else w
    tp, fp = w @ D["tp"], w @ D["fp"]
    ref, sec = w @ D["ref"], w @ D["sec"]
    fa = fp / (sec / 86400.0)
    ok = np.flatnonzero(fa <= b + 1e-12)
    if not len(ok):
        return 0.0, float("nan"), None
    j = ok[np.argmax(tp[ok])]
    return tp[j] / ref, float(fa[j]), TARGETS[j]


def oracle_job(job):
    path, arm = job
    P = np.load(path, allow_pickle=True).item()[arm]
    cases = sorted(P)
    blocks = {c: [(np.asarray(p, np.float64), y) for p, y in P[c]] for c in cases}
    sm = np.concatenate([smooth(p, K) for c in cases for p, _ in blocks[c]])
    q = 1.0 - np.logspace(np.log10(0.3), -6.5, 3000)
    grid = np.unique(np.concatenate([np.linspace(0.01, 0.999, 100), np.quantile(sm, q)]))
    tp = np.zeros((len(cases), len(grid)), np.int32)
    fp = np.zeros_like(tp)
    ref = np.zeros(len(cases), np.int64)
    sec = np.zeros(len(cases))
    for i, c in enumerate(cases):
        for g, t in enumerate(grid):
            a, b, r, s = count(blocks[c], float(t), K, N_RUN)
            tp[i, g], fp[i, g] = a, b
            ref[i], sec[i] = r, s
    return arm, {"grid": grid, "tp": tp, "fp": fp, "ref": ref, "sec": sec}


def oracle_pooled(T, b):
    tp, fp = T["tp"].sum(0), T["fp"].sum(0)
    fa = fp / (T["sec"].sum() / 86400.0)
    ok = np.flatnonzero(fa <= b + 1e-12)
    return tp[ok].max() / T["ref"].sum()


def oracle_per_patient(T, b):
    fa = T["fp"] / (T["sec"][:, None] / 86400.0)
    return np.where(fa <= b + 1e-12, T["tp"], 0).max(1).sum() / T["ref"].sum()


def verdict(d25, d10):
    if d25 >= 0.05 and d10 >= 0:
        return "TREES BETTER"
    if d25 <= -0.05:
        return "TREES WORSE"
    return "SAME"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=8)
    args = ap.parse_args()
    out = Path(args.out)
    P, T = {}, {}
    for f in sorted(out.glob("preds_*.npy")):
        part = np.load(f, allow_pickle=True).item()
        tpart = json.loads((out / f.name.replace("preds_", "thresholds_")
                            .replace(".npy", ".json")).read_text())
        for a in part:
            assert a not in P, ("arm in two files", a)
            P[a], T[a] = part[a], tpart[a]
    np.save(out / "preds.npy", np.array(P, dtype=object), allow_pickle=True)
    groups = {g: [a for a in arms if a in P] for g, arms in GROUPS.items()}
    groups = {g: a for g, a in groups.items() if a}
    cases = sorted(P["A0"])
    lines, res = [], {}

    def say(s=""):
        print(s, flush=True)
        lines.append(s)

    # ---- checks --------------------------------------------------------------------
    ref = np.load(REPO / "endtoend_uncalibrated.npz", allow_pickle=True)["data"].item()
    worst, tworst = 0.0, 0.0
    for c in cases:
        assert len(P["A0"][c]) == len(ref[c]["blocks"])
        for (p1, y1), (p2, y2) in zip(P["A0"][c], ref[c]["blocks"]):
            assert np.array_equal(np.asarray(y1, np.int64), np.asarray(y2, np.int64))
            worst = max(worst, float(np.max(np.abs(p1 - p2))))
        tworst = max(tworst, abs(T["A0"][c]["2.0"] - float(ref[c]["t"])))
    say("CHECK A0 test blocks vs endtoend_uncalibrated.npz: max |dp| %.2e (%s)"
        % (worst, "PASS" if worst < 1e-4 else "FAIL"))
    say("CHECK A0 thresholds at 2 FA/h vs stored: max |dt| %.2e (%s)"
        % (tworst, "PASS" if tworst < 1e-4 else "FAIL"))
    pl, _, _ = score_all(as_data(P["A0"]), {c: T["A0"][c]["2.0"] for c in cases}, K, N_RUN)
    say("CHECK A0 at target 2 FA/h: %d/%d at %.1f FA/day (README: 131/168 at 131.2)"
        % (pl["caught"], pl["events"], pl["fa_day"]))
    for a in P:
        assert sorted(P[a]) == cases, a
        for c in cases:
            for (p, y), (_, y0) in zip(P[a][c], P["A0"][c]):
                assert len(p) == len(y0) and np.array_equal(y, y0), (a, c)
    say("arms: " + ", ".join(P))
    say()

    # ---- deployable curves -----------------------------------------------------------
    D = {a: deploy_table(P[a], T[a], cases) for a in P}
    say("DEPLOYABLE CURVES: threshold from training patients only, at each target FA/h "
        "on their background. Cells: sens (test FA/day); trees = mean over seeds")
    say("%-8s" % "target" + "".join("%20s" % g for g in groups))
    for j, tg in enumerate(TARGETS):
        row = "%-8s" % tg
        for g, arms in groups.items():
            s = np.mean([D[a]["tp"][:, j].sum() / D[a]["ref"].sum() for a in arms])
            f = np.mean([D[a]["fp"][:, j].sum() / (D[a]["sec"].sum() / 86400) for a in arms])
            row += "%20s" % ("%.3f (%6.1f)" % (s, f))
        say(row)
    say()

    say("READOUT: best sensitivity on the deployable curve within each budget "
        "(per seed, then mean)")
    say("%-8s" % "" + "".join("%16s" % ("<=%d/d" % b) for b in BUDGETS))
    R = {}
    for g, arms in groups.items():
        for a in arms:
            R[a] = {b: readout(D[a], b) for b in BUDGETS}
            if len(arms) > 1:
                say("%-8s" % a + "".join("%16s" % ("%.3f (%5.1f)" % R[a][b][:2])
                                         for b in BUDGETS))
        R[g] = {b: float(np.mean([R[a][b][0] for a in arms])) for b in BUDGETS}
        say("%-8s" % g + "".join("%16s" % ("%.3f" % R[g][b]) for b in BUDGETS))
    say()
    d25 = R["Tamp"][25] - R["A0"][25]
    d10 = R["Tamp"][10] - R["A0"][10]
    amp25 = R["Tamp"][25] - R["Tfree"][25]
    v = verdict(d25, d10)
    amp = ("amplitude helps" if amp25 >= 0.05 else
           "amplitude hurts" if amp25 <= -0.05 else "no clear effect")
    say("PRIMARY: Tamp - A0 at 25/day %+.3f, at 10/day %+.3f -> %s" % (d25, d10, v))
    say("AMPLITUDE: Tamp - Tfree at 25/day %+.3f -> %s" % (amp25, amp))
    say("also: Tfree - A0 at 25/day %+.3f, 10/day %+.3f"
        % (R["Tfree"][25] - R["A0"][25], R["Tfree"][10] - R["A0"][10]))
    if "Tall" in R:
        say("also: Tall - A0 at 25/day %+.3f, 10/day %+.3f | Tall - Tamp at 25/day %+.3f"
            % (R["Tall"][25] - R["A0"][25], R["Tall"][10] - R["A0"][10],
               R["Tall"][25] - R["Tamp"][25]))
    say()
    res["readout"] = {g: R[g] for g in groups}
    res["primary"] = {"d25": d25, "d10": d10, "verdict": v, "amp25": amp25, "amp": amp}

    j2 = TARGETS.index("2.0")
    say("README OPERATING POINT (target 2 FA/h): caught/168 and FA/day")
    for g, arms in groups.items():
        for a in arms:
            say("  %-9s %3d/%d  %6.1f FA/day" % (a, D[a]["tp"][:, j2].sum(), D[a]["ref"].sum(),
                                                D[a]["fp"][:, j2].sum() / (D[a]["sec"].sum() / 86400)))
    say()

    # ---- bootstrap and leave-one-subject-out of the primary contrast -----------------
    subj = []
    for c in cases:
        s = "CHB01" if c == "CHB21" else c
        if s not in subj:
            subj.append(s)
    S = np.array([[1.0 if (c == s or (s == "CHB01" and c == "CHB21")) else 0.0
                   for c in cases] for s in subj])

    def contrast(w, b, g1="Tamp", g0="A0"):
        r1 = np.mean([readout(D[a], b, w)[0] for a in groups[g1]])
        r0 = np.mean([readout(D[a], b, w)[0] for a in groups[g0]])
        return r1 - r0

    rng = np.random.default_rng(20261104)
    idx = rng.integers(0, len(subj), size=(NBOOT, len(subj)))
    Wc = np.zeros((NBOOT, len(subj)))
    for r in range(NBOOT):
        np.add.at(Wc[r], idx[r], 1)
    Wc = Wc @ S
    boot = {b: np.array([contrast(Wc[r], b) for r in range(NBOOT)]) for b in (10, 25)}
    bamp = np.array([contrast(Wc[r], 25, "Tamp", "Tfree") for r in range(NBOOT)])
    say("SUBJECT BOOTSTRAP (%d draws, %d subjects, budget re-chosen in every draw)"
        % (NBOOT, len(subj)))
    for b in (10, 25):
        lo, hi = np.percentile(boot[b], [2.5, 97.5])
        say("  Tamp - A0 at %2d/d: point %+.3f | 95%% interval [%+.3f, %+.3f] | P(<=0) %.3f"
            % (b, d25 if b == 25 else d10, lo, hi, (boot[b] <= 0).mean()))
    vs = [verdict(boot[25][r], boot[10][r]) for r in range(NBOOT)]
    say("  verdict share: " + ", ".join("%s %.0f%%" % (x, 100.0 * vs.count(x) / NBOOT)
                                        for x in ("TREES BETTER", "SAME", "TREES WORSE")))
    lo, hi = np.percentile(bamp, [2.5, 97.5])
    say("  Tamp - Tfree at 25/d: point %+.3f | 95%% interval [%+.3f, %+.3f]" % (amp25, lo, hi))
    loo = [contrast(1.0 - S[i], 25) for i in range(len(subj))]
    say("  leave one subject out, Tamp - A0 at 25/d: smallest %+.3f (without %s), "
        "largest %+.3f (without %s)" % (min(loo), subj[int(np.argmin(loo))], max(loo),
                                        subj[int(np.argmax(loo))]))
    say()
    res["boot"] = {str(b): [float(x) for x in np.percentile(boot[b], [2.5, 97.5])]
                   for b in (10, 25)}

    # ---- oracle views ------------------------------------------------------------------
    tabs = {}
    with Pool(args.procs) as pool:
        for arm, Tb in pool.imap_unordered(oracle_job, [(str(out / "preds.npy"), a) for a in P]):
            tabs[arm] = Tb
            np.savez_compressed(out / ("oracle_%s.npz" % arm), cases=np.array(cases), **Tb)
    for name, fn in (("POOLED RAW-SCORE ORACLE (mostly cross-fold calibration)", oracle_pooled),
                     ("PER-PATIENT ORACLE (each patient's own best threshold)", oracle_per_patient)):
        say(name)
        say("%-8s" % "" + "".join("%10s" % ("<=%d/d" % b) for b in BUDGETS))
        for g, arms in groups.items():
            say("%-8s" % g + "".join("%10.3f" % np.mean([fn(tabs[a], b) for a in arms])
                                     for b in BUDGETS))
        say()

    # ---- window AUC per held-out case ---------------------------------------------------
    auc = {}
    for g, arms in groups.items():
        auc[g] = np.array([np.mean([roc_auc_score(
            np.concatenate([y for _, y in P[a][c]]),
            np.concatenate([smooth(np.asarray(p, np.float64), K) for p, _ in P[a][c]]))
            for a in arms]) for c in cases])
    say("WINDOW AUC per held-out case (smoothed k=5; trees = mean over seeds)")
    say("  median: " + ", ".join("%s %.3f" % (g, np.median(v)) for g, v in auc.items()))
    for g in [x for x in groups if x != "A0"]:
        d = auc[g] - auc["A0"]
        say("  %s - A0: median %+.3f, better in %d of %d cases, Wilcoxon p %.3f"
            % (g, np.median(d), int((d > 0).sum()), len(d), wilcoxon(d).pvalue))
    say("  per case: " + " ".join("%s %s" % (c, "/".join("%.2f" % auc[g][i] for g in groups))
                                  for i, c in enumerate(cases)))
    say("  (order: %s)" % "/".join(groups))
    res["auc"] = {g: v.tolist() for g, v in auc.items()}

    (out / "score_results.txt").write_text("\n".join(lines) + "\n", encoding="ascii")
    (out / "score_results.json").write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
