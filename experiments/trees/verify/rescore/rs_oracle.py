"""Exact oracle tables and window AUC, recomputed independently.

For every test block the event score is a step function of the threshold that can only
change at the block's own smoothed values, so each block is scored once at every distinct
smoothed value (timescoring called only when the post-processed detection changes). Case
and pooled curves are sums of these step functions, which gives the oracle over ALL
thresholds (finer than any grid). The script's grid is also rebuilt and evaluated on the
same step functions, and the saved out/oracle_<arm>.npz tables are compared cell by cell.
Run with cwd C:\\EEG, python -B."""
import sys
sys.dont_write_bytecode = True
import time
from multiprocessing import Pool
import numpy as np

sys.path.insert(0, r"C:\EEG")
from rs_common import (load_all, my_smooth, my_detect_from_smoothed, ev_count, runs, ARMS,
                       GROUPS, BUDGETS, OUT, HERE, SEC_PER_WIN, MINRUN, case_count_sm)


def block_steps(sm, y):
    """u (ascending distinct values), tp[k], fp[k] = score at threshold u[k]; ref."""
    truth = y.astype(bool)
    u = np.unique(sm)
    tp = np.zeros(len(u), np.int32)
    fp = np.zeros(len(u), np.int32)
    _, _, ref = ev_count(truth, np.zeros(len(y), bool))
    last_key, last = None, (0, 0)
    calls = 0
    n = len(sm)
    for k in range(len(u) - 1, -1, -1):           # high to low threshold
        lo, hi = runs(sm >= u[k])
        keep = (hi - lo) >= MINRUN
        lo, hi = lo[keep], hi[keep]
        key = (lo.tobytes(), hi.tobytes())
        if key != last_key:
            if len(lo) == 0:
                last = (0, 0)
            else:
                d = np.zeros(n + 1, np.int32)
                np.add.at(d, lo, 1)
                np.add.at(d, hi, -1)
                det = np.cumsum(d[:n]) > 0
                a, b, _ = ev_count(truth, det)
                last = (a, b)
                calls += 1
            last_key = key
        tp[k], fp[k] = last
    return u, tp, fp, ref, calls


def at(u, v, t):
    """Step function value at thresholds t (array): v at smallest u >= t, 0 above max."""
    idx = np.searchsorted(u, t, side="left")
    return np.concatenate([v, [0]])[idx]


def job(arm):
    t0 = time.time()
    P, _ = load_all()
    P = P[arm]
    cases = sorted(P)
    out = {}
    calls = 0
    for c in cases:
        bl = []
        for p, y in P[c]:
            sm = my_smooth(p)
            u, tp, fp, ref, n = block_steps(sm, np.asarray(y))
            calls += n
            bl.append((sm, np.asarray(y), u, tp, fp, ref))
        out[c] = bl
    return arm, out, calls, time.time() - t0


def main():
    with Pool(3) as pool:
        res = pool.map(job, ARMS, chunksize=1)
    lines = []

    def say(s=""):
        print(s, flush=True)
        lines.append(s)

    S = {arm: out for arm, out, _, _ in res}
    for arm, _, calls, dt in res:
        say("%-9s %7d timescoring calls, %.0f s" % (arm, calls, dt))
    cases = sorted(S["A0"])

    # spot-check step functions against direct scoring at random thresholds
    rng = np.random.default_rng(7)
    bad = 0
    for arm in ARMS:
        for c in rng.choice(cases, 4, replace=False):
            for t in np.concatenate([rng.uniform(0, 1, 3), rng.choice(
                    np.concatenate([b[2] for b in S[arm][c]]), 3)]):
                d = case_count_sm([b[0] for b in S[arm][c]], [b[1] for b in S[arm][c]], t)
                s = (sum(int(at(b[2], b[3], np.array([t]))[0]) for b in S[arm][c]),
                     sum(int(at(b[2], b[4], np.array([t]))[0]) for b in S[arm][c]))
                bad += (d[0], d[1]) != s
    say("step-function spot check vs direct scoring (7 arms x 4 cases x 6 thresholds): "
        "%d mismatches" % bad)

    tabs = {}
    for arm in ARMS:
        caseT = {}
        sec = np.array([sum(len(b[1]) for b in S[arm][c]) * SEC_PER_WIN for c in cases])
        ref = np.array([sum(b[5] for b in S[arm][c]) for c in cases])
        for c in cases:
            cand = np.unique(np.concatenate([b[2] for b in S[arm][c]] + [[np.inf]]))
            tp = sum(at(b[2], b[3], cand) for b in S[arm][c])
            fp = sum(at(b[2], b[4], cand) for b in S[arm][c])
            caseT[c] = (cand, tp, fp)
        allc = np.unique(np.concatenate([caseT[c][0] for c in cases]))
        TP = np.zeros(len(allc), np.int64)
        FP = np.zeros(len(allc), np.int64)
        for c in cases:
            for b in S[arm][c]:
                TP += at(b[2], b[3], allc)
                FP += at(b[2], b[4], allc)
        # the script's grid, rebuilt from its formula on my smoothed values
        sm_all = np.concatenate([b[0] for c in cases for b in S[arm][c]])
        q = 1.0 - np.logspace(np.log10(0.3), -6.5, 3000)
        grid = np.unique(np.concatenate([np.linspace(0.01, 0.999, 100), np.quantile(sm_all, q)]))
        gtp = np.stack([sum(at(b[2], b[3], grid) for b in S[arm][c]) for c in cases])
        gfp = np.stack([sum(at(b[2], b[4], grid) for b in S[arm][c]) for c in cases])
        tabs[arm] = dict(sec=sec, ref=ref, case=caseT, allc=allc, TP=TP, FP=FP,
                         grid=grid, gtp=gtp, gfp=gfp, n_distinct=len(allc))

    # compare with saved oracle tables
    say()
    for arm in ARMS:
        z = np.load(OUT / ("oracle_%s.npz" % arm))
        g = tabs[arm]
        same_grid = len(z["grid"]) == len(g["grid"]) and np.allclose(z["grid"], g["grid"], 0, 1e-12)
        msg = "%-9s saved grid %d pts, mine %d pts, grids equal %s" % (
            arm, len(z["grid"]), len(g["grid"]), same_grid)
        if same_grid:
            msg += "; tp cells differing %d, fp cells differing %d; ref equal %s, sec equal %s" % (
                int((z["tp"] != g["gtp"]).sum()), int((z["fp"] != g["gfp"]).sum()),
                np.array_equal(z["ref"], g["ref"]), np.allclose(z["sec"], g["sec"]))
        say(msg)
    say("distinct thresholds evaluated exactly per arm: " +
        ", ".join("%s %d" % (a, tabs[a]["n_distinct"]) for a in ARMS))

    def pp_exact(a, b):
        g = tabs[a]
        tot = 0
        per = []
        for i, c in enumerate(cases):
            cand, tp, fp = g["case"][c]
            ok = fp / (g["sec"][i] / 86400) <= b + 1e-12
            per.append(int(tp[ok].max()))
        return np.array(per)

    def pp_grid(a, b):
        g = tabs[a]
        fa = g["gfp"] / (g["sec"][:, None] / 86400)
        return np.where(fa <= b + 1e-12, g["gtp"], 0).max(1)

    def pooled_exact(a, b):
        g = tabs[a]
        fa = g["FP"] / (g["sec"].sum() / 86400)
        ok = fa <= b + 1e-12
        return g["TP"][ok].max() / g["ref"].sum(), float(g["allc"][ok][np.argmax(g["TP"][ok])])

    def pooled_grid(a, b):
        g = tabs[a]
        fa = g["gfp"].sum(0) / (g["sec"].sum() / 86400)
        ok = np.flatnonzero(fa <= b + 1e-12)
        return g["gtp"].sum(0)[ok].max() / g["ref"].sum()

    say()
    for title, fn in (("POOLED ORACLE, script grid", pooled_grid),
                      ("POOLED ORACLE, exact (every distinct threshold)",
                       lambda a, b: pooled_exact(a, b)[0]),
                      ("PER-PATIENT ORACLE, script grid",
                       lambda a, b: pp_grid(a, b).sum() / tabs[a]["ref"].sum()),
                      ("PER-PATIENT ORACLE, exact",
                       lambda a, b: pp_exact(a, b).sum() / tabs[a]["ref"].sum())):
        say(title)
        say("%-9s" % "" + "".join("%9s" % ("<=%d/d" % b) for b in BUDGETS))
        for a in ARMS:
            say("%-9s" % a + "".join("%9.3f" % fn(a, b) for b in BUDGETS))
        for gname, arms in GROUPS.items():
            say("%-9s" % gname + "".join("%9.3f" % np.mean([fn(a, b) for a in arms])
                                         for b in BUDGETS))
        say()

    say("PER-PATIENT ORACLE per case, exact: caught/events at <=10/d and <=25/d "
        "(trees: mean over seeds)")
    say("%-6s %4s %14s %14s %14s" % ("case", "ev", "A0", "Tamp", "Tfree"))
    for i, c in enumerate(cases):
        row = "%-6s %4d" % (c, tabs["A0"]["ref"][i])
        for gname, arms in GROUPS.items():
            v10 = np.mean([pp_exact(a, 10)[i] for a in arms])
            v25 = np.mean([pp_exact(a, 25)[i] for a in arms])
            row += " %6.2f / %5.2f" % (v10, v25)
        say(row)
    say()

    # window AUC ----------------------------------------------------------------------
    from scipy.stats import rankdata, wilcoxon
    from sklearn.metrics import roc_auc_score

    def mw_auc(y, s):
        r = rankdata(s)
        n1 = y.sum()
        n0 = len(y) - n1
        return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

    auc = {}
    worst = 0.0
    for gname, arms in GROUPS.items():
        vals = []
        for c in cases:
            per = []
            for a in arms:
                y = np.concatenate([b[1] for b in S[a][c]]).astype(int)
                s = np.concatenate([b[0] for b in S[a][c]])
                m, k = mw_auc(y, s), roc_auc_score(y, s)
                worst = max(worst, abs(m - k))
                per.append(m)
            vals.append(np.mean(per))
        auc[gname] = np.array(vals)
    say("WINDOW AUC (smoothed k=5, per held-out case; trees mean over seeds); "
        "rank-formula vs sklearn max diff %.2g" % worst)
    say("  median: " + ", ".join("%s %.3f" % (g, np.median(v)) for g, v in auc.items()))
    for g in ("Tamp", "Tfree"):
        d = auc[g] - auc["A0"]
        w = wilcoxon(d)
        say("  %s - A0: median %+.3f, better in %d of %d, Wilcoxon p %.3f (stat %.1f)"
            % (g, np.median(d), int((d > 0).sum()), len(d), w.pvalue, w.statistic))
    # subject-level variant: CHB01 and CHB21 averaged into one subject
    keep = [i for i, c in enumerate(cases) if c != "CHB21"]
    i01, i21 = cases.index("CHB01"), cases.index("CHB21")
    for g in ("Tamp", "Tfree"):
        d = auc[g] - auc["A0"]
        d2 = d.copy()
        d2[i01] = (d[i01] + d[i21]) / 2
        d2 = d2[keep]
        say("  (subject-level, CHB01+CHB21 averaged: %s - A0 better in %d of %d, Wilcoxon p %.3f)"
            % (g, int((d2 > 0).sum()), len(d2), wilcoxon(d2).pvalue))
    say("  per case A0/Tamp/Tfree: " + " ".join(
        "%s %.3f/%.3f/%.3f" % (c, auc["A0"][i], auc["Tamp"][i], auc["Tfree"][i])
        for i, c in enumerate(cases)))
    say("  per arm per case:")
    for a in ARMS:
        v = []
        for c in cases:
            y = np.concatenate([b[1] for b in S[a][c]]).astype(int)
            s = np.concatenate([b[0] for b in S[a][c]])
            v.append(mw_auc(y, s))
        say("   %-9s median %.3f | %s" % (a, np.median(v), " ".join("%.3f" % x for x in v)))
    (HERE / "rs_oracle_out.txt").write_text("\n".join(lines) + "\n", encoding="ascii")


if __name__ == "__main__":
    main()
