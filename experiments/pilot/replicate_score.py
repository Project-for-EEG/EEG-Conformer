"""
Replication, step 3: per-case score tables, then the rule in replicate_rule.txt.

Tables: for every arm, smoothing k and grid threshold, the repo's official count()
(timescoring) per case gives caught, false alarms, events and seconds. Every pooled,
subset, per-patient, leave-one-out and bootstrap number is then a sum over rows.

Reads <out>/all_predictions.npy (from replicate_heads.py). Writes tables_*.npz,
score_results.txt and score_results.json to <out>/. CPU only.
"""
import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
REPO = Path(r"C:\EEG")
sys.path.insert(0, str(REPO))
from score_official import count            # noqa: E402
from evaluate_temporal import smooth        # noqa: E402

TRIAL = ["CHB12", "CHB15", "CHB24", "CHB13", "CHB06", "CHB16"]
ARMS = ["A0", "E0_s0", "E0_s1", "E0_s2", "E1_s0", "E1_s1", "E1_s2"]
E0 = ["E0_s0", "E0_s1", "E0_s2"]
E1 = ["E1_s0", "E1_s1", "E1_s2"]
BUDGETS = (5, 10, 25, 50, 100)
N_RUN = 3
NBOOT = 2000
PRED = None


def grid_for(blocks_by_case, k):
    sm = np.concatenate([smooth(np.asarray(p, np.float64), k)
                         for c in blocks_by_case for p, _ in blocks_by_case[c]])
    q = 1.0 - np.logspace(np.log10(0.3), -6.5, 3000)
    return np.unique(np.concatenate([np.linspace(0.01, 0.999, 100), np.quantile(sm, q)]))


def build(job):
    path, arm, k = job
    P = np.load(path, allow_pickle=True).item()[arm]
    cases = sorted(P)
    blocks = {c: [(np.asarray(p, np.float64), y) for p, y in P[c]] for c in cases}
    grid = grid_for(blocks, k)
    tp = np.zeros((len(cases), len(grid)), np.int32)
    fp = np.zeros((len(cases), len(grid)), np.int32)
    ref = np.zeros(len(cases), np.int64)
    sec = np.zeros(len(cases), np.float64)
    for i, c in enumerate(cases):
        for g, t in enumerate(grid):
            a, b, r, s = count(blocks[c], float(t), k, N_RUN)
            tp[i, g], fp[i, g] = a, b
            if g == 0:
                ref[i], sec[i] = r, s
            else:
                assert r == ref[i] and abs(s - sec[i]) < 1e-6, (arm, c)
    return arm, k, {"grid": grid, "tp": tp, "fp": fp, "ref": ref, "sec": sec,
                    "cases": np.array(cases)}


def best(T, b, w=None):
    """Oracle: best pooled sensitivity with pooled FA/day <= b. Ties -> lowest FA."""
    w = np.ones(len(T["ref"])) if w is None else np.asarray(w, np.float64)
    tp, fp = w @ T["tp"], w @ T["fp"]
    ref, sec = w @ T["ref"], w @ T["sec"]
    fa = fp / (sec / 86400.0)
    ok = np.flatnonzero(fa <= b + 1e-12)
    m = tp[ok].max()
    cand = ok[tp[ok] == m]
    j = cand[np.argmin(fa[cand])]
    return m / ref, float(fa[j]), int(j)


def per_patient(T, b, w=None):
    """Each case at its own best threshold within its own FA/day <= b."""
    w = np.ones(len(T["ref"])) if w is None else np.asarray(w, np.float64)
    fa = T["fp"] / (T["sec"][:, None] / 86400.0)
    caught = np.where(fa <= b + 1e-12, T["tp"], 0).max(1)
    return float(w @ caught) / float(w @ T["ref"])


def verdict(d):
    if d[25] >= 0.05 and d[10] >= 0 and d[50] >= 0:
        return "GO"
    if d[25] <= 0:
        return "STOP"
    return "UNCLEAR"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=8)
    args = ap.parse_args()
    out = Path(args.out)
    path = str(out / "all_predictions.npy")
    lines, res = [], {}

    def say(s=""):
        print(s, flush=True)
        lines.append(s)

    jobs = [(path, a, k) for k in (5, 1) for a in ARMS]
    tabs = {}
    with Pool(args.procs) as pool:
        for arm, k, T in pool.imap_unordered(build, jobs):
            tabs[(arm, k)] = T
            np.savez_compressed(out / ("tables_%s_k%d.npz" % (arm, k)), **T)
            print("  table %s k=%d: %d thresholds" % (arm, k, len(T["grid"])), flush=True)
    cases = list(tabs[("A0", 5)]["cases"])
    T0 = tabs[("A0", 5)]
    for T in tabs.values():
        assert list(T["cases"]) == cases
        assert np.array_equal(T["ref"], T0["ref"]) and np.allclose(T["sec"], T0["sec"])
    new = np.array([c not in TRIAL for c in cases], np.float64)
    old = 1.0 - new
    say("cases %d, seizures %d, %.1f h | new folds: %d cases, %d seizures, %.1f h | "
        "pilot folds: %d cases, %d seizures, %.1f h"
        % (len(cases), T0["ref"].sum(), T0["sec"].sum() / 3600,
           new.sum(), new @ T0["ref"], new @ T0["sec"] / 3600,
           old.sum(), old @ T0["ref"], old @ T0["sec"] / 3600))
    say()

    hdr = "%-9s" % "" + "".join("%16s" % ("<=%d/d" % b) for b in BUDGETS)

    def table(title, k, w=None, fn="pooled"):
        say(title)
        say(hdr)
        r = {}
        for a in ARMS:
            T = tabs[(a, k)]
            row = "%-9s" % a
            for b in BUDGETS:
                if fn == "pooled":
                    s, f, _ = best(T, b, w)
                    row += "%16s" % ("%.3f (%5.1f)" % (s, f))
                else:
                    s = per_patient(T, b, w)
                    row += "%16s" % ("%.3f" % s)
                r[(a, b)] = s
            say(row)
        d = {b: float(np.mean([r[(a, b)] for a in E1]) - np.mean([r[(a, b)] for a in E0]))
             for b in BUDGETS}
        say("%-9s" % "E1-E0" + "".join("%16s" % ("%+.3f" % d[b]) for b in BUDGETS))
        say()
        return r, d

    # ---- the decision ---------------------------------------------------------
    _, d_all = table("MAIN TEST: all 23 folds pooled, k=5, oracle global threshold "
                     "(sens (achieved FA/day))", 5)
    _, d_new = table("FRESH-PATIENT GUARD: the 17 new folds only, k=5", 5, new)
    v_main = verdict(d_all)
    guard = d_new[25] > 0
    final = "UNCLEAR" if (v_main == "GO" and not guard) else v_main
    say("main test E1-E0: 10/d %+.3f | 25/d %+.3f | 50/d %+.3f -> %s"
        % (d_all[10], d_all[25], d_all[50], v_main))
    say("fresh-patient guard: 17 new folds E1-E0 at 25/d %+.3f -> %s"
        % (d_new[25], "passes" if guard else "fails"))
    say("VERDICT (rule fixed in replicate_rule.txt): %s" % final)
    say()
    res["main"], res["new"], res["verdict"] = d_all, d_new, final

    # ---- reported, not used for the verdict ------------------------------------
    _, d_old = table("Pilot 6 folds on this grid, k=5 (consistency with the pilot)", 5, old)
    _, d_pp = table("Per-patient oracle thresholds, all 23 folds, k=5 (sens)", 5, fn="pp")
    _, d_ppn = table("Per-patient oracle thresholds, 17 new folds, k=5 (sens)", 5, new, fn="pp")
    _, d_k1 = table("No smoothing (k=1), all 23 folds pooled", 1)
    _, d_k1n = table("No smoothing (k=1), 17 new folds", 1, new)
    res.update({"pilot6": d_old, "per_patient": d_pp, "per_patient_new": d_ppn,
                "k1": d_k1, "k1_new": d_k1n})
    for name, d in (("per-patient, all", d_pp), ("per-patient, new", d_ppn),
                    ("k=1, all", d_k1), ("k=1, new", d_k1n)):
        say("rule applied to %-18s %s" % (name + ":", verdict(d)))
    say()

    # subjects: CHB01 and CHB21 are one person
    subj = []
    for c in cases:
        s = "CHB01" if c == "CHB21" else c
        if s not in subj:
            subj.append(s)
    S = np.array([[1.0 if (c == s or (s == "CHB01" and c == "CHB21")) else 0.0
                   for c in cases] for s in subj])          # subjects x cases

    # leave one subject out
    say("LEAVE ONE SUBJECT OUT, all 23 folds, k=5: E1-E0 with that subject removed")
    loo = {}
    for i, s in enumerate(subj):
        w = 1.0 - S[i]
        dd = {}
        for b in (10, 25, 50):
            e1 = np.mean([best(tabs[(a, 5)], b, w)[0] for a in E1])
            e0 = np.mean([best(tabs[(a, 5)], b, w)[0] for a in E0])
            dd[b] = float(e1 - e0)
        loo[s] = dd
    for b in (10, 25, 50):
        vals = [loo[s][b] for s in subj]
        lo, hi = int(np.argmin(vals)), int(np.argmax(vals))
        say("  %2d/d: smallest %+.3f (without %s), largest %+.3f (without %s)"
            % (b, vals[lo], subj[lo], vals[hi], subj[hi]))
    counts = {}
    for s in subj:
        counts[verdict(loo[s])] = counts.get(verdict(loo[s]), 0) + 1
    say("  verdict with one subject removed: %s" % ", ".join(
        "%s %d" % (k, v) for k, v in sorted(counts.items())))
    say()
    res["loo"] = loo

    # subject bootstrap, budget re-chosen in every draw
    rng = np.random.default_rng(20261104)
    idx = rng.integers(0, len(subj), size=(NBOOT, len(subj)))
    Wsub = np.zeros((NBOOT, len(subj)))
    for r in range(NBOOT):
        np.add.at(Wsub[r], idx[r], 1)
    Wc = Wsub @ S                                                 # draws x cases
    boot = {}
    for b in (10, 25, 50):
        sens = {}
        for a in E0 + E1:
            T = tabs[(a, 5)]
            o = np.empty(NBOOT)
            for s0 in range(0, NBOOT, 250):
                w = Wc[s0:s0 + 250]
                TP, FP = w @ T["tp"], w @ T["fp"]
                REF, SEC = w @ T["ref"], w @ T["sec"]
                ok = FP / (SEC[:, None] / 86400.0) <= b + 1e-12
                o[s0:s0 + 250] = np.where(ok, TP, -1).max(1) / REF
            sens[a] = o
        boot[b] = np.mean([sens[a] for a in E1], 0) - np.mean([sens[a] for a in E0], 0)
    say("SUBJECT BOOTSTRAP, all 23 folds, k=5, %d draws of %d subjects" % (NBOOT, len(subj)))
    for b in (10, 25, 50):
        x = boot[b]
        lo, hi = np.percentile(x, [2.5, 97.5])
        say("  %2d/d: point %+.3f | 95%% interval [%+.3f, %+.3f] | P(<=0) %.3f | P(>=0.05) %.3f"
            % (b, d_all[b], lo, hi, (x <= 0).mean(), (x >= 0.05).mean()))
    vs = [verdict({b: boot[b][r] for b in (10, 25, 50)}) for r in range(NBOOT)]
    say("  verdict share over draws: " + ", ".join(
        "%s %.0f%%" % (v, 100.0 * vs.count(v) / NBOOT) for v in ("GO", "UNCLEAR", "STOP")))
    say()
    res["boot"] = {b: [float(np.percentile(boot[b], 2.5)), float(np.percentile(boot[b], 97.5)),
                       float((boot[b] <= 0).mean()), float((boot[b] >= 0.05).mean())]
                   for b in (10, 25, 50)}

    # where the gain sits: per case at each arm's own pooled 25/d threshold
    say("PER CASE at each arm's pooled 25/day threshold, k=5 (seed means): caught E0, E1, "
        "events | FA E0, E1")
    e0c = np.mean([tabs[(a, 5)]["tp"][:, best(tabs[(a, 5)], 25)[2]] for a in E0], 0)
    e1c = np.mean([tabs[(a, 5)]["tp"][:, best(tabs[(a, 5)], 25)[2]] for a in E1], 0)
    e0f = np.mean([tabs[(a, 5)]["fp"][:, best(tabs[(a, 5)], 25)[2]] for a in E0], 0)
    e1f = np.mean([tabs[(a, 5)]["fp"][:, best(tabs[(a, 5)], 25)[2]] for a in E1], 0)
    for i, c in enumerate(cases):
        say("  %-6s%s %6.2f %6.2f %4d | %6.1f %6.1f"
            % (c, "*" if c in TRIAL else " ", e0c[i], e1c[i], T0["ref"][i], e0f[i], e1f[i]))
    say("  total   %6.2f %6.2f %4d | %6.1f %6.1f   (* = pilot fold)"
        % (e0c.sum(), e1c.sum(), T0["ref"].sum(), e0f.sum(), e1f.sum()))

    (out / "score_results.txt").write_text("\n".join(lines) + "\n", encoding="ascii")
    (out / "score_results.json").write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
