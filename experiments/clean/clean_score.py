"""
Scoring for clean_rule.txt v2 (sections 3-6). CPU only. Run with cwd C:\\EEG.

Reads OLD\\out\\preds_{a,b,c,d}.npy + thresholds (A0, Tamp, Tfree, Tall) and every
NEW\\trees_out\\preds_<tag>.npy + thresholds (Trel, Trelf, later N1). OLD is only read.
Writes NEW\\score_out\\clean_results.txt and clean_results.json.

Order: checks (arms, labels, block names, artifact rule) -> continuity check (must pass)
-> curves, README point, questions, secondary views. Questions whose arms are missing
are skipped and say so.
"""
import argparse
import json
import re
import sys
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score

sys.dont_write_bytecode = True
REPO = Path(r"C:\EEG")
sys.path.insert(0, str(REPO))
from score_official import count                              # noqa: E402
from evaluate_temporal import smooth                          # noqa: E402
from finetune_per_patient import patient_files, find_split    # noqa: E402

OLD = Path(r"C:\Users\user\AppData\Local\Temp\claude\36c1f29744f93fe6"
           r"\b3889703-629c-4c5a-b583-69f722743ff4\scratchpad\trees")
NEW = Path(r"C:\Users\user\AppData\Local\Temp\claude\36c1f29744f93fe6"
           r"\c7848ca6-bfef-414b-a321-d6686c1a43f8\scratchpad\clean")
TARGETS = ("0.05", "0.1", "0.15", "0.2", "0.3", "0.4", "0.5", "0.75", "1.0", "1.5", "2.0",
           "3.0", "4.0", "5.0")
GRID = (50, 60, 75, 90, 100, 115, 125, 150)
BUDGETS = (5, 10, 25, 50, 100)
K, N_RUN, NBOOT, MARGIN = 5, 3, 2000, 0.05
ART_MIN_CH, ART_FRAC = 8, 0.5
GROUPS = {"A0": ["A0"], "N1": ["N1"],
          "Tamp": ["Tamp_s0", "Tamp_s1", "Tamp_s2"],
          "Tfree": ["Tfree_s0", "Tfree_s1", "Tfree_s2"],
          "Tall": ["Tall"],
          "Trel": ["Trel_s0", "Trel_s1", "Trel_s2"],
          "Trelf": ["Trelf_s0"]}
EARLY = ["CHB02", "CHB04", "CHB05", "CHB09", "CHB10", "CHB12", "CHB15", "CHB16", "CHB24"]
SAME15 = ["CHB03", "CHB06", "CHB07", "CHB08", "CHB11", "CHB13", "CHB14", "CHB17", "CHB18",
          "CHB19", "CHB20", "CHB22", "CHB23"]
MOTIVATED = ["CHB06", "CHB15"]

LINES = []


def say(s=""):
    print(s, flush=True)
    LINES.append(s)


# ---------------------------------------------------------------- loading and checks
def load_arms():
    P, T, src = {}, {}, {}
    parts = [(OLD / "out", t) for t in "abcd"]
    parts += [(NEW / "trees_out", f.stem[len("preds_"):])
              for f in sorted((NEW / "trees_out").glob("preds_*.npy"))]
    for d, tag in parts:
        part = np.load(d / ("preds_%s.npy" % tag), allow_pickle=True).item()
        tpart = json.loads((d / ("thresholds_%s.json" % tag)).read_text())
        for a in part:
            assert a not in P, ("arm in two files", a)
            P[a], T[a], src[a] = part[a], tpart[a], "%s\\preds_%s.npy" % (d.name, tag)
    return P, T, src


def block_names(cases, P):
    """Rebuild the file stem and post-split start of every stored test block."""
    names = {}
    for c in cases:
        files = patient_files(c)
        fi, wi = find_split(files, 1)
        out = []
        for i, f in enumerate(files):
            if i < fi:
                continue
            with np.load(OLD / "feats" / ("%s__%s.npz" % (c, f.stem))) as d:
                lab = d["labels"]
                flat = np.isnan(d["feats"][:, :, 0])
            lo = wi if i == fi else 0
            if len(lab) - lo >= 10:
                out.append({"stem": f.stem, "lo": lo, "labels": lab[lo:], "flat": flat[lo:]})
        assert len(out) == len(P["A0"][c]), (c, len(out), len(P["A0"][c]))
        for b, (_, y) in zip(out, P["A0"][c]):
            assert np.array_equal(np.asarray(b["labels"], np.int64), np.asarray(y, np.int64)), \
                (c, b["stem"])
        names[c] = out
    return names


def blocks_of(P_arm_c):
    return [(np.asarray(p, np.float64), np.asarray(y)) for p, y in P_arm_c]


# ---------------------------------------------------------------- tables and curves
def deploy_table(P_arm, T_arm, cases, keep):
    tp = np.zeros((len(cases), len(TARGETS)), np.int64)
    fp = np.zeros_like(tp)
    ref = np.zeros(len(cases), np.int64)
    sec = np.zeros(len(cases))
    for i, c in enumerate(cases):
        blocks = [b for b, k in zip(blocks_of(P_arm[c]), keep[c]) if k]
        for j, tg in enumerate(TARGETS):
            a, b, r, s = count(blocks, float(T_arm[c][tg]), K, N_RUN)
            tp[i, j], fp[i, j] = a, b
            ref[i], sec[i] = r, s
    return {"tp": tp, "fp": fp, "ref": ref, "sec": sec}


def curve(D, w):
    tp, fp, ref, sec = w @ D["tp"], w @ D["fp"], w @ D["ref"], w @ D["sec"]
    return fp / (sec / 86400.0), tp / ref


def interp(fa, sens, levels, clamp):
    """Points sorted by FA/day; equal FA keeps the highest sensitivity; linear interpolation.
    Outside the curve's range: NaN, or the end value when clamp is True."""
    order = np.lexsort((-sens, fa))
    fa, sens = fa[order], sens[order]
    first = np.concatenate([[True], np.diff(fa) > 0])
    fa, sens = fa[first], sens[first]
    L = np.asarray(levels, np.float64)
    if clamp:
        return np.interp(L, fa, sens)
    return np.interp(L, fa, sens, left=np.nan, right=np.nan)


def group_at(D, arms, w, levels, clamp):
    """Mean over the given seeds at each level; NaN where any seed is NaN."""
    v = np.array([interp(*curve(D[a], w), levels, clamp) for a in arms])
    return v.mean(0)


def decide(point, lo, hi):
    if point >= MARGIN and lo > 0:
        return "BETTER"
    if point <= -MARGIN and hi < 0:
        return "WORSE"
    if lo >= -MARGIN and hi <= MARGIN:
        return "SAME"
    return "UNCLEAR"


class Bench:
    """One dataset version (155 or 168 events) over a set of cases, with its bootstrap."""

    def __init__(self, D, cases, use_cases, label, rng_seed=7):
        self.D, self.cases, self.label = D, cases, label
        subj = []
        for c in use_cases:
            s = "CHB01" if c == "CHB21" else c
            if s not in subj:
                subj.append(s)
        self.subj = subj
        S = np.array([[1.0 if c in use_cases and (c == s or (s == "CHB01" and c == "CHB21"))
                       else 0.0 for c in cases] for s in subj])
        self.w1 = S.sum(0)
        rng = np.random.default_rng(rng_seed)
        idx = rng.integers(0, len(subj), size=(NBOOT, len(subj)))
        Ws = np.zeros((NBOOT, len(subj)))
        for r in range(NBOOT):
            np.add.at(Ws[r], idx[r], 1)
        self.W = Ws @ S
        self.SI = rng.integers(0, 3, size=(NBOOT, 3))

    def reach(self, g):
        arms = GROUPS[g]
        return [L for L in GRID if not np.isnan(group_at(self.D, arms, self.w1, [L], False)[0])]

    def compare(self, gx, gy):
        levels = [L for L in self.reach(gx) if L in self.reach(gy)]
        res = {"x": gx, "y": gy, "levels": levels, "set": self.label}
        if not levels:
            res.update(decision="UNDETERMINED")
            return res
        ax, ay = GROUPS[gx], GROUPS[gy]
        vx = group_at(self.D, ax, self.w1, levels, False)
        vy = group_at(self.D, ay, self.w1, levels, False)
        point = float(np.mean(vx - vy))
        per_seed = {}
        for a in ax:
            per_seed[a] = float(np.mean(group_at(self.D, [a], self.w1, levels, False) - vy))
        bn, bc, partial = np.full(NBOOT, np.nan), np.full(NBOOT, np.nan), 0
        for r in range(NBOOT):
            w = self.W[r]
            sx = [ax[i] for i in self.SI[r]] if len(ax) == 3 else ax
            sy = [ay[i] for i in self.SI[r]] if len(ay) == 3 else ay
            d = group_at(self.D, sx, w, levels, False) - group_at(self.D, sy, w, levels, False)
            ok = ~np.isnan(d)
            if ok.any():
                bn[r] = d[ok].mean()
                partial += int(ok.sum() < len(levels))
            bc[r] = np.mean(group_at(self.D, sx, w, levels, True)
                            - group_at(self.D, sy, w, levels, True))
        kept = ~np.isnan(bn)
        lo, hi = np.percentile(bn[kept], [2.5, 97.5])
        clo, chi = np.percentile(bc, [2.5, 97.5])
        d1, d2 = decide(point, lo, hi), decide(point, clo, chi)
        res.update(point=point, per_level=dict(zip(levels, (vx - vy).tolist())),
                   per_seed=per_seed, lo=float(lo), hi=float(hi), clo=float(clo),
                   chi=float(chi), partial=partial, dropped=int((~kept).sum()),
                   decision=d1 if d1 == d2 else "UNCLEAR")
        return res


def show(res):
    if res["decision"] == "UNDETERMINED":
        say("    [%s] %s vs %s: no level of the grid is reached by both -> UNDETERMINED"
            % (res["set"], res["x"], res["y"]))
        return
    say("    [%s] %s - %s = %+.1f points over levels %s | 95%% [%+.1f, %+.1f] | clamped "
        "[%+.1f, %+.1f] | partial %d, dropped %d draws -> %s"
        % (res["set"], res["x"], res["y"], 100 * res["point"], res["levels"],
           100 * res["lo"], 100 * res["hi"], 100 * res["clo"], 100 * res["chi"],
           res["partial"], res["dropped"], res["decision"]))
    say("        per level: " + ", ".join("%d: %+.1f" % (L, 100 * v)
                                         for L, v in res["per_level"].items()))
    if len(res["per_seed"]) > 1:
        say("        per seed:  " + ", ".join("%s %+.1f" % (a, 100 * v)
                                             for a, v in res["per_seed"].items()))


def combine(r155, r168):
    a, b = r155["decision"], r168["decision"]
    if a == b:
        return a
    if a in ("BETTER", "WORSE") or b in ("BETTER", "WORSE"):
        return "UNCLEAR (%s with 155 events, %s with 168: depends on the artifact files)" % (a, b)
    return "UNCLEAR"


# ---------------------------------------------------------------- secondary views
def oracle_job(job):
    arm, blocks_by_case, cases = job
    sm = np.concatenate([smooth(p, K) for c in cases for p, _ in blocks_by_case[c]])
    q = 1.0 - np.logspace(np.log10(0.3), -6.5, 3000)
    grid = np.unique(np.concatenate([np.linspace(0.01, 0.999, 100), np.quantile(sm, q)]))
    tp = np.zeros((len(cases), len(grid)), np.int32)
    fp = np.zeros_like(tp)
    ref = np.zeros(len(cases), np.int64)
    sec = np.zeros(len(cases))
    for i, c in enumerate(cases):
        for g, t in enumerate(grid):
            a, b, r, s = count(blocks_by_case[c], float(t), K, N_RUN)
            tp[i, g], fp[i, g] = a, b
            ref[i], sec[i] = r, s
    return arm, {"tp": tp, "fp": fp, "ref": ref, "sec": sec}


def subjects(cases):
    subj = []
    for c in cases:
        s = "CHB01" if c == "CHB21" else c
        if s not in subj:
            subj.append(s)
    return subj


def per_subject(values, cases, how="mean", ref=None):
    """values per case -> per subject (CHB01+CHB21 merged)."""
    out = []
    for s in subjects(cases):
        idx = [i for i, c in enumerate(cases) if c == s or (s == "CHB01" and c == "CHB21")]
        if how == "mean":
            out.append(np.mean([values[i] for i in idx]))
        else:
            out.append(sum(values[i] for i in idx) / max(1, sum(ref[i] for i in idx)))
    return np.array(out)


def wtest(d):
    d = np.asarray(d, np.float64)
    if np.allclose(d, 0):
        return float("nan")
    return float(wilcoxon(d, zero_method="zsplit").pvalue)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--no-oracle", action="store_true", help="skip the slow oracle (tests)")
    ap.add_argument("--outdir", default=str(NEW / "score_out"))
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    res = {}

    P, T, src = load_arms()
    cases = sorted(P["A0"])
    say("clean_score.py -- clean_rule.txt v2")
    say("arms: " + ", ".join("%s (%s)" % (a, src[a]) for a in P))
    for a in P:
        assert sorted(P[a]) == cases, a
        for c in cases:
            assert len(P[a][c]) == len(P["A0"][c]), (a, c)
            for (p, y), (_, y0) in zip(P[a][c], P["A0"][c]):
                assert len(p) == len(y0) and np.array_equal(np.asarray(y), np.asarray(y0)), (a, c)
                assert not np.isnan(np.asarray(p, np.float64)).any(), (a, c)
    say("CHECK every arm has the same %d cases, block lengths and labels as A0: PASS"
        % len(cases))
    groups = {g: arms for g, arms in GROUPS.items() if all(a in P for a in arms)}
    say("groups present: " + ", ".join("%s (%d)" % (g, len(a)) for g, a in groups.items()))

    # ---- N1 alignment with the committed README procedure ---------------------------
    if "N1" in P:
        ref_path = NEW / "endtoend_uncalibrated_fixed15.npz"
        ref = np.load(ref_path, allow_pickle=True)["data"].item()
        worst, tworst = 0.0, 0.0
        for c in cases:
            assert len(P["N1"][c]) == len(ref[c]["blocks"]), c
            for (p1, y1), (p2, y2) in zip(P["N1"][c], ref[c]["blocks"]):
                assert np.array_equal(np.asarray(y1, np.int64), np.asarray(y2, np.int64))
                worst = max(worst, float(np.max(np.abs(np.asarray(p1) - np.asarray(p2)))))
            tworst = max(tworst, abs(T["N1"][c]["2.0"] - float(ref[c]["t"])))
        ok = worst < 1e-4 and tworst < 1e-4
        say("CHECK N1 vs %s: max |dp| %.2e, max |dt@2FA/h| %.2e (%s)"
            % (ref_path.name, worst, tworst, "PASS" if ok else "FAIL"))
        if not ok:
            raise SystemExit("N1 alignment check failed")

    # ---- block names and the artifact rule ------------------------------------------
    names = block_names(cases, P)
    say("CHECK block-to-file mapping rebuilt for every case, labels equal block by block: PASS")
    keep168 = {c: [True] * len(names[c]) for c in cases}
    keep155, hits, partial_hits = {}, [], []
    for c in cases:
        keep155[c] = []
        for b in names[c]:
            frac = b["flat"].mean(0)
            n_ch = int((frac > ART_FRAC).sum())
            if n_ch >= 1:
                partial_hits.append("%s (%d channels)" % (b["stem"], n_ch))
            drop = n_ch >= ART_MIN_CH
            if drop:
                hits.append(b["stem"])
            keep155[c].append(not drop)
    say("ARTIFACT RULE (>= %d of 23 channels flat in > %d%% of a test recording's windows)"
        % (ART_MIN_CH, 100 * ART_FRAC))
    say("  test recordings with any such channel: " + ", ".join(partial_hits))
    say("  excluded: " + ", ".join(hits))
    dropped_events = 0
    for c in cases:
        for (p, y), k in zip(blocks_of(P["A0"][c]), keep155[c]):
            if not k:
                dropped_events += count([(p, y)], 0.5, K, N_RUN)[2]
    say("  reference events in excluded recordings: %d" % dropped_events)
    assert dropped_events == 13, dropped_events
    res["artifact"] = {"excluded": hits, "any": partial_hits, "events": dropped_events}

    # ---- deployable tables -----------------------------------------------------------
    D = {"168": {a: deploy_table(P[a], T[a], cases, keep168) for a in P},
         "155": {a: deploy_table(P[a], T[a], cases, keep155) for a in P}}
    for k in D:
        a0 = D[k]["A0"]
        say("set %s: %d events, %.1f h of test recording"
            % (k, a0["ref"].sum(), a0["sec"].sum() / 3600))
    assert D["155"]["A0"]["ref"].sum() == 155 and D["168"]["A0"]["ref"].sum() == 168

    def row_strings(Dk, g):
        arms = GROUPS[g]
        out = []
        for j in range(len(TARGETS)):
            s = np.mean([Dk[a]["tp"][:, j].sum() / Dk[a]["ref"].sum() for a in arms])
            f = np.mean([Dk[a]["fp"][:, j].sum() / (Dk[a]["sec"].sum() / 86400) for a in arms])
            out.append("%.3f (%6.1f)" % (s, f))
        return out

    # ---- continuity check -----------------------------------------------------------
    old = (OLD / "out" / "score_results_main.txt").read_text().splitlines()
    i0 = next(i for i, l in enumerate(old) if l.startswith("DEPLOYABLE CURVES"))
    old_rows = {}
    for line in old[i0 + 2:i0 + 2 + len(TARGETS)]:
        tg = line.split()[0]
        old_rows[tg] = re.findall(r"\d\.\d{3} \(\s*[\d.]+\)", line)
    bad = 0
    for gi, g in enumerate(["A0", "Tamp", "Tfree"]):
        mine = row_strings(D["168"], g)
        for j, tg in enumerate(TARGETS):
            if re.sub(r"\s+", " ", mine[j]) != re.sub(r"\s+", " ", old_rows[tg][gi]):
                bad += 1
                say("  MISMATCH %s %s: %s vs old %s" % (g, tg, mine[j], old_rows[tg][gi]))
    say("CONTINUITY CHECK (168 events, A0/Tamp/Tfree vs OLD\\out\\score_results_main.txt, "
        "%d cells): %s" % (3 * len(TARGETS), "PASS" if bad == 0 else "FAIL"))
    if bad:
        raise SystemExit("continuity check failed; no result is read")
    say()

    # ---- curves and README point ------------------------------------------------------
    for k in ("155", "168"):
        say("DEPLOYABLE CURVES, %s events. Cells: sensitivity (false alarms/day); "
            "trees = mean over seeds" % k)
        say("%-7s" % "target" + "".join("%17s" % g for g in groups))
        rows = {g: row_strings(D[k], g) for g in groups}
        for j, tg in enumerate(TARGETS):
            say("%-7s" % tg + "".join("%17s" % rows[g][j] for g in groups))
        say()
    j2 = TARGETS.index("2.0")
    say("README OPERATING POINT (target 2 FA/h): caught / events, false alarms per day")
    for a in P:
        cells = []
        for k in ("155", "168"):
            d = D[k][a]
            cells.append("%3d/%d %6.1f/day" % (d["tp"][:, j2].sum(), d["ref"].sum(),
                                               d["fp"][:, j2].sum() / (d["sec"].sum() / 86400)))
        say("  %-9s %s   |   %s" % (a, cells[0], cells[1]))
    say()

    # ---- questions ---------------------------------------------------------------------
    B = {k: Bench(D[k], cases, cases, k) for k in ("155", "168")}
    say("Reach on the grid %s (levels every seed reaches), 155 / 168 events:" % (GRID,))
    for g in groups:
        say("  %-6s %s / %s" % (g, B["155"].reach(g), B["168"].reach(g)))
    say()
    Q = [("Q1 (confirmatory) Tamp vs N1", "Tamp", "N1"),
         ("Q2 Trel vs Tamp", "Trel", "Tamp"),
         ("Q3 N1 vs A0 (all folds)", "N1", "A0"),
         ("Q4 Trel vs N1", "Trel", "N1")]
    res["questions"] = {}
    for title, gx, gy in Q:
        if gx not in groups or gy not in groups:
            say("%s: not available yet (%s missing)" % (title, gx if gx not in groups else gy))
            continue
        say(title)
        r155, r168 = B["155"].compare(gx, gy), B["168"].compare(gx, gy)
        show(r155)
        show(r168)
        dec = combine(r155, r168)
        say("    DECISION: %s" % dec)
        res["questions"][title] = {"155": r155, "168": r168, "decision": dec}
        if gx == "Trel" or gy == "Trel":
            sub = [c for c in cases if c not in MOTIVATED]
            for k in ("155", "168"):
                show(Bench(D[k], cases, sub, k + " without CHB06, CHB15").compare(gx, gy))
        if title.startswith("Q3"):
            for lab, sub in (("(a) 9 early-stopped folds", EARLY),
                             ("(b) 13 folds 15/15 in both runs", SAME15)):
                for k in ("155", "168"):
                    show(Bench(D[k], cases, sub, "%s %s" % (k, lab)).compare("N1", "A0"))
        say()
    say("Q1 is the only confirmatory question. Q2-Q4 are secondary, same rule, no "
        "multiplicity correction. SAME cannot occur at this sample size; UNCLEAR means no "
        "evidence either way. Intervals exclude the network's run-to-run noise.")
    say()
    if "Trelf" in groups and "Trel" in groups:
        say("SECONDARY (no decision): Trelf_s0 vs Trel, what causality costs")
        for k in ("155", "168"):
            show(B[k].compare("Trelf", "Trel"))
        say()

    # ---- window AUC and per-patient oracle (secondary, exploratory) ---------------------
    pairs = [(x, y) for x, y in (("Tamp", "N1"), ("Trel", "N1"), ("Trel", "Tamp"), ("N1", "A0"))
             if x in groups and y in groups]
    for k, keep in (("155", keep155), ("168", keep168)):
        bl = {a: {c: [b for b, kk in zip(blocks_of(P[a][c]), keep[c]) if kk] for c in cases}
              for a in P}
        auc = {}
        for g, arms in groups.items():
            per_case = [np.mean([roc_auc_score(np.concatenate([y for _, y in bl[a][c]]),
                                               np.concatenate([smooth(p, K) for p, _ in bl[a][c]]))
                                 for a in arms]) for c in cases]
            auc[g] = per_subject(per_case, cases)
        say("WINDOW AUC per subject, %s events (smoothed k=5, trees mean over seeds, "
            "CHB01+CHB21 by mean): medians " % k
            + ", ".join("%s %.3f" % (g, np.median(v)) for g, v in auc.items()))
        for x, y in pairs:
            d = auc[x] - auc[y]
            say("  %s - %s: median %+.3f, better in %d of %d subjects, Wilcoxon zsplit p %.3f "
                "(unadjusted)" % (x, y, np.median(d), int((d > 0).sum()), len(d), wtest(d)))
        if "N1" in groups:
            for lab, sub in (("(a) early-stopped", EARLY), ("(b) 15/15 in both", SAME15)):
                ix = [i for i, s in enumerate(subjects(cases)) if s in sub]
                d = auc["N1"][ix] - auc["A0"][ix]
                say("  Q3 %s: N1 - A0 per fold AUC: %s" % (lab, " ".join("%+.3f" % v for v in d)))
        res.setdefault("auc", {})[k] = {g: v.tolist() for g, v in auc.items()}
        if args.no_oracle:
            say("(oracle skipped: --no-oracle)")
            say()
            continue

        with Pool(args.procs) as pool:
            tabs = dict(pool.map(oracle_job, [(a, bl[a], cases) for a in P]))
        say("PER-PATIENT ORACLE, %s events: pooled sensitivity with each case's own best "
            "threshold within the budget" % k)
        say("%-7s" % "" + "".join("%9s" % ("<=%d/d" % b) for b in BUDGETS))
        orc = {}
        for g, arms in groups.items():
            vals = []
            for b in BUDGETS:
                per_arm = []
                for a in arms:
                    Tb = tabs[a]
                    fa = Tb["fp"] / (Tb["sec"][:, None] / 86400.0)
                    best = np.where(fa <= b + 1e-12, Tb["tp"], 0).max(1)
                    per_arm.append(per_subject(best, cases, "ratio", Tb["ref"]))
                vals.append(np.mean(per_arm, 0))
            orc[g] = vals
            tot = tabs[arms[0]]["ref"].sum()
            say("%-7s" % g + "".join(
                "%9.3f" % np.mean([np.where(tabs[a]["fp"] / (tabs[a]["sec"][:, None] / 86400.0)
                                            <= b + 1e-12, tabs[a]["tp"], 0).max(1).sum() / tot
                                   for a in arms]) for b in BUDGETS))
        for x, y in pairs:
            say("  %s - %s per-subject oracle, Wilcoxon zsplit p (unadjusted) at %s: %s"
                % (x, y, "/".join(str(b) for b in BUDGETS),
                   " ".join("%.3f" % wtest(orc[x][i] - orc[y][i]) for i in range(len(BUDGETS)))))
        say()

    (outdir / "clean_results.txt").write_text("\n".join(LINES) + "\n", encoding="ascii")
    (outdir / "clean_results.json").write_text(json.dumps(res, indent=1, default=str))
    say("saved %s" % (outdir / "clean_results.txt"))


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        main()
