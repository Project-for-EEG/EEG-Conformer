"""C8 oracle detail and C4 robustness (Trel vs A0). Read-only, 4 CPU processes.
Run with cwd C:\EEG and python -B."""
import json, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
NEW = Path(r"C:\Users\user\AppData\Local\Temp\claude\36c1f29744f93fe6"
           r"\c7848ca6-bfef-414b-a321-d6686c1a43f8\scratchpad\clean")
OUT = NEW / "verify2" / "claims"
sys.path.insert(0, str(NEW))
sys.dont_write_bytecode = True
import clean_score as cs  # noqa: E402

def main():
    P, T, src = cs.load_arms()
    cases = sorted(P["A0"])
    names = cs.block_names(cases, P)
    keep = {"168": {c: [True] * len(names[c]) for c in cases},
            "155": {c: [int((b["flat"].mean(0) > cs.ART_FRAC).sum()) < cs.ART_MIN_CH for b in names[c]]
                    for c in cases}}
    lines = []
    def say(s):
        print(s, flush=True); lines.append(s)
    # ---- C4 robustness: Trel vs A0 (a second, independent network training run)
    arms = ["A0", "N1", "Trel_s0", "Trel_s1", "Trel_s2", "Tamp_s0", "Tamp_s1", "Tamp_s2"]
    D = {k: {a: cs.deploy_table(P[a], T[a], cases, keep[k]) for a in arms} for k in keep}
    for k in ("155", "168"):
        B = cs.Bench(D[k], cases, cases, k)
        for gx, gy in (("Trel", "A0"), ("Tamp", "A0")):
            r = B.compare(gx, gy)
            say("[%s] %s - %s = %+.2f over %s | 95%% [%+.2f, %+.2f] clamped [%+.2f, %+.2f] -> %s | per level %s"
                % (k, gx, gy, 100 * r["point"], r["levels"], 100 * r["lo"], 100 * r["hi"],
                   100 * r["clo"], 100 * r["chi"], r["decision"],
                   {L: round(100 * v, 2) for L, v in r["per_level"].items()}))
        # deployable sensitivity at 50 / 100 per day for comparison with oracle
        w1 = np.ones(len(cases))
        for g in ("N1", "Tamp", "A0", "Trel"):
            v = np.mean([cs.interp(*cs.curve(D[k][a], w1), [50, 60, 100], False) for a in cs.GROUPS[g]], 0)
            say("[%s] deployable %s at 50/60/100 per day: %s" % (k, g, np.round(v, 4).tolist()))
    # ---- C8: per-subject oracle for N1, A0 and Tamp seeds
    for k in ("155", "168"):
        bl = {a: {c: [b for b, kk in zip(cs.blocks_of(P[a][c]), keep[k][c]) if kk] for c in cases}
              for a in ("N1", "A0", "Tamp_s0", "Tamp_s1", "Tamp_s2")}
        with Pool(4) as pool:
            tabs = dict(pool.map(cs.oracle_job, [(a, bl[a], cases) for a in bl]))
        subj = cs.subjects(cases)
        per = {}
        for a, Tb in tabs.items():
            fa = Tb["fp"] / (Tb["sec"][:, None] / 86400.0)
            per[a] = {}
            for b in cs.BUDGETS:
                okm = fa <= b + 1e-12
                tpb = np.where(okm, Tb["tp"], -1)
                gi = tpb.argmax(1)
                best = Tb["tp"][np.arange(len(cases)), gi]
                fpb = Tb["fp"][np.arange(len(cases)), gi]
                pooled_fa = fpb.sum() / (Tb["sec"].sum() / 86400)
                per[a][b] = {"sens": best.sum() / Tb["ref"].sum(), "pooled_fa": pooled_fa,
                             "subj": cs.per_subject(best, cases, "ratio", Tb["ref"])}
        for b in cs.BUDGETS:
            n1 = per["N1"][b]; a0 = per["A0"][b]
            ts = np.mean([per[a][b]["subj"] for a in cs.GROUPS["Tamp"]], 0)
            tsens = np.mean([per[a][b]["sens"] for a in cs.GROUPS["Tamp"]])
            tfa = np.mean([per[a][b]["pooled_fa"] for a in cs.GROUPS["Tamp"]])
            d = ts - n1["subj"]
            say("[%s] oracle <=%d/d: N1 %.4f (pooled FA %.1f/d), Tamp %.4f (pooled FA %.1f/d), A0 %.4f | Tamp-N1 %+.2f pts | per subject Tamp>N1 %d, <%d, tie %d"
                % (k, b, n1["sens"], n1["pooled_fa"], tsens, tfa, a0["sens"], 100 * (tsens - n1["sens"]),
                   (d > 1e-12).sum(), (d < -1e-12).sum(), (np.abs(d) <= 1e-12).sum()))
    (OUT / "c8_oracle.txt").write_text("\n".join(lines) + "\n")

if __name__ == "__main__":
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        main()
