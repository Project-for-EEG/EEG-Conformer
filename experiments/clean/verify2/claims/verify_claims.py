"""Independent checks of draft claims C1-C8 against clean_rule.txt v2. Read-only.
Run: cd C:\EEG && python -B <this file>. Writes only to NEW\verify2\claims."""
import json, sys
from pathlib import Path
import numpy as np
NEW = Path(r"C:\Users\user\AppData\Local\Temp\claude\36c1f29744f93fe6"
           r"\c7848ca6-bfef-414b-a321-d6686c1a43f8\scratchpad\clean")
OUT = NEW / "verify2" / "claims"
sys.path.insert(0, str(NEW))
sys.dont_write_bytecode = True
import clean_score as cs  # noqa: E402

LOG = []
def say(s=""):
    print(s, flush=True); LOG.append(str(s))

def main():
    P, T, src = cs.load_arms()
    cases = sorted(P["A0"])
    names = cs.block_names(cases, P)
    keep168 = {c: [True] * len(names[c]) for c in cases}
    keep155 = {c: [int((b["flat"].mean(0) > cs.ART_FRAC).sum()) < cs.ART_MIN_CH for b in names[c]]
               for c in cases}
    arms = ["A0", "N1", "Tamp_s0", "Tamp_s1", "Tamp_s2", "Trel_s0", "Trel_s1", "Trel_s2",
            "Tfree_s0", "Tfree_s1", "Tfree_s2", "Tall", "Trelf_s0"]
    D = {"155": {a: cs.deploy_table(P[a], T[a], cases, keep155) for a in arms},
         "168": {a: cs.deploy_table(P[a], T[a], cases, keep168) for a in arms}}
    j2 = cs.TARGETS.index("2.0")
    res = {}
    say("C1 README point (target 2 FA/h)")
    for k in ("155", "168"):
        for a in ("N1", "A0"):
            d = D[k][a]
            say("  [%s] %s caught %d/%d, FA %d over %.1f h = %.2f/day" % (
                k, a, d["tp"][:, j2].sum(), d["ref"].sum(), d["fp"][:, j2].sum(),
                d["sec"].sum() / 3600, d["fp"][:, j2].sum() / (d["sec"].sum() / 86400)))
    # A0 and N1 interpolated at each other's README FA level
    for k in ("155", "168"):
        w1 = np.ones(len(cases))
        fa_n, s_n = cs.curve(D[k]["N1"], w1)
        fa_a, s_a = cs.curve(D[k]["A0"], w1)
        say("  [%s] at N1's README FA %.1f/day: A0 %.4f vs N1 %.4f | at A0's %.1f/day: A0 %.4f vs N1 %.4f"
            % (k, fa_n[j2], cs.interp(fa_a, s_a, [fa_n[j2]], False)[0], s_n[j2],
               fa_a[j2], s_a[j2], cs.interp(fa_n, s_n, [fa_a[j2]], False)[0]))

    B = {k: cs.Bench(D[k], cases, cases, k) for k in ("155", "168")}
    sub = [c for c in cases if c not in cs.MOTIVATED]
    Bsub = {k: cs.Bench(D[k], cases, sub, k + " w/o CHB06,CHB15") for k in ("155", "168")}

    def per_seed_level(bench, gx, gy):
        levels = [L for L in bench.reach(gx) if L in bench.reach(gy)]
        vy = cs.group_at(bench.D, cs.GROUPS[gy], bench.w1, levels, False)
        out = {}
        for a in cs.GROUPS[gx]:
            vx = cs.group_at(bench.D, [a], bench.w1, levels, False)
            out[a] = dict(zip(levels, (100 * (vx - vy)).round(2).tolist()))
        return levels, out

    for title, gx, gy, benches in (("C2/Q1 Tamp - N1", "Tamp", "N1", [B]),
                                   ("C4/Q4 Trel - N1", "Trel", "N1", [B, Bsub]),
                                   ("Q2 Trel - Tamp", "Trel", "Tamp", [B, Bsub])):
        say(); say(title)
        for bb in benches:
            for k in ("155", "168"):
                r = bb[k].compare(gx, gy)
                say("  [%s] point %+.2f  95%% [%+.2f, %+.2f]  clamped [%+.2f, %+.2f]  partial %d dropped %d  -> %s"
                    % (r["set"], 100 * r["point"], 100 * r["lo"], 100 * r["hi"], 100 * r["clo"],
                       100 * r["chi"], r["partial"], r["dropped"], r["decision"]))
                say("      per level: " + ", ".join("%d: %+.2f" % (L, 100 * v) for L, v in r["per_level"].items()))
                levels, ps = per_seed_level(bb[k], gx, gy)
                for a, v in ps.items():
                    neg = [L for L, x in v.items() if x <= 0]
                    say("      %s per level: %s | mean %+.2f | levels <= 0: %s" % (
                        a, ", ".join("%d: %+.2f" % (L, x) for L, x in v.items()),
                        np.mean(list(v.values())), neg))
                res.setdefault(title, {})[r["set"]] = {"point": r["point"], "lo": r["lo"], "hi": r["hi"],
                                                       "per_seed_level": ps}

    say(); say("C5 sensitivity interpolated at 25 FA/day (and 20, 30), each seed, then mean (NaN if outside)")
    for k in ("155", "168"):
        w1 = np.ones(len(cases))
        for g in ("Trel", "N1", "A0", "Tamp", "Trelf"):
            vals = {}
            for L in (20, 25, 30):
                vals[L] = [float(cs.interp(*cs.curve(D[k][a], w1), [L], False)[0]) for a in cs.GROUPS[g]]
            fa_min = min(float(cs.curve(D[k][a], w1)[0].min()) for a in cs.GROUPS[g])
            say("  [%s] %-5s min FA/day on curve %.2f | " % (k, g, fa_min) + " | ".join(
                "@%d: %s mean %.4f" % (L, ["%.4f" % x for x in v], np.mean(v)) for L, v in vals.items()))
    (OUT / "verify_claims.txt").write_text("\n".join(LOG) + "\n", encoding="ascii", errors="replace")
    (OUT / "verify_claims.json").write_text(json.dumps(res, indent=1, default=str))

if __name__ == "__main__":
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        main()
