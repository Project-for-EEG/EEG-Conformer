"""Extra: subset analyses with the MAIN comparison's level set instead of a level set re-derived
on the subset (reading of 'fixed from the full data and never re-chosen'). Reuses the functions of
rule_stats.py (executed up to its question loop)."""
from pathlib import Path
p = Path(__file__).with_name("rule_stats.py")
src = p.read_text()
exec(src[:src.index("results = {}")])
NOMOT = [s for s in SUBJ if s not in ("CHB06", "CHB15")]
out = []
for name, gx, gy, sub, main_lv in (("Q2noMOT", "Trel", "Tamp", NOMOT, [60, 75, 90, 100]),
                                    ("Q4noMOT", "Trel", "N1", NOMOT, [50, 60, 75, 90, 100]),
                                    ("Q3a", "N1", "A0", EARLY, [50, 60, 75, 90, 100, 115, 125]),
                                    ("Q3b", "N1", "A0", SAME15, [50, 60, 75, 90, 100, 115, 125])):
    for k in ("155", "168"):
        w1 = S_matrix([s for s in SUBJ if s in sub]).sum(0)
        rx, ry = reach(gx, DS[k], w1), reach(gy, DS[k], w1)
        lv = [L for L in main_lv if L in rx and L in ry]
        r = compare(gx, gy, k, sub, levels=lv)
        out.append("%s [%s] main-comparison levels %s, reachable on subset %s: %s"
                   % (name, k, main_lv, lv, fmt(r)))
print("\n".join(out))
(p.with_name("rule_extra.txt")).write_text("\n".join(out) + "\n")
