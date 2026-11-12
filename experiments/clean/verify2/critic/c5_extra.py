"""critic: claims C5/C8 numbers, Q4 strength checks (exploratory)."""
import sys
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT))
from c_lib import TAB, GROUPS, CASES, weights, gat, compare, curve, interp, NB  # noqa: E402

lines = []


def say(s=""):
    print(s, flush=True)
    lines.append(s)


w1 = weights(CASES)[1].sum(0)
say("C5: matched sensitivity at 25 FA/day (full data, per seed and mean)")
for k in ("155", "168"):
    for g in ("Trel", "N1", "Tamp", "A0"):
        v = [interp(*curve(TAB[k][a], w1), [25.0], False)[0] for a in GROUPS[g]]
        say("  [%s] %-5s %s mean %.4f" % (k, g, " ".join("%.4f" % x for x in v), np.mean(v)))
    for g in ("N1", "Tamp", "A0", "Trel"):
        mins = [curve(TAB[k][a], w1)[0].min() for a in GROUPS[g]]
        say("  [%s] %-5s lowest FA/day on the curve: %s" % (k, g, " ".join("%.2f" % m for m in mins)))

say()
say("C8: deployable matched sensitivity at 100 FA/day")
for k in ("155", "168"):
    for g in ("N1", "Tamp", "Trel", "A0"):
        v = gat(k, GROUPS[g], w1, [100.0])[0]
        say("  [%s] %-5s %.4f" % (k, g, v))

say()
say("Q4 strength: per-level NaN rate in the bootstrap, full-only draws, 3-question percentile")
for k in ("155", "168"):
    r = compare(k, "Trel", "N1")
    bn = r["bn"]
    say("  [%s] point %+.4f, 95%% [%+.4f, %+.4f]; 98.33%% (0.833..99.167 pct) [%+.4f, %+.4f]; 99%% [%+.4f, %+.4f]"
        % (k, 100 * r["point"], 100 * r["lo"], 100 * r["hi"],
           100 * np.nanpercentile(bn, 0.8333), 100 * np.nanpercentile(bn, 99.1667),
           100 * np.nanpercentile(bn, 0.5), 100 * np.nanpercentile(bn, 99.5)))
    # rebuild per-level NaN pattern with the same rng-7 draws
    subj, S = weights(CASES)
    rng = np.random.default_rng(7)
    idx = rng.integers(0, len(subj), size=(NB, len(subj)))
    SI = rng.integers(0, 3, size=(NB, 3))
    lv = r["levels"]
    nan_ct = np.zeros(len(lv), int)
    full = []
    for q in range(NB):
        w = np.bincount(idx[q], minlength=len(subj)).astype(float) @ S
        sx = [GROUPS["Trel"][i] for i in SI[q]]
        dd = gat(k, sx, w, lv) - gat(k, ["N1"], w, lv)
        nan_ct += np.isnan(dd)
        if not np.isnan(dd).any():
            full.append(dd.mean())
    full = np.array(full)
    say("  [%s] NaN count per level %s: %s; full draws %d, their 95%% [%+.4f, %+.4f]"
        % (k, lv, nan_ct.tolist(), len(full), 100 * np.percentile(full, 2.5),
           100 * np.percentile(full, 97.5)))

say()
say("Q1 per-level NaN pattern (which level the partial draws lose)")
for k in ("155",):
    r = compare(k, "Tamp", "N1")
    subj, S = weights(CASES)
    rng = np.random.default_rng(7)
    idx = rng.integers(0, len(subj), size=(NB, len(subj)))
    SI = rng.integers(0, 3, size=(NB, 3))
    lv = r["levels"]
    nan_ct = np.zeros(len(lv), int)
    full = []
    for q in range(NB):
        w = np.bincount(idx[q], minlength=len(subj)).astype(float) @ S
        sx = [GROUPS["Tamp"][i] for i in SI[q]]
        dd = gat(k, sx, w, lv) - gat(k, ["N1"], w, lv)
        nan_ct += np.isnan(dd)
        if not np.isnan(dd).any():
            full.append(dd.mean())
    full = np.array(full)
    say("  [%s] NaN count per level %s: %s; full draws %d, their 95%% [%+.4f, %+.4f]"
        % (k, lv, nan_ct.tolist(), len(full), 100 * np.percentile(full, 2.5),
           100 * np.percentile(full, 97.5)))

(OUT / "c5_extra.txt").write_text("\n".join(lines) + "\n")
