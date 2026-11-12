"""Independent rescoring, step 2: pooled tables + Q2 per-level differences. numpy only."""
import os, re
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
C = np.load(os.path.join(HERE, "counts.npy"), allow_pickle=True).item()
TARGETS = ["0.05", "0.1", "0.15", "0.2", "0.3", "0.4", "0.5", "0.75", "1.0", "1.5", "2.0",
           "3.0", "4.0", "5.0"]
DROP = {("CHB12", 6), ("CHB12", 7), ("CHB12", 8)}

def pooled(arm, tg, drop=frozenset(), skip_cases=()):
    tp = fp = ref = 0; sec = 0.0
    for c, d in C[arm].items():
        if c in skip_cases:
            continue
        for i, (a, b, r, s) in enumerate(d[tg]):
            if (c, i) in drop:
                continue
            tp += a; fp += b; ref += r; sec += s
    return tp, fp, ref, sec

# event bookkeeping
for nm, dr in (("168", frozenset()), ("155", DROP)):
    tp, fp, ref, sec = pooled("A0", "2.0", dr)
    print("set %s: %d events, %.1f h" % (nm, ref, sec / 3600))
print("events in CHB12 blocks 6,7,8:", [C["A0"]["CHB12"]["2.0"][i][2] for i in (6, 7, 8)],
      " CHB12 n blocks", len(C["A0"]["CHB12"]["2.0"]))

def cell(arm, tg, dr, skip=()):
    tp, fp, ref, sec = pooled(arm, tg, dr, skip)
    return tp, ref, fp / (sec / 86400.0), tp / ref

ARMS = ["A0", "Tamp_s0", "Tamp_s1", "Tamp_s2", "Trel_s0", "Trel_s1", "Trel_s2", "Trelf_s0"]
print("\n(1) caught/events and FA/day, per arm")
print("%-9s %-8s %22s %22s" % ("arm", "target", "155 set", "168 set"))
res = {}
for arm in ARMS:
    for tg in ("2.0", "0.05"):
        row = []
        for nm, dr in (("155", DROP), ("168", frozenset())):
            tp, ref, fad, sens = cell(arm, tg, dr)
            res[(arm, tg, nm)] = (tp, ref, fad, sens)
            row.append("%3d/%3d %.3f %6.1f/d" % (tp, ref, sens, fad))
        print("%-9s %-8s %22s %22s" % (arm, tg, row[0], row[1]))

print("\nseed means (sens, FA/day)")
for g in ("Tamp", "Trel"):
    for tg in ("2.0", "0.05"):
        for nm in ("155", "168"):
            v = [res[("%s_s%d" % (g, s), tg, nm)] for s in range(3)]
            print("  %-5s target %-5s set %s: sens %.3f  FA/day %.1f" % (
                g, tg, nm, np.mean([x[3] for x in v]), np.mean([x[2] for x in v])))

# ---- compare with score_interim.log
LOG = os.path.join(HERE, "..", "..", "score_interim.log")
txt = open(LOG, encoding="utf-8").read().splitlines()
print("\nCOMPARE README operating point lines")
mism = 0
for ln in txt:
    m = re.match(r"\s+(\S+)\s+(\d+)/155\s+([\d.]+)/day\s+\|\s+(\d+)/168\s+([\d.]+)/day", ln)
    if m and m.group(1) in ARMS:
        a = m.group(1)
        mine155 = res[(a, "2.0", "155")]; mine168 = res[(a, "2.0", "168")]
        ok = (int(m.group(2)) == mine155[0] and abs(float(m.group(3)) - round(mine155[2], 1)) < 1e-9
              and int(m.group(4)) == mine168[0] and abs(float(m.group(5)) - round(mine168[2], 1)) < 1e-9)
        mism += not ok
        print("  %-9s log %s/155 %s | %s/168 %s   mine %d %.1f | %d %.1f  %s" % (
            a, m.group(2), m.group(3), m.group(4), m.group(5), mine155[0], mine155[2],
            mine168[0], mine168[2], "OK" if ok else "MISMATCH"))
print("  mismatches:", mism)

# deployable curve table, seed-mean cells, all 14 targets, compare with the log
def curve(arm, dr, skip=()):
    pts = [cell(arm, tg, dr, skip) for tg in TARGETS]
    return np.array([p[2] for p in pts]), np.array([p[3] for p in pts])

groups = {"A0": ["A0"], "Tamp": ["Tamp_s%d" % s for s in range(3)],
          "Trel": ["Trel_s%d" % s for s in range(3)], "Trelf": ["Trelf_s0"]}
print("\nCOMPARE deployable curve tables (cols A0, Tamp, Trel, Trelf; seed means)")
for nm, dr in (("155", DROP), ("168", frozenset())):
    # locate the table in the log
    start = [i for i, l in enumerate(txt) if l.startswith("DEPLOYABLE CURVES, %s events" % nm)][0]
    hdr = txt[start + 1].split()[1:]
    bad = 0
    for k, tg in enumerate(TARGETS):
        ln = txt[start + 2 + k]
        assert ln.split()[0] == tg
        vals = re.findall(r"([\d.]+) \(\s*([\d.]+)\)", ln)
        logd = dict(zip(hdr, vals))
        for g, arms in groups.items():
            s = np.mean([curve(a, dr)[1][k] for a in arms])
            f = np.mean([curve(a, dr)[0][k] for a in arms])
            ls, lf = float(logd[g][0]), float(logd[g][1])
            if abs(round(s, 3) - ls) > 1e-9 or abs(round(f, 1) - lf) > 1e-9:
                bad += 1
                print("   MISMATCH set %s target %s %s: log %.3f (%.1f) mine %.4f (%.2f)" % (
                    nm, tg, g, ls, lf, s, f))
    print("  set %s: %d cells compared, %d mismatches" % (nm, 14 * len(groups), bad))

# ---- Q2 per level
G = [50, 60, 75, 90, 100, 115, 125, 150]
def interp_curve(arm, dr, skip=()):
    x, y = curve(arm, dr, skip)
    o = np.argsort(x, kind="stable")
    return x[o], y[o]

def q2(dr, skip=(), label=""):
    cur = {a: interp_curve(a, dr, skip) for a in groups["Trel"] + groups["Tamp"]}
    # ties in x?
    for a, (x, y) in cur.items():
        if len(np.unique(x)) < len(x):
            print("   note: tied FA/day values in", a, label)
    lo = {g: max(cur[a][0].min() for a in groups[g]) for g in ("Trel", "Tamp")}
    hi = {g: min(cur[a][0].max() for a in groups[g]) for g in ("Trel", "Tamp")}
    levels = [L for L in G if all(lo[g] <= L <= hi[g] for g in ("Trel", "Tamp"))]
    print("  Q2 %s: Trel range [%.1f, %.1f], Tamp range [%.1f, %.1f] -> levels %s" % (
        label, lo["Trel"], hi["Trel"], lo["Tamp"], hi["Tamp"], levels))
    diffs = []
    for L in levels:
        r = np.mean([np.interp(L, *cur[a]) for a in groups["Trel"]])
        t = np.mean([np.interp(L, *cur[a]) for a in groups["Tamp"]])
        diffs.append(100 * (r - t))
        print("     level %3d: Trel %.4f  Tamp %.4f  diff %+.2f pts   (Trel seeds %s, Tamp seeds %s)" % (
            L, r, t, 100 * (r - t),
            " ".join("%.4f" % np.interp(L, *cur[a]) for a in groups["Trel"]),
            " ".join("%.4f" % np.interp(L, *cur[a]) for a in groups["Tamp"])))
    print("     mean over levels: %+.2f pts" % np.mean(diffs))
    for s in range(3):
        a = "Trel_s%d" % s
        v = [100 * (np.interp(L, *cur[a]) - np.mean([np.interp(L, *cur[b]) for b in groups["Tamp"]]))
             for L in levels]
        print("     per seed %s - mean Tamp: %+.2f" % (a, np.mean(v)))
    return levels, diffs

print("\n(2) Q2 Trel - Tamp")
q2(DROP, (), "155")
q2(frozenset(), (), "168")
q2(DROP, ("CHB06", "CHB15"), "155 without CHB06,CHB15")
q2(frozenset(), ("CHB06", "CHB15"), "168 without CHB06,CHB15")
