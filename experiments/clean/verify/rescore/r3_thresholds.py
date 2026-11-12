"""Internal consistency of the stored thresholds. json + numpy only."""
import os, json
import numpy as np
T = r"C:\Users\user\AppData\Local\Temp\claude\36c1f29744f93fe6"
OLD = os.path.join(T, "b3889703-629c-4c5a-b583-69f722743ff4", "scratchpad", "trees", "out")
NEW = os.path.join(T, "c7848ca6-bfef-414b-a321-d6686c1a43f8", "scratchpad", "clean", "trees_out")
TARGETS = ["0.05", "0.1", "0.15", "0.2", "0.3", "0.4", "0.5", "0.75", "1.0", "1.5", "2.0",
           "3.0", "4.0", "5.0"]
LIN = np.linspace(0.001, 0.999, 400)
SRC = {"Trel_s0": (NEW, "rel0"), "Trel_s1": (NEW, "rel1"), "Trel_s2": (NEW, "rel2"),
       "Trelf_s0": (NEW, "relf0"), "Tamp_s0": (OLD, "a"), "Tamp_s1": (OLD, "b"),
       "Tamp_s2": (OLD, "c"), "A0": (OLD, "a")}
folds = sorted(os.path.basename(p)[5:-3] for p in os.listdir(r"C:\EEG\checkpoints\lopo")
               if p.startswith("lopo_") and p.endswith(".pt"))
folds = [f[5:-3] if f.startswith("lopo_") else f for f in folds]
print("folds (%d):" % len(folds), folds[:3], "...")
probs = 0
for arm, (d, tag) in SRC.items():
    TH = json.load(open(os.path.join(d, "thresholds_%s.json" % tag)))[arm]
    LOGS = [r for r in json.load(open(os.path.join(d, "train_log_%s.json" % tag))) if r["arm"] == arm]
    issues = []
    # 1 every case has all 14 targets, values in (0, 1]
    assert len(TH) == 24
    for c, t in TH.items():
        if sorted(t, key=float) != sorted(TARGETS, key=float):
            issues.append("targets %s" % c)
        v = np.array([t[k] for k in TARGETS])
        if not np.all((v > 0) & (v <= 1)):
            issues.append("range %s" % c)
        # 2 monotone: larger target -> threshold <= (non-increasing)
        if np.any(np.diff(v) > 0):
            issues.append("monotone %s %s" % (c, v))
    # 3 one threshold set per fold: CHB01 == CHB21
    if TH["CHB01"] != TH["CHB21"]:
        issues.append("CHB01 != CHB21")
    # 4 train log: 23 folds, each fold's t_2fa equals held-out cases' 2.0 threshold
    if len(LOGS) != 23:
        issues.append("train log has %d folds" % len(LOGS))
    for r in LOGS:
        for c in r["fold"].split("_"):
            if abs(TH[c]["2.0"] - r["t_2fa"]) > 0:
                issues.append("log t_2fa mismatch %s" % c)
    held = sorted(c for r in LOGS for c in r["fold"].split("_"))
    if held != sorted(TH):
        issues.append("held-out cases in log != threshold cases")
    # descriptive
    allv = np.array([[TH[c][k] for k in TARGETS] for c in sorted(TH)])
    ones = int((allv == 1.0).sum())
    onlin = int(np.isclose(allv[..., None], LIN[None, None, :], rtol=0, atol=1e-12).any(-1).sum())
    distinct_sets = len({tuple(TH[c][k] for k in TARGETS) for c in TH})
    ties = int((np.diff(allv, axis=1) == 0).sum())
    print("%-9s issues %d %s | distinct threshold sets %d (expect 23) | thr==1.0: %d | on linspace grid: %d/%d | "
          "equal adjacent targets: %d | t@2.0 range [%.4f, %.4f] | t@0.05 range [%.4f, %.4f]"
          % (arm, len(issues), issues[:4], distinct_sets, ones, onlin, allv.size, ties,
             allv[:, 10].min(), allv[:, 10].max(), allv[:, 0].min(), allv[:, 0].max()))
    probs += len(issues)
# seeds differ from each other and from Tamp (thresholds not copied)
TR = {a: json.load(open(os.path.join(SRC[a][0], "thresholds_%s.json" % SRC[a][1])))[a] for a in SRC}
for a, b in (("Trel_s0", "Trel_s1"), ("Trel_s0", "Trel_s2"), ("Trel_s1", "Trel_s2"),
             ("Trel_s0", "Tamp_s0"), ("Trel_s0", "Trelf_s0")):
    same = sum(TR[a][c] == TR[b][c] for c in TR[a])
    print("  %s vs %s: identical threshold sets in %d of 24 cases" % (a, b, same))
# fold key in thresholds for every fold: compare to train log fold names
print("total issues:", probs)
