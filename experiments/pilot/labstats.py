import os, sys, numpy as np
os.chdir(r"C:\EEG"); sys.path.insert(0, r"C:\EEG")
from pathlib import Path
from evaluate_temporal import find_runs
root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("preprocessed_data")
cap = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
def find_split_labels(labs, n=1):
    seen = 0
    for fi, lab in enumerate(labs):
        for lo, hi in find_runs(lab.astype(bool)):
            seen += 1
            if seen == n:
                rem = int(lab[hi:].sum()) + sum(int(l.sum()) for l in labs[fi+1:])
                return (fi, hi) if rem > 0 else None
    return None
tot = {"win":0,"sz":0}
rows = []
for p in sorted(os.listdir(root)):
    d = root / p
    if not d.is_dir(): continue
    labs = []
    for f in sorted(d.glob("*.npz")):
        with np.load(f) as z:
            labs.append(z["labels"].copy())
    n = sum(len(l) for l in labs); s = sum(int(l.sum()) for l in labs)
    ev = sum(len(find_runs(l.astype(bool))) for l in labs)
    sp = find_split_labels(labs)
    if sp:
        fi, wi = sp
        pre = sum(len(l) for l in labs[:fi]) + wi
        pre_bg = sum(int((l==0).sum()) for l in labs[:fi]) + int((labs[fi][:wi]==0).sum())
        test_w = n - pre
        test_ev = sum(len(find_runs(l.astype(bool))) for l in labs[fi+1:]) + len(find_runs(labs[fi][wi:].astype(bool)))
        # first-file windows before wi that are seizure (context peek)
        tail = labs[fi][max(0,wi-16):wi]
        peek_sz = int(tail.sum())
        nfile_pre = fi+1
    else:
        pre=pre_bg=test_w=test_ev=peek_sz=nfile_pre=0
    bg = n - s
    seen_bg = max(0, cap - s)
    rows.append((p, len(labs), n, s, ev, sp is not None, pre, pre_bg, int(0.3*pre_bg), test_w, test_ev, round(min(1,seen_bg/bg),3) if bg else 0, peek_sz, nfile_pre))
    tot["win"] += n; tot["sz"] += s
print("%-6s %4s %7s %6s %4s %5s %7s %7s %6s %7s %5s %6s %4s %4s" % ("pt","nf","win","szwin","ev","split","pre","pre_bg","resv","test_w","t_ev","bgseen","peek","fpre"))
for r in rows: print("%-6s %4d %7d %6d %4d %5s %7d %7d %6d %7d %5d %6.3f %4d %4d" % r)
print(tot, "sum test ev", sum(r[10] for r in rows), "sum test windows", sum(r[9] for r in rows))
