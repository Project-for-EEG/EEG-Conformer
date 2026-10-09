"""Leave-one-annotator-out human ceiling on Helsinki, SzCORE event rules, 1 s resolution."""
import csv, sys
import numpy as np

BASE = sys.argv[1]
FILES = ["annotations_2017_A_fixed.csv", "annotations_2017_B.csv", "annotations_2017_C.csv"]
MERGE, PRE, POST, SPLIT = 90, 30, 60, 300  # seconds (project's own constants)

def load(p):
    rows = list(csv.reader(open(p)))
    data = rows[1:]
    n_b = len(rows[0])
    cols = []
    for j in range(n_b):
        v = []
        for r in data:
            c = r[j] if j < len(r) else ""
            if c == "":
                break
            v.append(int(c))
        cols.append(np.array(v, dtype=np.int8))
    return cols

A, B, C = (load(f"{BASE}/{f}") for f in FILES)

def runs(mask):
    m = np.concatenate(([0], mask.astype(np.int8), [0]))
    d = np.diff(m)
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))  # [lo, hi)

def merge(ev, gap):
    out = []
    for lo, hi in ev:
        if out and lo - out[-1][1] < gap:
            out[-1] = (out[-1][0], hi)
        else:
            out.append((lo, hi))
    return out

def split(ev, n):
    out = []
    for lo, hi in ev:
        while hi - lo > n:
            out.append((lo, lo + n)); lo += n
        out.append((lo, hi))
    return out

def score(det_mask, ref_mask):
    det = split(merge(runs(det_mask), MERGE), SPLIT)
    ref = split(runs(ref_mask), SPLIT)
    ext = [(max(0, lo - PRE), hi + POST) for lo, hi in ref]
    hit = sum(1 for lo, hi in ext if any(d0 < hi and d1 > lo for d0, d1 in det))
    fa = sum(1 for d0, d1 in det if not any(d0 < hi and d1 > lo for lo, hi in ext))
    return hit, len(ref), fa

for held, (det_cols, o1, o2) in zip("ABC", [(A, B, C), (B, A, C), (C, A, B)]):
    for mode in ("both", "either"):
        H = R = F = 0
        secs = 0
        for i in range(len(det_cols)):
            n = min(len(det_cols[i]), len(o1[i]), len(o2[i]))
            d = det_cols[i][:n]
            ref = (o1[i][:n] & o2[i][:n]) if mode == "both" else (o1[i][:n] | o2[i][:n])
            h, r, f = score(d, ref)
            H += h; R += r; F += f; secs += n
        days = secs / 86400.0
        sens = H / R if R else float("nan")
        print(f"held-out {held}  ref={mode:6s}  n_ref_events={R:4d}  sens={sens:.3f}  FA={F:4d}  FA/day={F/days:6.2f}  hours={secs/3600:.1f}")

# model-comparable variant: reference = majority of all three (project default label)
print()
maj_sens = []
for held, (det_cols, o1, o2) in zip("ABC", [(A, B, C), (B, A, C), (C, A, B)]):
    H = R = F = 0; secs = 0
    for i in range(len(det_cols)):
        n = min(len(det_cols[i]), len(o1[i]), len(o2[i]))
        maj = ((det_cols[i][:n].astype(int) + o1[i][:n] + o2[i][:n]) >= 2).astype(np.int8)
        h, r, f = score(det_cols[i][:n], maj)
        H += h; R += r; F += f; secs += n
    days = secs / 86400.0
    print(f"held-out {held}  ref=majority-of-3 (INFLATED, annotator is in ref)  sens={H/R:.3f}  FA/day={F/days:6.2f}  n_ref={R}")

# pairwise annotator agreement, event level
print()
for (x, nx), (y, ny) in [((A,'A'),(B,'B')), ((A,'A'),(C,'C')), ((B,'B'),(C,'C'))]:
    H=R=F=0; secs=0
    for i in range(len(x)):
        n=min(len(x[i]),len(y[i]))
        h,r,f = score(x[i][:n], y[i][:n]); H+=h; R+=r; F+=f; secs+=n
    print(f"{nx} as detector vs {ny} as reference: sens={H/R:.3f} FA/day={F/(secs/86400.0):6.2f} n_ref={R}")

# patient-level disagreement
pat = 0
for i in range(len(A)):
    n = min(len(A[i]), len(B[i]), len(C[i]))
    flags = [A[i][:n].any(), B[i][:n].any(), C[i][:n].any()]
    if 0 < sum(flags) < 3:
        pat += 1
print(f"\npatients where annotators disagree on seizure-positive at all: {pat} of {len(A)}")
