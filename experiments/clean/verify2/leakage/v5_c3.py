"""leakage audit, part 5: C3 independently. CPU only, read-only. cwd C:\\EEG.
  a) per-batch training losses printed by tqdm in lopo_run.log (A0) vs train_fixed15.log (N1).
     Bit-identical training => identical loss sequences for fold 1 (all 15 epochs) and fold 2
     (epochs 1-14, A0 stopped at 14); from fold 3 on the torch stream differs.
  b) CHB01_CHB21 logits: n1_cache vs A0 pilot_cache2, my own read of both caches.
  c) raw file comparison of the two checkpoints.
"""
import hashlib
import re
from pathlib import Path

import numpy as np

NEW = Path(r"C:\Users\user\AppData\Local\Temp\claude\36c1f29744f93fe6"
           r"\c7848ca6-bfef-414b-a321-d6686c1a43f8\scratchpad\clean")
OLD2 = Path(r"C:\Users\user\AppData\Local\Temp\claude\36c1f29744f93fe6"
            r"\f241281f-add4-41ac-9210-f47dab0b83b0\scratchpad")
A0LOG = Path("C:/EEG/lopo_run.log")
CKDIR = Path("C:/EEG/checkpoints/lopo")
PAT = re.compile(r"Fold (\d+) Epoch (\d+)/15:\s*\d+%\|[^|]*\|\s*(\d+)/\d+ \[[^\]]*loss=([-\d.na]+)\]")
SPLIT = re.compile("[\r\n]+")


def losses(path):
    """(fold, epoch) -> sequence of displayed loss strings, consecutive repeats collapsed.
    set_postfix redraws once per batch, so every batch's loss appears; the displayed n lags
    irregularly (tqdm miniters), so n is NOT used as a key."""
    d = {}
    for l in SPLIT.split(path.read_text(encoding="utf-8", errors="replace")):
        m = PAT.search(l)
        if m:
            f, e, n, loss = m.groups()
            seq = d.setdefault((int(f), int(e)), [])
            if not seq or seq[-1] != loss:
                seq.append(loss)
    return d


a = losses(A0LOG)
n = losses(NEW / "train_fixed15.log")
print("a) per (fold, epoch): collapsed per-batch loss sequence A0 vs N1: EQ(len) or diff@first index")
summary = {}
for f in range(1, 24):
    row = []
    for e in range(1, 16):
        if (f, e) in a and (f, e) in n:
            sa, sn = a[(f, e)], n[(f, e)]
            if sa == sn:
                row.append("%d:EQ(%d)" % (e, len(sa)))
                summary.setdefault(f, []).append("EQ")
            else:
                k = next((i for i, (x, y) in enumerate(zip(sa, sn)) if x != y), min(len(sa), len(sn)))
                row.append("%d:diff@%d(%d/%d)" % (e, k, len(sa), len(sn)))
                summary.setdefault(f, []).append("D")
        elif (f, e) in n:
            row.append("%d:N1only(%d)" % (e, len(n[(f, e)])))
    if f <= 4 or f in (8, 12, 23):
        print("  fold %2d  %s" % (f, "  ".join(row)))
print("  epochs equal per fold:", {f: "%d/%d" % (v.count("EQ"), len(v)) for f, v in summary.items()})

# b) logits
fold = "CHB01_CHB21"
mx, nw, nf, same_obj = 0.0, 0, 0, 0
for f in sorted((NEW / "n1_cache").glob("*.npz")):
    g = OLD2 / "pilot_cache2" / f.name
    assert f.resolve() != g.resolve()
    with np.load(f) as dn, np.load(g) as da:
        x, y = dn["logit_" + fold], da["logit_" + fold]
        if f.name.startswith("CHB03__"):
            print("   %s: N1 vs A0 logit_CHB03 max|d| %.4g" % (
                f.name, float(np.abs(dn["logit_CHB03"].astype(np.float64) - da["logit_CHB03"]).max())))
    assert x.shape == y.shape
    mx = max(mx, float(np.abs(x.astype(np.float64) - y).max()))
    same_obj += int(np.array_equal(x.view(np.uint32), y.view(np.uint32)))
    nw += len(x)
    nf += 1
print("b) %s: %d files, %d windows, max|N1-A0| %.3g, files bitwise equal %d/%d" % (
    fold, nf, nw, mx, same_obj, nf))

# c) checkpoint files
for name in ("lopo_CHB01_CHB21.pt", "fixed15_CHB01_CHB21.pt"):
    p = CKDIR / name
    print("c) %-24s %7d bytes sha256 %s" % (name, p.stat().st_size,
                                             hashlib.sha256(p.read_bytes()).hexdigest()[:24]))
