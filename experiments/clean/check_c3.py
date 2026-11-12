"""Check C3 (clean_rule.txt v2 section 1). Read-only.

Fold CHB01+CHB21 runs first in both A0 and N1 from the same random state, and A0
ran it for the full 15 rounds, so the two models should agree up to GPU float
noise. Compares N1's logit_CHB01_CHB21 (NEW\\n1_cache) with A0's (OLD\\pilot_cache2)
over every window of every recording, and separately over the held-out cases.
For contrast it does the same for CHB03, a fold where the random state had
already diverged (fold 2, CHB02, stopped at round 14 in A0).
"""
from pathlib import Path

import numpy as np

OLD = Path(r"C:\Users\user\AppData\Local\Temp\claude\36c1f29744f93fe6"
           r"\f241281f-add4-41ac-9210-f47dab0b83b0\scratchpad")   # same as trees_train.py
NEW = Path(r"C:\Users\user\AppData\Local\Temp\claude\36c1f29744f93fe6"
           r"\c7848ca6-bfef-414b-a321-d6686c1a43f8\scratchpad\clean")
A0 = OLD / "pilot_cache2"
N1 = NEW / "n1_cache"


def compare(fold):
    held = fold.split("_")
    a_all, n_all, a_h, n_h, nfiles = [], [], [], [], 0
    for f in sorted(N1.glob("*.npz")):
        with np.load(f) as dn, np.load(A0 / f.name) as da:
            assert np.array_equal(dn["labels"], da["labels"]), f.name
            a, n = da["logit_" + fold].astype(np.float64), dn["logit_" + fold].astype(np.float64)
        assert a.shape == n.shape, f.name
        a_all.append(a)
        n_all.append(n)
        if f.name.split("__")[0] in held:
            a_h.append(a)
            n_h.append(n)
        nfiles += 1
    out = {"fold": fold, "files": nfiles}
    for tag, aa, nn in (("all", a_all, n_all), ("held-out", a_h, n_h)):
        aa, nn = np.concatenate(aa), np.concatenate(nn)
        d = np.abs(aa - nn)
        out[tag] = {"windows": len(aa), "max_abs_diff": float(d.max()),
                    "mean_abs_diff": float(d.mean()),
                    "corr": float(np.corrcoef(aa, nn)[0, 1]),
                    "logit_sd": float(aa.std()),
                    "same_sign_frac": float(np.mean(np.sign(aa) == np.sign(nn)))}
    return out


if __name__ == "__main__":
    import json
    n_a0 = len(list(A0.glob("*.npz")))
    n_n1 = len(list(N1.glob("*.npz")))
    print("files: A0 cache %d, N1 cache %d" % (n_a0, n_n1))
    assert n_a0 == n_n1
    res = [compare("CHB01_CHB21"), compare("CHB03")]
    print(json.dumps(res, indent=2))
    (NEW / "check_c3.json").write_text(json.dumps(res, indent=2))
