"""
Relative size features for the Trel / Trelf arms (clean_rule.txt section 2).

For every OLD\\feats\\<CASE>__<stem>.npz (feats (n, 23, 19), labels):
  lg   = log of the 8 size features (columns 0..7); non-finite -> NaN (as r_refit.py)
  rel  = lg - running median of lg over the previous 300 windows of the same file and
         channel, current window included (pandas rolling(300, min_periods=1), NaN skipped)
  relf = lg - whole-file median of lg (np.nanmedian over the file; r_refit.py's Trel)
Writes NEW\\rel\\<CASE>__<stem>.npz with rel, relf (n, 23, 8) float32 and labels.
Then spot-checks the running median against an explicit np.nanmedian loop.
Reads OLD only. Writes only NEW\\rel.
"""
import sys
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
OLD = Path(r"C:\Users\user\AppData\Local\Temp\claude\36c1f29744f93fe6"
           r"\b3889703-629c-4c5a-b583-69f722743ff4\scratchpad\trees\feats")
NEW = Path(r"C:\Users\user\AppData\Local\Temp\claude\36c1f29744f93fe6"
           r"\c7848ca6-bfef-414b-a321-d6686c1a43f8\scratchpad\clean\rel")
WIN = 300


def log_size(feats):
    with np.errstate(divide="ignore", invalid="ignore"):
        lg = np.log(feats[:, :, :8].astype(np.float64))
    lg[~np.isfinite(lg)] = np.nan
    return lg


def running_median(lg):
    n = len(lg)
    df = pd.DataFrame(lg.reshape(n, -1))
    return df.rolling(WIN, min_periods=1).median().to_numpy().reshape(lg.shape)


def one(path):
    with np.load(path) as d:
        feats, labels = d["feats"], d["labels"]
    lg = log_size(feats)
    rel = lg - running_median(lg)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(lg, axis=0, keepdims=True) if np.isfinite(lg).any() else 0.0
    relf = lg - med
    np.savez(NEW / path.name, rel=rel.astype(np.float32), relf=relf.astype(np.float32),
             labels=labels)
    return path.name, len(labels), int(np.isnan(rel).sum()), int(np.isnan(lg).sum())


def spot_check(names, rng):
    worst = 0.0
    for name in names:
        with np.load(OLD / name) as d:
            lg = log_size(d["feats"])
        with np.load(NEW / name) as d:
            rel = d["rel"].astype(np.float64)
        n = len(lg)
        for i in rng.choice(n, size=min(n, 40), replace=False):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                med = np.nanmedian(lg[max(0, i - WIN + 1):i + 1], axis=0)
            ref = lg[i] - med
            both = np.isnan(ref) == np.isnan(rel[i])
            assert both.all(), (name, i)
            ok = ~np.isnan(ref)
            if ok.any():
                worst = max(worst, float(np.max(np.abs(ref[ok] - rel[i][ok]))))
    return worst


def main():
    NEW.mkdir(parents=True, exist_ok=True)
    paths = sorted(OLD.glob("*.npz"))
    assert len(paths) == 152, len(paths)
    t0 = time.time()
    with Pool(6) as pool:
        res = pool.map(one, paths)
    n = sum(r[1] for r in res)
    print("%d files, %d windows, %.0f s" % (len(res), n, time.time() - t0))
    print("NaN cells: rel %d, log %d (must be equal: the median adds no NaN where log is finite"
          " ... except where every past window is NaN)" % (sum(r[2] for r in res),
                                                         sum(r[3] for r in res)))
    rng = np.random.default_rng(1)
    pick = [paths[i].name for i in rng.choice(len(paths), size=12, replace=False)]
    pick += ["CHB12__chb12_27_labeled.npz", "CHB13__chb13_40_labeled.npz"]
    worst = spot_check([p for p in pick if (NEW / p).exists()], rng)
    print("spot check vs explicit np.nanmedian over %d files: max |diff| %.2e" % (len(pick), worst))
    assert worst < 1e-5


if __name__ == "__main__":
    main()
