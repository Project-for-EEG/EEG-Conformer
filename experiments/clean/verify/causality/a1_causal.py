"""
Audit task 1: is rel[i] a function of windows <= i of the same file only, and are labels unused?

(a) Explicit-loop recomputation (np.nanmedian over lg[max(0,i-299):i+1]) on 4 files, vs the
    stored NEW\\rel arrays.
(b) Perturb-the-future: for every cut i (exhaustive on three 1-h files, 400 cuts on a 3.7-h
    file) replace feats[i+1:] by garbage (random rescale, NaN, zeros, 1e30), recompute rel
    with the PRODUCTION functions from rel_feats.py, and require rel[:i+1] bit-identical.
    Also shuffle labels (labels are not an input, shown by signature + AST scan).
(c) All 152 files: stored rel == production recomputation from OLD feats (bitwise) and
    == explicit sliding nanmedian (tolerance), and NaN pattern of rel == NaN pattern of log.
Writes only to this folder (stdout captured by caller).
"""
import ast
import sys
import time
import warnings

sys.dont_write_bytecode = True
import os  # noqa: E402
import numpy as np  # noqa: E402
from numpy.lib.stride_tricks import sliding_window_view  # noqa: E402

CLEAN = r"C:\Users\user\AppData\Local\Temp\claude\36c1f29744f93fe6\c7848ca6-bfef-414b-a321-d6686c1a43f8\scratchpad\clean"
OLDF = r"C:\Users\user\AppData\Local\Temp\claude\36c1f29744f93fe6\b3889703-629c-4c5a-b583-69f722743ff4\scratchpad\trees\feats"
REL = os.path.join(CLEAN, "rel")
sys.path.insert(0, CLEAN)
import rel_feats  # noqa: E402  (production code; main() is guarded)

WIN = 300


def load(name):
    with np.load(os.path.join(OLDF, name)) as d:
        feats, labels = d["feats"], d["labels"]
    with np.load(os.path.join(REL, name)) as d:
        rel, relf, lab2 = d["rel"], d["relf"], d["labels"]
    assert np.array_equal(labels, lab2)
    return feats, labels, rel, relf


def explicit_rel(feats):
    """Independent: own log + explicit per-window loop of np.nanmedian over the past 300."""
    x = feats[:, :, :8].astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        lg = np.log(x)
    lg[~np.isfinite(lg)] = np.nan
    out = np.empty_like(lg)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for i in range(len(lg)):
            out[i] = lg[i] - np.nanmedian(lg[max(0, i - WIN + 1):i + 1], axis=0)
    return out


def prod_rel(feats):
    lg = rel_feats.log_size(feats)
    return lg - rel_feats.running_median(lg)


def garbage(feats, i, rng):
    f = feats.copy()
    tail = f[i + 1:].astype(np.float64)
    if len(tail):
        tail = tail * np.exp(rng.normal(0, 3, size=tail.shape))
        u = rng.random(tail.shape)
        tail[u < 0.15] = np.nan
        tail[(u >= 0.15) & (u < 0.25)] = 0.0
        tail[(u >= 0.25) & (u < 0.30)] = 1e30
        f[i + 1:] = tail.astype(np.float32)
    return f


def part_a_b():
    rng = np.random.default_rng(20261105)
    files = [("CHB15__chb15_54_labeled.npz", "all"),   # 5 seizures, one at window 131
             ("CHB13__chb13_40_labeled.npz", "all"),   # 6 flat channels, seizures at 71, 265
             ("CHB19__chb19_01_labeled.npz", "all"),   # no seizure, 6 flat channels
             ("CHB06__chb06_04_labeled.npz", 400)]     # 6629 windows, seizures at 163, 3105
    for name, mode in files:
        feats, labels, rel, _ = load(name)
        n = len(labels)
        t0 = time.time()
        ex = explicit_rel(feats)
        nan_same = np.array_equal(np.isnan(ex), np.isnan(rel))
        ok = ~np.isnan(ex)
        dmax = float(np.max(np.abs(ex[ok] - rel.astype(np.float64)[ok])))
        ex32_equal = np.array_equal(ex.astype(np.float32), rel, equal_nan=True)
        print("(a) %-28s n=%5d seizures(windows)=%4d  explicit-loop vs stored: NaN pattern equal %s, "
              "max|diff| %.2e, float32-identical %s  [%.0f s]"
              % (name, n, int(labels.sum()), nan_same, dmax, ex32_equal, time.time() - t0))
        base = prod_rel(feats)
        assert np.array_equal(base.astype(np.float32), rel, equal_nan=True), name
        if mode == "all":
            cuts = np.arange(n)
        else:
            cuts = np.unique(np.concatenate([np.arange(0, 12), np.arange(290, 310),
                                             np.arange(150, 175), np.arange(3095, 3115),
                                             rng.choice(n, mode - 77, replace=False), [n - 1]]))
        t0 = time.time()
        bad, changed_after = 0, 0
        for i in cuts:
            r2 = prod_rel(garbage(feats, int(i), rng))
            if not np.array_equal(r2[:i + 1], base[:i + 1], equal_nan=True):
                bad += 1
            if i < n - 1 and not np.array_equal(r2[i + 1:], base[i + 1:], equal_nan=True):
                changed_after += 1
        # current window is included: perturb only window i, rel[i] must move
        moved = 0
        for i in cuts[:50]:
            f = feats.copy()
            fin = np.isfinite(f[i, :, :8])
            f[i, :, :8] = np.where(fin, f[i, :, :8] * 7.0, f[i, :, :8])
            r2 = prod_rel(f)
            moved += bool(np.any(np.abs(r2[i][fin] - base[i][fin]) > 1e-9)) if fin.any() else True
        print("(b) %-28s cuts %5d: rel[:i+1] changed after garbling windows > i in %d cuts; "
              "garbling did change rel[i+1:] in %d of %d cuts with a future; "
              "perturbing window i alone moves rel[i] in %d/50  [%.0f s]"
              % (name, len(cuts), bad, changed_after, int(np.sum(cuts < n - 1)), moved,
                 time.time() - t0))
        lab_shuf = rng.permutation(labels)
        # labels are not an argument of log_size/running_median; show the result does not move
        assert np.array_equal(prod_rel(feats), base, equal_nan=True)
        del lab_shuf


def ast_label_scan():
    src = open(os.path.join(CLEAN, "rel_feats.py"), encoding="utf-8").read()
    tree = ast.parse(src)
    uses = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == "labels":
            uses.append((node.lineno, type(node.ctx).__name__))
        if isinstance(node, ast.Constant) and node.value == "labels":
            uses.append((node.lineno, "str"))
    lines = src.splitlines()
    print("(b) AST: every occurrence of 'labels' in rel_feats.py:")
    for ln, kind in sorted(set(uses)):
        print("      line %3d [%s]: %s" % (ln, kind, lines[ln - 1].strip()))
    fn_args = {n.name: [a.arg for a in n.args.args] for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef)}
    print("      function signatures:", fn_args)


def part_c():
    names = sorted(f for f in os.listdir(OLDF) if f.endswith(".npz"))
    assert len(names) == 152, len(names)
    t0 = time.time()
    worst, n_bits_bad, n_nan_bad, n_cells = 0.0, 0, 0, 0
    extra_nan = 0          # cells where feats is finite but rel is NaN
    feat_nan = 0
    allnan_window = 0      # cells where lg finite? never; cells where log NaN but feats finite
    for name in names:
        feats, labels, rel, relf = load(name)
        lg = rel_feats.log_size(feats)
        prod = (lg - rel_feats.running_median(lg)).astype(np.float32)
        if not np.array_equal(prod, rel, equal_nan=True):
            n_bits_bad += 1
        if not np.array_equal(np.isnan(lg), np.isnan(rel)):
            n_nan_bad += 1
        fin_feat = np.isfinite(feats[:, :, :8])
        extra_nan += int(np.sum(fin_feat & np.isnan(rel)))
        feat_nan += int(np.sum(~fin_feat))
        allnan_window += int(np.sum(fin_feat & ~np.isfinite(lg)))
        # explicit nanmedian loop on a sample: first 5 windows, windows 295..304, 45 random
        n = len(lg)
        rs = np.random.default_rng(len(name) * 7919 + n)
        pick = np.unique(np.concatenate([np.arange(min(5, n)), np.arange(295, min(305, n)),
                                         rs.choice(n, min(n, 45), replace=False)]))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for i in pick:
                ex = lg[i] - np.nanmedian(lg[max(0, i - WIN + 1):i + 1], axis=0)
                if not np.array_equal(np.isnan(ex), np.isnan(rel[i])):
                    n_nan_bad += 1
                ok = ~np.isnan(ex)
                if ok.any():
                    worst = max(worst, float(np.max(np.abs(ex[ok] - rel[i].astype(np.float64)[ok]))))
        n_checked = globals().setdefault("_nchk", [0])
        n_checked[0] += len(pick)
        n_cells += rel.size
    print("(c) %d files, %d cells: stored rel != production recompute in %d files; "
          "NaN pattern of rel != NaN pattern of log/explicit in %d files; "
          "explicit nanmedian loop on %d sampled windows vs stored max|diff| %.2e  [%.0f s]"
          % (len(names), n_cells, n_bits_bad, n_nan_bad, globals()["_nchk"][0], worst,
             time.time() - t0))
    print("(c) size-feature cells NaN in OLD feats: %d; feats finite but rel NaN (log of <=0): %d; "
          "feats finite but log non-finite: %d" % (feat_nan, extra_nan, allnan_window))


if __name__ == "__main__":
    ast_label_scan()
    part_a_b()
    part_c()
