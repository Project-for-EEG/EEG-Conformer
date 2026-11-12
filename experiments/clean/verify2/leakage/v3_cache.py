"""leakage audit, part 3: NEW\\n1_cache wiring. CPU only, read-only. cwd C:\\EEG.

a) every n1_cache file <-> one preprocessed_data recording; keys = labels + 23 logit_<fold>;
   lengths and labels equal to preprocessed_data and to A0's caches.
b) no two fold keys hold the same array (a copy/paste wiring bug would show up here).
c) recompute ~300 windows per (fold, file) on CPU from fixed15_<fold>.pt AND lopo_<fold>.pt
   and compare with the cached N1 / A0 logits.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(r"C:\EEG")
sys.path.insert(0, str(REPO))
sys.dont_write_bytecode = True
from config import ModelConfig  # noqa: E402
from model import create_model  # noqa: E402
from train_lopo import subject_groups  # noqa: E402

torch.set_num_threads(2)
NEW = Path(r"C:\Users\user\AppData\Local\Temp\claude\36c1f29744f93fe6"
           r"\c7848ca6-bfef-414b-a321-d6686c1a43f8\scratchpad\clean")
OLD2 = Path(r"C:\Users\user\AppData\Local\Temp\claude\36c1f29744f93fe6"
            r"\f241281f-add4-41ac-9210-f47dab0b83b0\scratchpad")
N1 = NEW / "n1_cache"
PC1, PC2 = OLD2 / "pilot_cache", OLD2 / "pilot_cache2"
PREP = REPO / "preprocessed_data"
CK = REPO / "checkpoints" / "lopo"
FOLDS = ["_".join(g) for g in subject_groups()]
TRIAL = ["CHB12", "CHB15", "CHB24", "CHB13", "CHB06", "CHB16"]

res = {}
# ---- a) file mapping, keys, labels
prep_files = sorted((c.name, f) for c in PREP.iterdir() if c.is_dir() for f in sorted(c.glob("*.npz")))
want_names = sorted("%s__%s.npz" % (c, f.stem) for c, f in prep_files)
have_names = sorted(p.name for p in N1.glob("*.npz"))
print("prep recordings:", len(prep_files), "| n1_cache files:", len(have_names),
      "| same names:", want_names == have_names)
bad_keys, bad_lab, bad_len, dup_pairs = [], [], [], []
n_win = 0
win_per_case = {}
for c, f in prep_files:
    name = "%s__%s.npz" % (c, f.stem)
    with np.load(f) as d:
        y = d["labels"]
    with np.load(N1 / name) as dn:
        keys = set(dn.files)
        want = {"labels"} | {"logit_" + k for k in FOLDS}
        if keys != want:
            bad_keys.append((name, sorted(keys ^ want)))
        yl = dn["labels"]
        if not np.array_equal(yl.astype(np.int64), y.astype(np.int64)):
            bad_lab.append(name)
        lg = {k: dn["logit_" + k] for k in FOLDS}
    for k, v in lg.items():
        if v.shape != (len(y),) or v.dtype != np.float32:
            bad_len.append((name, k, v.shape, str(v.dtype)))
    # b) duplicates between fold keys
    ks = list(lg)
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            if np.array_equal(lg[ks[i]], lg[ks[j]]):
                dup_pairs.append((name, ks[i], ks[j]))
    # A0 caches' labels
    with np.load(PC1 / name) as d1, np.load(PC2 / name) as d2:
        if not (np.array_equal(d1["labels"], yl) and np.array_equal(d2["labels"], yl)):
            bad_lab.append(("A0cache", name))
    n_win += len(y)
    win_per_case[c] = win_per_case.get(c, 0) + len(y)
print("windows total:", n_win)
print("bad key sets:", bad_keys[:3], "| bad labels:", bad_lab[:3], "| bad shapes/dtypes:", bad_len[:3])
print("identical arrays under two different fold keys:", len(dup_pairs), dup_pairs[:3])
res["a"] = dict(n_files=len(have_names), names_match=want_names == have_names, n_win=n_win,
                bad_keys=bad_keys, bad_labels=bad_lab, bad_shapes=bad_len, dup_pairs=dup_pairs)

# ---- c) recompute on CPU
def load(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = create_model(ModelConfig(**ck["model_config"]))
    m.load_state_dict(ck["model_state_dict"])
    m.eval()
    return m, list(ck["val_patients"])


def logits(m, X):
    with torch.no_grad():
        out = m(torch.from_numpy(X).float())
    return (out[:, 1] - out[:, 0]).double().numpy()


# (fold, recording): one held-out recording of the fold and one training-patient recording
plan = [("CHB03", "CHB03__chb03_03_labeled.npz"), ("CHB03", "CHB10__chb10_12_labeled.npz"),
        ("CHB15", "CHB15__chb15_06_labeled.npz"), ("CHB15", "CHB03__chb03_03_labeled.npz")]
rng = np.random.default_rng(0)
rows = []
for fold, rec in plan:
    if not (N1 / rec).exists():
        cands = sorted(p.name for p in N1.glob(rec.split("__")[0] + "__*.npz"))
        rec = cands[len(cands) // 2]
    case, stem = rec.split("__")
    with np.load(N1 / rec) as dn:
        yl = dn["labels"]
        cn = dn["logit_" + fold].astype(np.float64)
        others = {k: dn["logit_" + k].astype(np.float64) for k in FOLDS if k != fold}
    a0src = PC1 if fold in TRIAL else PC2
    with np.load(a0src / rec) as da:
        ca = da["logit_" + fold].astype(np.float64)
    seiz = np.where(yl == 1)[0]
    idx = np.sort(np.concatenate([rng.choice(len(yl), 250, replace=False),
                                  seiz[:50] if len(seiz) else np.array([], int)]))
    idx = np.unique(idx)
    with np.load(PREP / case / stem) as d:
        X = d["segments"][idx]
    mn, heldn = load(CK / ("fixed15_%s.pt" % fold))
    ma, helda = load(CK / ("lopo_%s.pt" % fold))
    rn, ra = logits(mn, X), logits(ma, X)
    best_other = min(((np.abs(others[k][idx] - rn).max(), k) for k in others))
    row = dict(fold=fold, rec=rec, n=len(idx), n_seiz=int(yl[idx].sum()),
               held_in_ckpt=heldn, recording_is_held_out=case in heldn,
               n1_cache_vs_cpu_fixed15=float(np.abs(cn[idx] - rn).max()),
               a0_cache_vs_cpu_lopo=float(np.abs(ca[idx] - ra).max()),
               n1_cache_vs_cpu_lopo=float(np.abs(cn[idx] - ra).max()),
               corr_n1cache_cpu=float(np.corrcoef(cn[idx], rn)[0, 1]),
               closest_other_fold_key=(float(best_other[0]), best_other[1]),
               logit_sd=float(rn.std()))
    rows.append(row)
    print(json.dumps(row))
    del X
res["c"] = rows
(Path(__file__).with_name("v3_cache.json")).write_text(json.dumps(res, indent=1, default=str))
