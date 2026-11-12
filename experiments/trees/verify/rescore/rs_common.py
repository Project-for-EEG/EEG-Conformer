"""Independent re-scoring helpers (label: rescore). Does not import trees_score.py or
trees_train.py. Event scoring goes straight to the timescoring package."""
import json
import sys
from pathlib import Path

import numpy as np
from timescoring import scoring
from timescoring.annotations import Annotation

sys.dont_write_bytecode = True
TREES = Path(r"C:\Users\user\AppData\Local\Temp\claude\36c1f29744f93fe6"
             r"\b3889703-629c-4c5a-b583-69f722743ff4\scratchpad\trees")
OUT = TREES / "out"
HERE = TREES / "verify" / "rescore"
SEC_PER_WIN = 2.0          # 512-sample stride at 256 Hz
FS = 1.0 / SEC_PER_WIN
K, MINRUN = 5, 3
TARGETS = ("0.05", "0.1", "0.15", "0.2", "0.3", "0.4", "0.5", "0.75", "1.0", "1.5",
           "2.0", "3.0", "4.0", "5.0")
BUDGETS = (5, 10, 25, 50, 100)
ARMS = ["A0", "Tamp_s0", "Tamp_s1", "Tamp_s2", "Tfree_s0", "Tfree_s1", "Tfree_s2"]
GROUPS = {"A0": ["A0"], "Tamp": ["Tamp_s0", "Tamp_s1", "Tamp_s2"],
          "Tfree": ["Tfree_s0", "Tfree_s1", "Tfree_s2"]}


def my_smooth(p, k=K):
    """Centred moving average, window shrinks at the edges; identity if len < k."""
    p = np.asarray(p, np.float64)
    n = len(p)
    if k <= 1 or n < k:
        return p
    h = k // 2
    pad = np.concatenate([np.zeros(h), p, np.zeros(h)])
    one = np.concatenate([np.zeros(h), np.ones(n), np.zeros(h)])
    num = sum(pad[i:i + n] for i in range(k))
    den = sum(one[i:i + n] for i in range(k))
    return num / den


def runs(mask):
    m = np.concatenate([[False], np.asarray(mask, bool), [False]])
    e = np.flatnonzero(m[1:] != m[:-1])
    return e[::2], e[1::2]


def my_detect_from_smoothed(sm, t, minrun=MINRUN):
    pred = sm >= t
    lo, hi = runs(pred)
    out = np.zeros(len(sm), bool)
    for a, b in zip(lo, hi):
        if b - a >= minrun:
            out[a:b] = True
    return out


def ev_count(truth, pred):
    s = scoring.EventScoring(Annotation(np.asarray(truth, bool), FS),
                             Annotation(np.asarray(pred, bool), FS))
    return s.tp, s.fp, s.refTrue


def case_count_sm(sm_blocks, ys, t):
    """(tp, fp, ref, seconds) for one case from pre-smoothed blocks."""
    tp = fp = ref = 0
    sec = 0.0
    for sm, y in zip(sm_blocks, ys):
        a, b, r = ev_count(y.astype(bool), my_detect_from_smoothed(sm, t))
        tp += a
        fp += b
        ref += r
        sec += len(y) * SEC_PER_WIN
    return tp, fp, ref, sec


def load_all():
    P, T = {}, {}
    for tag in "abc":
        part = np.load(OUT / ("preds_%s.npy" % tag), allow_pickle=True).item()
        tp = json.loads((OUT / ("thresholds_%s.json" % tag)).read_text())
        for a in part:
            assert a not in P
            P[a] = part[a]
            T[a] = tp[a]
    return P, T
