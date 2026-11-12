"""
Free check 1: is wider smoothing a cheap form of 'context'?

Saved predictions only (endtoend_uncalibrated.npz). Moving average over k windows,
then today's 3-in-a-row rule. Oracle budgets: one global threshold swept over the
test predictions, labelled as an upper bound, achieved FA printed.
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(r"C:\EEG")
sys.path.insert(0, str(REPO))
from score_official import score_all  # noqa: E402

BUDGETS = (5, 10, 25, 50, 100)


def sweep(data, k, n=3):
    probs = np.concatenate([np.asarray(b[0], dtype=np.float64)
                            for p in data for b in data[p]["blocks"]])
    grid = np.unique(np.concatenate([np.linspace(0.05, 0.999, 100),
                                     np.quantile(probs, np.linspace(0.80, 0.99999, 400))]))
    return [(score_all(data, {p: float(t) for p in data}, k, n)[0], t) for t in grid]


def main():
    out = []
    data = np.load(REPO / "endtoend_uncalibrated.npz", allow_pickle=True)["data"].item()
    stored = {p: float(data[p]["t"]) for p in data}
    out.append("Wider smoothing on endtoend_uncalibrated.npz (24 cases, 168 events)")
    out.append("")
    out.append("stored thresholds (picked for k=5, test-free):")
    for k in (5, 9, 17, 33):
        pl, sb, _ = score_all(data, stored, k, 3)
        out.append("  k=%2d (%3d s)  %3d/%d  sens %.3f  FA/day %6.1f"
                   % (k, 2 * k + 2, pl["caught"], pl["events"], pl["sens"], pl["fa_day"]))
    out.append("")
    out.append("ORACLE best pooled sensitivity within budget (achieved FA/day in brackets):")
    out.append("k      " + "".join("%16s" % ("<=%d/d" % b) for b in BUDGETS))
    for k in (5, 9, 17, 33):
        sw = sweep(data, k)
        row = "k=%-4d " % k
        for b in BUDGETS:
            ok = [s for s in sw if s[0]["fa_day"] <= b]
            if ok:
                pl, t = max(ok, key=lambda s: s[0]["sens"])
                row += "%9.3f (%5.1f)" % (pl["sens"], pl["fa_day"])
            else:
                row += "%16s" % "-"
        out.append(row)
    text = "\n".join(out)
    print(text)
    Path(sys.argv[1]).write_text(text + "\n", encoding="ascii")


if __name__ == "__main__":
    main()
