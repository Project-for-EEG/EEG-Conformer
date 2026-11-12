"""Read-only. Parse N1 (train_fixed15.log) and A0 (C:\\EEG\\lopo_run.log) training logs:
rounds reached per fold, early-stopping lines, cached lines. Checks the EARLY / SAME15
lists used by clean_score.py against the logs."""
import re
from pathlib import Path

NEW = Path(r"C:\Users\user\AppData\Local\Temp\claude\36c1f29744f93fe6"
           r"\c7848ca6-bfef-414b-a321-d6686c1a43f8\scratchpad\clean")


def parse(path):
    txt = Path(path).read_text(encoding="utf-8", errors="replace")
    lines = re.split(r"[\r\n]+", txt)
    folds, cur, maxep, es = [], None, {}, {}
    n_cached = sum("cached" in l.lower() for l in lines)
    for l in lines:
        m = re.match(r"\s*\[(\d+)/23\] holding out (.*?) \|", l)
        if m:
            cur = int(m.group(1))
            folds.append((cur, m.group(2).strip()))
        m = re.search(r"Fold (\d+) Epoch (\d+)/(\d+)", l)
        if m:
            f, e, t = map(int, m.groups())
            maxep[f] = max(maxep.get(f, 0), e)
        m = re.search(r"Early stopping at epoch (\d+)", l)
        if m:
            es[cur] = int(m.group(1))
    return folds, maxep, es, n_cached, len(lines)


for name, path in (("N1", NEW / "train_fixed15.log"), ("A0", r"C:\EEG\lopo_run.log")):
    folds, maxep, es, n_cached, nl = parse(path)
    print("%s %s: %d lines, %d 'holding out' lines, %d 'cached' lines, %d 'Early stopping' lines"
          % (name, path, nl, len(folds), n_cached, len(es)))
    keys = sorted(maxep)
    print("  fold keys in Epoch lines: %s" % keys)
    for i, held in folds:
        print("  [%2d] %-20s max epoch %s  early-stop %s" % (i, held, maxep.get(i),
                                                           es.get(i)))
