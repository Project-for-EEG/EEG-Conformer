"""leakage audit, part 2 (C1): parse NEW\\train_fixed15.log and C:\\EEG\\lopo_run.log on \\r and \\n.
Read-only."""
import re
from pathlib import Path

NEW = Path(r"C:\Users\user\AppData\Local\Temp\claude\36c1f29744f93fe6"
           r"\c7848ca6-bfef-414b-a321-d6686c1a43f8\scratchpad\clean")


def parse(path):
    txt = path.read_text(encoding="utf-8", errors="replace")
    lines = re.split(r"[\r\n]+", txt)
    info = {"n_lines": len(lines)}
    info["early"] = [l.strip() for l in lines if "early stopping" in l.lower()]
    info["cached"] = [l.strip() for l in lines if "cached" in l.lower()]
    info["errors"] = [l.strip() for l in lines if re.search(r"Traceback|Error|error|Killed|MemoryError|nan loss", l)]
    info["headers"] = [l.strip() for l in lines if re.match(r"\s*epochs \d+ \| max-segments", l)]
    info["folds_over"] = [l.strip() for l in lines if "folds over" in l]
    holds, trains, saves = [], [], []
    # per fold: max epoch seen, and whether the epoch-15 bar reached n/n
    cur = None
    max_ep, full15, totals = {}, {}, {}
    for l in lines:
        m = re.match(r"\s*\[(\d+)/23\] holding out (.*?) \| training on (\d+) cases", l)
        if m:
            cur = int(m.group(1))
            holds.append((cur, m.group(2), int(m.group(3))))
            continue
        m = re.match(r"\s*train ([\d,]+) seg / (\d+) seizure \| val ([\d,]+) / (\d+)", l)
        if m:
            trains.append((cur,) + tuple(int(x.replace(",", "")) for x in m.groups()))
            continue
        m = re.match(r"\s*AUC ([\d.na]+) \| recall ([\d.]+) \| saved (\S+)", l)
        if m:
            saves.append((cur, m.group(1), m.group(2), m.group(3)))
            continue
        m = re.search(r"Fold (\d+) Epoch (\d+)/(\d+):\s*(\d+)%\|.*?\|\s*(\d+)/(\d+)", l)
        if m:
            f, e, t, pct, n, tot = map(int, m.groups())
            assert f == cur, (f, cur, l[:120])
            max_ep[f] = max(max_ep.get(f, 0), e)
            totals.setdefault(f, set()).add((t, tot))
            if e == t and n == tot:
                full15[f] = True
    info.update(holds=holds, trains=trains, saves=saves, max_ep=max_ep, full15=full15,
                totals=totals)
    return info


n1 = parse(NEW / "train_fixed15.log")
a0 = parse(Path(r"C:\EEG\lopo_run.log"))
for tag, d in (("N1 train_fixed15.log", n1), ("A0 lopo_run.log", a0)):
    print("=" * 70)
    print(tag, "| split lines:", d["n_lines"])
    print("  header lines:", d["headers"])
    print("  'folds over' lines:", d["folds_over"])
    print("  early-stopping lines (%d):" % len(d["early"]), d["early"])
    print("  cached lines (%d):" % len(d["cached"]), d["cached"][:5])
    print("  error-like lines (%d):" % len(d["errors"]), d["errors"][:5])
    print("  holding-out lines:", len(d["holds"]), "| train lines:", len(d["trains"]),
          "| saved lines:", len(d["saves"]))
    print("  folds with progress bars:", len(d["max_ep"]),
          "| max epoch per fold:", d["max_ep"])
    print("  folds whose last-epoch bar reached n/n at epoch==total:",
          sorted(k for k in d["full15"]))
    print("  (epoch total, batches/epoch) per fold:", {k: sorted(v) for k, v in d["totals"].items()})

print("=" * 70)
print("fold-by-fold N1 vs A0 (held-out, n_train_cases, train seg/seiz, val seg/seiz)")
for hn, ha, tn, ta in zip(n1["holds"], a0["holds"], n1["trains"], a0["trains"]):
    same = hn == ha and tn == ta
    print("  %-22s %-22s %s %s %s" % (hn, "", tn[1:], ta[1:], "SAME" if same else "DIFF"))
print("saves N1:", [(s[0], s[1], s[3]) for s in n1["saves"]])
