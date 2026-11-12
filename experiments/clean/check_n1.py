"""Checks C1 and C2 on N1 (clean_rule.txt v2 section 1). Read-only. cwd C:\EEG."""
import re, sys
from pathlib import Path
import torch
sys.path.insert(0, r"C:\EEG")
sys.dont_write_bytecode = True
from train_lopo import subject_groups

NEW = Path(r"C:\Users\user\AppData\Local\Temp\claude\36c1f29744f93fe6\c7848ca6-bfef-414b-a321-d6686c1a43f8\scratchpad\clean")
txt = (NEW / "train_fixed15.log").read_text(encoding="utf-8", errors="replace")
lines = re.split(r"[\r\n]+", txt)

print("C1")
print("  'Early stopping' lines:", sum("early stopping" in l.lower() for l in lines))
print("  'cached' lines:", sum("cached" in l for l in lines))
max_ep = {}
for l in lines:
    m = re.search(r"Fold (\d+) Epoch (\d+)/(\d+)", l)
    if m:
        f, e, t = map(int, m.groups())
        max_ep[f] = max(max_ep.get(f, 0), e)
        assert t == 15, l
print("  folds seen:", len(max_ep), "| min top epoch:", min(max_ep.values()),
      "| folds not at 15:", [f for f, e in max_ep.items() if e != 15])
holds = [l.strip() for l in lines if re.match(r"\s*\[\d+/23\] holding out", l)]
print("  'holding out' lines:", len(holds))
print("  header:", [l.strip() for l in lines if "early stop" in l and "epochs" in l])

print("C2")
groups = subject_groups()
n_cases = sum(len(g) for g in groups)
bad = []
for g in groups:
    name = "_".join(g)
    p = Path("checkpoints/lopo") / ("fixed15_%s.pt" % name)
    ck = torch.load(p, map_location="cpu", weights_only=False)
    want = n_cases - len(g)
    ok = (ck.get("early_stop") is False and ck.get("epochs") == 15
          and ck.get("n_train_cases") == want and list(ck["val_patients"]) == g)
    if not ok:
        bad.append((name, ck.get("early_stop"), ck.get("epochs"), ck.get("n_train_cases"), want))
print("  checkpoints:", len(groups), "| failing:", bad)
print("  fixed15 files on disk:", len(list(Path("checkpoints/lopo").glob("fixed15_*.pt"))))
