"""leakage audit, misc: (a) shallow-copy alias of best_model_state on the real model, CPU;
(b) every preprocessed_data/<case>/ file carries that case's prefix; (c) train_log_n1.json content.
Read-only. cwd C:\EEG."""
import json, sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, r"C:\EEG")
sys.dont_write_bytecode = True
from config import get_config
from model import create_model
torch.manual_seed(0)
cfg = get_config()
m = create_model(cfg.model)
opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-3)
snap = m.state_dict().copy()
deep = {k: v.clone() for k, v in m.state_dict().items()}
m.train()
x = torch.randn(4, 23, 1024); y = torch.tensor([0, 1, 0, 1])
for _ in range(2):
    opt.zero_grad(); loss = torch.nn.functional.cross_entropy(m(x), y); loss.backward(); opt.step()
cur = m.state_dict()
alias = all(torch.equal(snap[k], cur[k]) for k in cur)
moved = sum(not torch.equal(deep[k], cur[k]) for k in cur)
print("a) after 2 AdamW steps: shallow-copied 'best' state equals live weights for every tensor:", alias,
      "| tensors that actually changed vs a deep copy:", moved, "of", len(cur))
NEW = Path(r"C:\Users\user\AppData\Local\Temp\claude\36c1f29744f93fe6\c7848ca6-bfef-414b-a321-d6686c1a43f8\scratchpad\clean")
bad = [(d.name, f.name) for d in Path("preprocessed_data").iterdir() if d.is_dir()
       for f in d.glob("*.npz") if not f.name.lower().startswith(d.name.lower() + "_")]
print("b) files whose name does not start with their case folder:", bad)
log = json.loads((NEW / "trees_out" / "train_log_n1.json").read_text())
print("c) train_log_n1.json: %d rows, arms %s, folds %d, n_train %s" % (
    len(log), sorted({r["arm"] for r in log}), len({r["fold"] for r in log}), sorted({r["n_train"] for r in log})))
