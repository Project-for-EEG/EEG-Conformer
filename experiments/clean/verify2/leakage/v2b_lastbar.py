"""max batch index displayed in each fold's epoch-15 progress bar (N1 log). Read-only."""
import re
from pathlib import Path
p = Path(r"C:\Users\user\AppData\Local\Temp\claude\36c1f29744f93fe6\c7848ca6-bfef-414b-a321-d6686c1a43f8\scratchpad\clean\train_fixed15.log")
mx = {}
for l in re.split(r"[\r\n]+", p.read_text(encoding="utf-8", errors="replace")):
    m = re.search(r"Fold (\d+) Epoch 15/15:\s*\d+%\|.*?\|\s*(\d+)/(\d+)", l)
    if m:
        f, n, t = map(int, m.groups())
        mx[f] = (max(mx.get(f, (0, t))[0], n), t)
print("fold: (max displayed batch at epoch 15, batches per epoch)")
print(mx)
print("min fraction reached:", min(n / t for n, t in mx.values()))
