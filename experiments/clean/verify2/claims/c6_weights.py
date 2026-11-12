"""C6: compare A0 and N1 fold checkpoints weight by weight on CPU. Read-only."""
import torch, numpy as np
from pathlib import Path
L = Path(r"C:\EEG\checkpoints\lopo")
for fold in ["CHB01_CHB21", "CHB02", "CHB03"]:
    a = torch.load(L / ("lopo_%s.pt" % fold), map_location="cpu", weights_only=False)
    n = torch.load(L / ("fixed15_%s.pt" % fold), map_location="cpu", weights_only=False)
    sa, sn = a["model_state_dict"], n["model_state_dict"]
    assert list(sa) == list(sn)
    eq = all(torch.equal(sa[k], sn[k]) for k in sa)
    mx = max(float((sa[k].double() - sn[k].double()).abs().max()) for k in sa if sa[k].numel())
    print(fold, "tensors", len(sa), "all bit-equal:", eq, "max|dw|", mx,
          "| A0 keys:", sorted(set(a) - {"model_state_dict"}),
          "| A0 metrics", {k: a.get("metrics", {}).get(k) for k in ("auc",)} if isinstance(a.get("metrics"), dict) else None,
          "| N1 early_stop/epochs", n.get("early_stop"), n.get("epochs"))
