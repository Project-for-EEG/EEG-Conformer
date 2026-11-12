import torch, glob, os, hashlib
os.chdir(r"C:\EEG")
def show(pat):
    for p in sorted(glob.glob(pat)):
        ck = torch.load(p, map_location="cpu", weights_only=False)
        keys = [k for k in ck if k not in ("model_state_dict",)]
        meta = {k: ck[k] for k in keys if k not in ("metrics","model_config")}
        mc = ck.get("model_config", {})
        print(os.path.basename(p), {k: (v if not isinstance(v,(list,tuple)) or len(v)<15 else f"list[{len(v)}]") for k,v in meta.items()}, "nch", mc.get("num_channels"), "auc", round(ck.get("metrics",{}).get("auc",float('nan')),4))
show(r"checkpoints\lopo\lopo_*.pt")
print()
show(r"checkpoints\lopo\sienaonly__*.pt")
print()
show(r"checkpoints\helscv*_67.pt")
