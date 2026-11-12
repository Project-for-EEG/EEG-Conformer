import torch, torch.nn as nn, torch.optim as optim
torch.manual_seed(0)
m = nn.Sequential(nn.Conv1d(2,4,3), nn.BatchNorm1d(4), nn.Flatten(), nn.Dropout(0.1), nn.Linear(4*8,2))
for foreach in (False, True):
    opt = optim.AdamW(m.parameters(), lr=1e-2, foreach=foreach)
    scaler = torch.amp.GradScaler("cpu")
    x = torch.randn(16,2,10); y = torch.randint(0,2,(16,))
    best = m.state_dict().copy()
    snap = {k: v.clone() for k, v in m.state_dict().items()}
    m.train()
    for _ in range(3):
        opt.zero_grad()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            loss = nn.functional.cross_entropy(m(x), y)
        scaler.scale(loss).backward(); scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        scaler.step(opt); scaler.update()
    now = m.state_dict()
    print("foreach", foreach, "| best aliases live:", all(torch.equal(best[k], now[k]) for k in now),
          "| changed:", sorted(k for k in now if not torch.equal(snap[k], now[k])))
