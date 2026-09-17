"""
Where does training time actually go?

The trainer runs at batch 32 with num_workers=0 and pin_memory=False, the last
two hardcoded over config.py's own num_workers=4 / pin_memory=True. GPU
utilisation during real runs sits around 30-60%, so the model is waiting rather
than computing, and a 23-fold leave-one-patient-out run costs 13 hours.

This measures the two candidate fixes instead of assuming them:

  batch size    a 121k-parameter model on 32 windows finishes almost instantly
                and then waits for the next 32. Larger batches should keep the
                GPU busy.

  worker count  the usual fix for a loader bottleneck, but the premise may not
                hold here. The dataset is a TensorDataset already resident in
                RAM, not something read from disk per item, and on Windows
                workers are spawned as separate processes that need the tensors
                shipped to them. That overhead can easily exceed the loading it
                removes, so this is measured rather than assumed to help.

Reports throughput and the implied wall clock for one 15-epoch fold at the
74,000 training windows a merged-cohort fold actually uses.

Usage:
    python bench_loader.py
    python bench_loader.py --steps 150
"""
import argparse
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from config import get_config
from model import create_model

FOLD_WINDOWS = 74000     # a merged-cohort training fold
EPOCHS = 15


def run(ds, model, opt, lossf, dev, batch, workers, pin, steps):
    """Steady-state samples/second, warm-up excluded."""
    dl = DataLoader(ds, batch_size=batch, shuffle=True, num_workers=workers,
                    pin_memory=pin, persistent_workers=bool(workers),
                    drop_last=True)
    it = iter(dl)
    for _ in range(3):                      # warm up: spawn, allocate, compile
        try:
            xb, yb = next(it)
        except StopIteration:
            it = iter(dl)
            xb, yb = next(it)
        opt.zero_grad()
        lossf(model(xb.to(dev, non_blocking=pin)),
              yb.to(dev, non_blocking=pin)).backward()
        opt.step()
    torch.cuda.synchronize()

    seen = 0
    t0 = time.time()
    for _ in range(steps):
        try:
            xb, yb = next(it)
        except StopIteration:
            it = iter(dl)
            xb, yb = next(it)
        opt.zero_grad()
        lossf(model(xb.to(dev, non_blocking=pin)),
              yb.to(dev, non_blocking=pin)).backward()
        opt.step()
        seen += len(yb)
    torch.cuda.synchronize()
    dt = time.time() - t0
    del dl
    return seen / dt


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--n", type=int, default=20000,
                    help="synthetic windows held in RAM, matching how the "
                         "trainer holds a fold")
    args = ap.parse_args()

    cfg = get_config()
    dev = torch.device("cuda")
    print("GPU: %s" % torch.cuda.get_device_name(0))
    print("model: %d parameters, input (%d, %d)\n"
          % (sum(p.numel() for p in create_model(cfg.model).parameters()),
             cfg.model.num_channels, cfg.model.sequence_length))

    X = torch.from_numpy(
        np.random.randn(args.n, cfg.model.num_channels,
                        cfg.model.sequence_length).astype(np.float32))
    y = torch.from_numpy(np.random.randint(0, 2, args.n)).long()
    ds = TensorDataset(X, y)

    settings = [
        (32, 0, False, "current"),
        (64, 0, False, ""),
        (128, 0, False, ""),
        (256, 0, False, ""),
        (512, 0, False, ""),
        (256, 0, True, "pinned"),
        (256, 2, True, "2 workers"),
        (256, 4, True, "4 workers"),
    ]

    print("%-6s %8s %5s  %11s %9s  %s"
          % ("batch", "workers", "pin", "samples/s", "fold", "note"))
    base = None
    for batch, workers, pin, note in settings:
        model = create_model(cfg.model).to(dev)
        opt = torch.optim.AdamW(model.parameters(),
                                lr=cfg.training.learning_rate)
        lossf = torch.nn.CrossEntropyLoss()
        try:
            rate = run(ds, model, opt, lossf, dev, batch, workers, pin,
                       args.steps)
        except RuntimeError as e:
            print("%-6d %8d %5s  %11s %9s  %s"
                  % (batch, workers, pin, "-", "-", str(e).split("\n")[0][:40]))
            continue
        mins = EPOCHS * FOLD_WINDOWS / rate / 60
        if base is None:
            base = rate
        speed = "" if note == "current" else "%.1fx" % (rate / base)
        print("%-6d %8d %5s  %11.0f %7.1fm  %s %s"
              % (batch, workers, pin, rate, mins, note, speed))
        del model, opt
        torch.cuda.empty_cache()

    print("\n'fold' is one 15-epoch fold at %d windows; a 23-fold LOPO run is "
          "23 of those." % FOLD_WINDOWS)


if __name__ == "__main__":
    main()
