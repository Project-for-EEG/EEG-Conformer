"""
Batch size sweep. Writes results to a file as each setting completes.

The trainer runs at batch 32 with num_workers=0 and pin_memory=False, the last
two hardcoded over config.py's own num_workers=4 / pin_memory=True. GPU
utilisation during real runs sits around 30-60%, so the model waits rather than
computes, and a 23-fold leave-one-patient-out run costs 13 hours.

Workers are deliberately not swept. A first attempt including them ran for 22
minutes without finishing: the dataset is a TensorDataset already resident in
RAM, and on Windows each worker is a spawned process that needs the whole
tensor shipped to it. At 20,000 windows that is 1.9 GB per worker, which costs
far more than the loading it removes. num_workers > 0 is the right fix for a
disk-backed dataset and the wrong one here.

Results go to logs/bench_batch.log line by line rather than through a shell
pipe, because grep and tee both buffer and a long run then looks identical to a
hung one.
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from config import get_config
from model import create_model

N = 12000
STEPS = 80
FOLD_WINDOWS = 74000
EPOCHS = 15
OUT = Path("logs/bench_batch.log")


def emit(line):
    print(line)
    sys.stdout.flush()
    with OUT.open("a") as fh:
        fh.write(line + "\n")


def main():
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text("")
    cfg = get_config()
    dev = torch.device("cuda")

    emit("GPU %s" % torch.cuda.get_device_name(0))
    emit("%d steps per setting, %d windows in RAM" % (STEPS, N))
    emit("")
    emit("%5s %5s %10s %12s %7s" % ("batch", "pin", "samples/s", "15ep fold",
                                    "speedup"))

    X = torch.from_numpy(np.random.randn(N, 23, 1024).astype(np.float32))
    y = torch.from_numpy(np.random.randint(0, 2, N)).long()
    ds = TensorDataset(X, y)
    lossf = torch.nn.CrossEntropyLoss()

    base = None
    for batch, pin in [(32, False), (64, False), (128, False), (256, False),
                       (512, False), (256, True), (512, True)]:
        model = create_model(cfg.model).to(dev)
        opt = torch.optim.AdamW(model.parameters(),
                                lr=cfg.training.learning_rate)
        dl = DataLoader(ds, batch_size=batch, shuffle=True, num_workers=0,
                        pin_memory=pin, drop_last=True)

        it = iter(dl)
        for _ in range(3):
            xb, yb = next(it)
            opt.zero_grad()
            lossf(model(xb.to(dev)), yb.to(dev)).backward()
            opt.step()
        torch.cuda.synchronize()

        seen = 0
        t0 = time.time()
        for _ in range(STEPS):
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

        rate = seen / (time.time() - t0)
        if base is None:
            base = rate
        emit("%5d %5s %10.0f %10.1fm %6.2fx"
             % (batch, pin, rate, EPOCHS * FOLD_WINDOWS / rate / 60,
                rate / base))

        del model, opt, dl
        torch.cuda.empty_cache()

    emit("")
    emit("'15ep fold' is one fold of %d windows; a LOPO run is 23 of those."
         % FOLD_WINDOWS)


if __name__ == "__main__":
    main()
