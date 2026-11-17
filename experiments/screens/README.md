# Screens

Finished tests of ideas that are not part of the main pipeline, and two data
builders: one for `train_natural_prior.py` here, one for `train_cbramod.py`
in the top folder. They moved here from the top folder unchanged, so their
usage lines still show the old `python name.py` form.

Most of them import the main code (for example `config.py` and `model.py`)
from the top folder, and all of them use files and folders there. Run them
from the top folder as modules, e.g. `python -m experiments.screens.temporal_context`.

| script | what it does |
|---|---|
| `build_memmap.py` | packs every window into one memory-mapped array, for `train_natural_prior.py` |
| `train_natural_prior.py` | trains at a chosen seizure share (`--prior`), or at the true class ratio by default |
| `train_hard_negatives.py` | retrains each fold with part of the background (`--hard-fraction`, half by default) taken from the windows that fold's existing model scores highest; the rest stays random |
| `temporal_context.py` | tests a learned filter over the probability sequence, against the moving average and minimum-run rule |
| `edf_to_cbramod.py` | builds `cbramod_data/` in the format CBraMod needs; `train_cbramod.py` in the top folder reads it |
