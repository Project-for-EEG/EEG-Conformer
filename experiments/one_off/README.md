# One-off tools

Kept as a record. They moved here from the top folder unchanged, so their
usage lines still show the old `python name.py` form. Run them from the top
folder as modules, e.g. `python -m experiments.one_off.memcheck_fold`.

| script | what it did |
|---|---|
| `repair_labels.py` | fixed the seizure labels in `preprocessed_data/` once, after a parser bug. `--apply` runs the same check as `--verify`, then rewrites the labels in place; `--verify` only reads. |
| `memcheck_fold.py` | loaded the training set of one leave-one-patient-out fold, with the Helsinki babies in it, at several `--max-segments` values, and named the first that left enough memory free. This chose the setting in `scripts/runs/lopo_helsinki.sh`. The `build_memmap.py` it names is in `experiments/screens/`. |
