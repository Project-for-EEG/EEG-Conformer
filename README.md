# EEG-Conformer: Patient-Independent Seizure Detection

Seizure detection on [CHB-MIT](https://physionet.org/content/chbmit/1.0.0/), evaluated
patient-independently at the event level.

**83% of seizures caught, at 4.4 false alarms per hour.** Measured on 170 h held-out,
24 patients, 166 seizures. Patient-level 3-fold CV: AUC 0.7795 +/- 0.0168.

Scored with [SzCORE](https://epilepsybenchmarks.com) event rules; a stricter scorer gives
78% at 14.9 FA/h, so always name the scorer. Comparable published work reaches 75-85% at
1-5 FA/h. Not clinically usable, which needs under 1 FA/hour.

**[Full report](report.html)** covers the method, all negative results, and the pitfalls.

## Run it

```bash
pip install -r requirements.txt
python edf_to_npz.py --fetch-missing
python train_memory_efficient.py --folds 3 --epochs 15 --max-segments 2000 --balance none
python evaluate_end_to_end.py --target-fa 2.0
python score_szcore.py --cache endtoend_probs.npz
```

CUDA GPU, ~15 GB RAM. Training ~40 min, evaluation ~25 min.

## The label bug

The annotation parser took the first number on each line:

```python
re.search(r'(\d+)', line)   # "Seizure 1 Start Time: 1724 seconds" captures "1"
```

Every seizure was stored as second 1 to second 1. **18 of 24 patients lost their entire
ground truth.** Fixing it moved AUC from 0.32 to 0.78, more than every other experiment
combined.

## What didn't work

Of ten interventions, three helped: label repair, temporal smoothing, and selective
per-patient personalisation. Seven did nothing or made things worse: SMOTE, weighted
sampling, more background, more seizures, training class prior, a larger model, and
rolling threshold recalibration.

Class imbalance was never the real problem. The data only looked imbalanced because the
bug had erased the positive labels.

## Gotchas

- `chb01` and `chb21` are the same child recorded 18 months apart. Keep them in one fold.
- Counting alarm runs is gameable: flag everything and they merge into one run that
  overlaps a real seizure, booking almost no false alarms at 100% sensitivity.
- Patient grouping moves the headline by 8 points at 3 folds.

## Data

Not included (205 GB). CHB-MIT is public on PhysioNet, and `edf_to_npz.py
--fetch-missing` downloads and converts it.

MIT License
