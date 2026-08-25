# EEG-Conformer: Patient-Independent Seizure Detection

Seizure detection on [CHB-MIT](https://physionet.org/content/chbmit/1.0.0/), evaluated
patient-independently at the event level.

**83% of seizures caught, at 4.4 false alarms per hour.** Measured on 170 h held-out,
24 patients, 166 seizures. Patient-level 3-fold CV: AUC 0.7795 +/- 0.0168.

The dataset is 396,663 windows of 4 s over 220 h of EEG, of which **5,408 are seizure:
1.363%**, or about 180 minutes of seizure in total. Per patient it ranges from 0.157%
(CHB06) to 5.081% (CHB08).

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

Of eleven interventions, three helped: label repair, temporal smoothing, and selective
per-patient personalisation. Eight did nothing or made things worse: SMOTE, weighted
sampling, more background, more seizures, training class prior, a larger model, rolling
threshold recalibration, and swapping in a pretrained foundation model (below).

Class imbalance was never the real problem. The data only looked imbalanced because the
bug had erased the positive labels.

## A pretrained foundation model does not help

CBraMod (ICLR 2025, 4.9M params, pretrained on 9,000 h of hospital EEG) was fine-tuned
and compared against the 121k-param from-scratch model. Paired: identical folds, patients,
seed, windows and labels, with only the architecture differing.

| fold | EEG-Conformer | CBraMod | delta |
|---|---|---|---|
| 1 | 0.8353 | 0.8682 | +0.0329 |
| 2 | 0.7630 | 0.7354 | -0.0276 |
| 3 | 0.7939 | 0.7919 | -0.0020 |
| **mean** | **0.7974 +/- 0.0296** | **0.7985 +/- 0.0544** | **+0.0011** |

A mean paired difference of **+0.001**, with folds disagreeing in both directions and a
spread of differences (0.025) larger than the difference itself. CBraMod was also less
stable across folds. 40x the parameters and 9,000 h of pretraining bought nothing.

This matches independent benchmarks, which find these models frequently match their own
random initialisation on seizure tasks. Reproduce with `train_cbramod.py`; the data
conversion is `edf_to_cbramod.py`, verified to produce identical labels and window counts
to the main pipeline.

CHB03 and CHB05 are excluded from that comparison: their local copies predate `raw_data/`
and do not match the PhysioNet recordings, so they cannot be paired. 22 patients remain.

## Gotchas

- `chb01` and `chb21` are the same child recorded 18 months apart. Keep them in one fold.
- Counting alarm runs is gameable: flag everything and they merge into one run that
  overlaps a real seizure, booking almost no false alarms at 100% sensitivity.
- Patient grouping moves the headline by 8 points at 3 folds.

## Data

Not included (205 GB). CHB-MIT is public on PhysioNet, and `edf_to_npz.py
--fetch-missing` downloads and converts it.

MIT License
