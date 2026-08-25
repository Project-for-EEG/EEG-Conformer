# Patient-Independent Seizure Detection on CHB-MIT

Detecting epileptic seizures from EEG, tested only on patients the model has never seen.

**Catches 83% of seizures, at 4.4 false alarms per hour.**
Measured on 170 hours of held-out recording: 24 patients, 166 seizures.

Published work on this dataset reaches 75-85% at 1-5 false alarms/hour, so this sits in
the normal range. It is **not clinically usable**, which needs under 1 per hour.

The dataset is 220 hours of EEG cut into 396,663 four-second windows. Only **1.363% are
seizure** (about 180 minutes in total), ranging from 0.157% to 5.081% depending on the
patient.

**[Full report](report.html)** has the method, every negative result, and the pitfalls.

## The main finding: a bug had erased most of the labels

The code that reads seizure timestamps took the first number on each line:

```python
re.search(r'(\d+)', line)   # "Seizure 1 Start Time: 1724 seconds" captures "1"
```

Every seizure was recorded as starting *and* ending at second 1, so nothing was ever
labelled. **18 of the 24 patients lost their entire ground truth**, and the six that
happened to parse correctly hid the problem.

Fixing it took the model from **worse than random guessing (AUC 0.32) to 0.78** -- more
than every other experiment in this project combined.

## What did not work

Eleven things were tried. **Three helped:** fixing the labels, smoothing predictions over
time, and adapting the model to individual patients.

**Eight changed nothing or made it worse:** synthetic data (SMOTE), weighted sampling,
2.7x more background EEG, 52% more seizures, three different training class balances, a
2.8x larger model, rolling threshold recalibration, and swapping in a pretrained
foundation model.

Class imbalance was never the real problem. The data only *looked* imbalanced because the
bug had deleted the positive labels.

## A pretrained foundation model did not help

CBraMod (ICLR 2025, 4.9M parameters, pretrained on 9,000 hours of hospital EEG) was
compared against the 121k-parameter model trained from scratch. Same folds, same patients,
same seed -- only the model differed.

| fold | from scratch | CBraMod | difference |
|---|---|---|---|
| 1 | 0.8353 | 0.8682 | +0.0329 |
| 2 | 0.7630 | 0.7354 | -0.0276 |
| 3 | 0.7939 | 0.7919 | -0.0020 |
| **mean** | **0.7974** | **0.7985** | **+0.0011** |

The folds disagree in both directions and the average difference is one thousandth of a
point. **40x the parameters and 9,000 hours of pretraining bought nothing.** This matches
independent benchmarks, which find these models often score no better than their own
random starting weights on seizure tasks.

## Things that fooled the measurements

- **`chb01` and `chb21` are the same child**, recorded 18 months apart. Splitting them
  across folds leaks. It barely moved AUC but cost 8 points of event sensitivity.
- **Counting alarms as runs is gameable.** Flag everything, the alarms merge into one run
  that overlaps a real seizure, and you book almost no false alarms at 100% sensitivity.
- **Which patients land in which fold moves the result by 8 points.** Anything smaller
  than that is noise at 3 folds.
- **The scoring rules matter more than they sound.** The same predictions give 4.4 or 14.9
  false alarms/hour depending on the scorer. Always say which one.

## Run it

```bash
pip install -r requirements.txt
python edf_to_npz.py --fetch-missing
python train_memory_efficient.py --folds 3 --epochs 15 --max-segments 2000 --balance none
python evaluate_end_to_end.py --target-fa 2.0
python score_szcore.py --cache endtoend_probs.npz
```

CUDA GPU, about 15 GB RAM. Training ~40 min, evaluation ~25 min.
Data is not included (205 GB); `edf_to_npz.py --fetch-missing` downloads it from PhysioNet.

MIT License
