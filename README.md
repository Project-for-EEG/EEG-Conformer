# Patient-Independent Seizure Detection on CHB-MIT

Detecting epileptic seizures from EEG, tested only on patients the model has never seen.

**Catches 83% of seizures, at 4.4 false alarms per hour.**
170 hours of held-out recording, 24 cases (23 subjects), 166 seizures.

Published work on this dataset reaches 75-85% at 1-5 false alarms/hour. Not clinically
usable, which needs under 1 per hour.

**[Full report](report.html)** has the method and every negative result.

## A bug had erased most of the labels

The code reading seizure timestamps took the first number on each line:

```python
re.search(r'(\d+)', line)   # "Seizure 1 Start Time: 1724 seconds" captures "1"
```

Every seizure was stored as second 1 to second 1, so nothing was labelled. **18 of 24
patients lost their entire ground truth**, and the six that parsed correctly hid it.

Fixing it moved the model from **worse than random guessing (0.32) to 0.78** -- more than
every other experiment combined.

## What did not work

Eleven things tried, three helped: fixing the labels, smoothing predictions over time,
and adapting to individual patients.

Eight changed nothing or made it worse: synthetic data (SMOTE), weighted sampling, more
background, more seizures, three training class balances, a 2.8x larger model, rolling
recalibration, and a pretrained foundation model.

That last one is the notable result. **CBraMod** (4.9M parameters, pretrained on 9,000
hours of hospital EEG) was compared on identical folds, patients and seed:

| | from scratch | CBraMod |
|---|---|---|
| AUC | 0.7974 | 0.7985 |

A difference of **+0.001**. 40x the parameters and 9,000 hours of pretraining bought
nothing.

Class imbalance was never the real problem either -- the data only looked imbalanced
because the bug had deleted the positive labels. The true rate is 1.363% seizure, which
is simply what scalp EEG is.

## Things that fooled the measurements

- **`chb01` and `chb21` are the same child**, 18 months apart. Splitting them across folds
  leaks, and it barely moves AUC while costing 8 points of event sensitivity.
- **Counting alarms as runs is gameable.** Flag everything, they merge into one run
  overlapping a real seizure, and you score almost no false alarms at 100% sensitivity.
- **Which patients land in which fold moves the result by 8 points.** Anything smaller is
  noise at 3 folds.
- **Scoring rules matter.** The same predictions give 4.4 or 14.9 false alarms/hour
  depending on the scorer. Always say which one.

## Run it

```bash
pip install -r requirements.txt
python edf_to_npz.py --fetch-missing
python train_memory_efficient.py --folds 3 --epochs 15 --max-segments 2000 --balance none
python evaluate_end_to_end.py --target-fa 2.0
python score_szcore.py --cache endtoend_probs.npz
```

CUDA GPU, ~15 GB RAM. Training ~40 min, evaluation ~25 min. Data not included (205 GB);
the first command downloads it from PhysioNet.

MIT License
