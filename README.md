# EEG-Conformer: Patient-Independent Seizure Detection

Seizure detection on [CHB-MIT](https://physionet.org/content/chbmit/1.0.0/), evaluated
patient-independently at the event level.

**83% of seizures caught, at 4.4 false alarms per hour.** Measured on 170 h held-out,
24 patients, 166 seizures. Patient-level 3-fold CV gives AUC 0.7795 +/- 0.0168.

Scored with [SzCORE](https://epilepsybenchmarks.com) event rules. The same predictions
give 78% at 14.9 FA/h under a stricter scorer, so always name the scorer. 5 of 24
patients are personalised on one of their *own earlier* seizures. Splits are by patient
and ordered in time, and thresholds never touch the test period.

Comparable published work reaches 75-85% at 1-5 FA/h. This is **not clinically usable**,
since monitoring wants under 1 FA/hour.

**[Full report](report.html)** covers the method, all negative results, and the
measurement pitfalls.

## Run it

```bash
pip install -r requirements.txt

python edf_to_npz.py --fetch-missing     # download CHB-MIT, convert to .npz
python train_memory_efficient.py --folds 3 --epochs 15 --max-segments 2000 --balance none
python evaluate_end_to_end.py --target-fa 2.0
python score_szcore.py --cache endtoend_probs.npz
```

Needs a CUDA GPU and about 15 GB RAM. Training takes ~40 min, evaluation ~25 min.

## The label bug

The annotation parser took the first number on each line:

```python
re.search(r'(\d+)', line)   # "Seizure 1 Start Time: 1724 seconds" captures "1"
```

Every seizure was stored as starting *and* ending at second 1, so nothing was labelled.
**18 of 24 patients lost their entire ground truth**, and the six that parsed correctly
hid the problem. Fixing it moved AUC from **0.32 to 0.78**, more than every other
experiment combined. See `repair_labels.py`.

## What didn't work

Ten interventions were tried. Three helped: label repair, temporal smoothing (roughly
3x fewer false alarms), and selective per-patient personalisation (+0.13 to +0.22
sensitivity).

Seven did nothing or made things worse: SMOTE, weighted sampling, 2.7x more background,
52% more seizures, training class prior across an 8.6x range, 2.8x model capacity
*(worse)*, and rolling threshold recalibration *(worse)*.

Class imbalance was never the real problem. The data only *looked* imbalanced because
the bug had erased the positive labels. The true rate is 1.363%, which is simply what
scalp EEG is.

## Traps worth knowing

- **Counting alarm runs is gameable.** Flag everything and the runs merge into one that
  overlaps a real seizure, booking almost no false alarms at 100% sensitivity while
  being useless.
- **AUC hides end-to-end damage.** A same-subject leak moved AUC by 0.013 and event
  sensitivity by 0.084. Cases `chb01` and `chb21` are the same child, 18 months apart.
- **Patient grouping moves the headline by 8 points** at 3 folds. Anything finer needs
  leave-one-patient-out.

## Files

| | |
|---|---|
| `evaluate_end_to_end.py` | the honest pipeline, nothing selected using test data |
| `score_szcore.py` | SzCORE and strict scoring, side by side |
| `repair_labels.py` | rebuilds labels from corrected annotations |
| `edf_to_npz.py` | EDF to segmented `.npz`, skipping the CSV intermediate |
| `train_memory_efficient.py` | patient-level K-fold CV, bounded memory |

## Next

The remaining lever is longer temporal context, or a pre-trained backbone. CBraMod
(4.9M params, 9,000 h of pretraining) is verified to load and accept this montage
natively. But no foundation model has ever published an *event-level* CHB-MIT result,
and independent benchmarks find they often match their own random initialisation on
seizure tasks. Resolving that needs leave-one-patient-out evaluation, not 3 folds.

## Data

Not included, since it runs to 205 GB. CHB-MIT is public on PhysioNet, and
`edf_to_npz.py --fetch-missing` downloads the seizure-bearing recordings and converts
them.

## References

Song et al., *EEG Conformer*, IEEE TNSRE 2022. Shoeb, *CHB-MIT*, MIT 2009.
Dan et al., *SzCORE*, 2024.

MIT License
