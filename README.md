# EEG-Conformer — Patient-Independent Seizure Detection

Seizure detection on the [CHB-MIT Scalp EEG Database](https://physionet.org/content/chbmit/1.0.0/),
evaluated patient-independently at the event level.

## Result

| | |
|---|---|
| **Event sensitivity** | **83%** (140 of 168 events) |
| **False alarms** | **4.4 / hour** |
| Evaluated on | 170 h held-out, 24 patients, 166 seizures |
| Cross-validation | AUC 0.7795 ± 0.0168 (patient-level 3-fold) |

Scored with [SzCORE](https://epilepsybenchmarks.com) event rules (90 s merging, 30/60 s
tolerance). Under a stricter in-house scorer the identical predictions give **78% at
14.9 FA/h** — the scorer alone moves false alarms by 3.4×, so always name it.

5 of 24 patients are personalised on one of their *own earlier* seizures and evaluated
only on later ones. Splits are by patient and ordered in time; thresholds are chosen on
adaptation data, never on the test period.

**Full write-up, including the negative results:** [`report.html`](report.html)

### Honest positioning

| System | Sensitivity | False alarms |
|---|---|---|
| 2025 SzCORE challenge winner *(private, cross-hospital)* | 37% | 1.3 / day |
| Published CHB-MIT, leave-one-patient-out | 75–85% | 1–5 / hour |
| **This work** | **83%** | **4.4 / hour** |
| Rule-based baseline (Gotman, 1982) | 91% | ~10 / hour |

High sensitivity is cheap — a 1982 rule-based detector reaches 91% if you accept enough
false alarms. Suppressing false alarms is the difficulty, and it is not solved here.
This is **not clinically usable**: continuous monitoring wants < 1 FA/hour.

> Earlier versions of this README claimed 95–99% accuracy. Those figures come from
> *patient-specific* models trained and tested on the same person, which is a
> substantially easier problem than the one this repository addresses.

## Quick start

```bash
pip install -r requirements.txt

# 1. download CHB-MIT and convert straight to segmented .npz
python edf_to_npz.py --verify         # confirm converter reproduces the pipeline
python edf_to_npz.py --fetch-missing  # download seizure-bearing files, convert

# 2. train, patient-level 3-fold CV
python train_memory_efficient.py --folds 3 --epochs 15 --max-segments 2000 --balance none

# 3. the honest end-to-end number
python evaluate_end_to_end.py --target-fa 2.0
python score_szcore.py --cache endtoend_probs.npz
```

`--max-segments` bounds RAM: each segment is ~94 KB, so 24 patients × 10,000 needs
~15 GB. Every seizure is kept regardless; only background is subsampled.

## What's here

**Pipeline**

| File | Purpose |
|---|---|
| `edf_to_npz.py` | EDF → segmented `.npz` in one pass, skipping the CSV intermediate |
| `repair_labels.py` | Recomputes labels from corrected annotations (see *The label bug*) |
| `train_memory_efficient.py` | Patient-level K-fold CV, bounded memory |
| `model.py`, `config.py` | EEG-Conformer, 120,962 parameters |

**Evaluation** — the part worth reusing

| File | Purpose |
|---|---|
| `evaluate_end_to_end.py` | Full pipeline with nothing selected using test data |
| `evaluate_temporal.py` | Event-level scoring, temporal smoothing |
| `score_szcore.py` | SzCORE event rules vs. strict scoring, side by side |
| `calibrate_per_patient.py` | Unsupervised per-patient thresholds |
| `finetune_per_patient.py` | Personalisation with a strict chronological split |
| `selective_personalise.py` | Personalise only patients the base model fails |

**Experiments**

`train_natural_prior.py` (class-prior ablation, memmap-backed) ·
`build_memmap.py` (full dataset as one float16 memmap) ·
`compare_finetune.py`, `sweep_operating_points.py`, `compare_operating_points.py`

## The label bug

The original annotation parser took the first number on each line:

```python
match = re.search(r'(\d+)', line)   # "Seizure 1 Start Time: 1724 seconds"
                                    # captures "1", the index — not 1724
```

Every seizure was stored as starting and ending at second 1, so no samples were labelled.
**18 of 24 patients lost their entire ground truth.** Files with a single unnumbered
seizure parsed correctly, which is why six patients looked fine and hid the problem.

Fixing it took the dataset from 564 to 5,408 seizure segments and moved AUC from
**0.32 to 0.78** — more than every subsequent experiment combined.

`repair_labels.py` verifies against known-good patients before writing anything.

## What did not work

Ten interventions; three helped.

| | Intervention | Effect |
|---|---|---|
| ✅ | Repair the seizure labels | AUC 0.32 → 0.78 |
| ✅ | Temporal smoothing at evaluation | ~3× fewer false alarms, same sensitivity |
| ✅ | Selective per-patient personalisation | +0.13 to +0.22 event sensitivity |
| ⚪ | Synthetic oversampling (SMOTE) | none — removing it was marginally better |
| ⚪ | Weighted batch sampling | none |
| ⚪ | 2.7× more background EEG | none |
| ⚪ | +52% more seizures | none |
| ⚪ | Training class prior (12% / 3.6% / 1.4%) | none across an 8.6× range |
| ❌ | 2.8× model capacity | AUC 0.79 → 0.76 (overfitting) |
| ❌ | Rolling threshold recalibration | worse on both axes |

The class imbalance was never the real problem. The data only *appeared* six times more
imbalanced than it is, because the bug had erased most positive labels. The true rate is
**1.363%**, which is simply what scalp EEG is.

## Measurement pitfalls

Each of these produced a confidently wrong number before being caught.

**Counting alarm runs is gameable.** A threshold low enough to flag everything merges into
one run per recording, which overlaps a real seizure and books ~0 false alarms at 100%
sensitivity. Guard with a flagged-time bound (`max_flagged` in `calibrate_per_patient.py`).

**AUC cannot see what breaks end-to-end.** Cases `chb01` and `chb21` are the same child
recorded 18 months apart. Splitting them across folds moved AUC by 0.013 — and event
sensitivity by **0.084**.

**Patient grouping moves the headline ±8 points.** One fold reshuffle swung CHB06 from
33% to 100% and CHB14 from 43% to 14%. At 3 folds, anything finer than ±0.08 is noise;
leave-one-patient-out is needed to resolve smaller effects.

**Name your scorer.** 14.9 FA/h strict vs 4.4 FA/h under SzCORE, identical predictions.

## Limitations

- Not clinically usable (4.4 FA/h vs the < 1 wanted for monitoring)
- Personalisation requires one labelled seizure per patient; without it, the base model applies and performance is lower
- One hospital, 24 paediatric patients, no cross-institution validation
- Bimodal by patient — about a third reach 100% sensitivity at under ~1 FA/h while a handful produce most false alarms. The average describes almost nobody.

## Data

Not included (205 GB). CHB-MIT is public on PhysioNet; `edf_to_npz.py --fetch-missing`
downloads the seizure-bearing recordings and converts them directly.

## References

- Song et al., *EEG Conformer: Convolutional Transformer for EEG Decoding and Visualization*, IEEE TNSRE 2022
- Shoeb, *Application of Machine Learning to Epileptic Seizure Onset Detection*, MIT 2009 (CHB-MIT)
- Dan et al., *SzCORE: A Seizure Community Open-source Research Evaluation framework*, 2024

## License

MIT
