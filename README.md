# Patient-Independent Seizure Detection on CHB-MIT

Detecting epileptic seizures from scalp EEG, tested only on patients the model has never
seen. Every split is by patient and ordered in time. Two models are compared: a
121k-parameter EEG-Conformer trained from scratch, and CBraMod, a pretrained foundation
model.

## Result

| | |
|---|---|
| **Event sensitivity** | **83%** (140 of 168 events) |
| **False alarms** | **4.4 per hour** |
| Evaluated on | 170 h held-out, 24 cases, 166 seizures |
| Cross-validation | AUC 0.7795 +/- 0.0168 (patient-level, 3-fold) |

Scored with [SzCORE](https://epilepsybenchmarks.com) event rules (90 s merging, 30/60 s
tolerance around each seizure). Under a stricter in-house scorer the identical predictions
give **78% at 14.9 FA/h** -- the scorer alone moves false alarms by 3.4x, so it must always
be named.

5 of 24 patients are personalised on one of their *own earlier* seizures and evaluated
only on later ones. Thresholds are chosen from adaptation data, never from the test period.

**[Full report](report.html)** covers the method, every negative result, and the pitfalls.

### How this compares

| System | Sensitivity | False alarms |
|---|---|---|
| 2025 SzCORE challenge winner *(private, cross-hospital)* | 37% | 1.3 / day |
| Published CHB-MIT, leave-one-patient-out | 75-85% | 1-5 / hour |
| **This work** | **83%** | **4.4 / hour** |
| Rule-based baseline (Gotman, 1982) | 91% | ~10 / hour |

The Gotman row is the useful calibration: **high sensitivity is cheap.** A rule-based
detector from 1982 reaches 91% if you tolerate enough false alarms. Suppressing false
alarms is the entire difficulty, and it is not solved here. This is **not clinically
usable**, which needs under 1 FA/hour.

## The dataset

```
24 cases (chb01 ... chb24), but 23 subjects -- chb01 and chb21 are the same
   child recorded 18 months apart, per the PhysioNet description
152 recordings, 220 hours of EEG
396,663 windows of 4 s with 50% overlap
5,408 seizure windows = 1.363%, about 180 minutes of seizure in total
190 distinct seizure events
```

Seizures appear in every patient but are heavily skewed: from 0.157% of CHB06's recording
to 5.081% of CHB08's. The **top 3 patients hold 40%** of all seizure data, and the top 5
hold 52%. That skew, not the 1.363% average, is what drives the variance in the results.

## Quick start

```bash
pip install -r requirements.txt

# 1. download CHB-MIT and convert straight to segmented .npz
python edf_to_npz.py --verify          # confirm the converter reproduces the pipeline
python edf_to_npz.py --fetch-missing   # download seizure-bearing files, convert

# 2. train, patient-level 3-fold CV
python train_memory_efficient.py --folds 3 --epochs 15 --max-segments 2000 --balance none

# 3. the honest end-to-end number
python evaluate_end_to_end.py --target-fa 2.0
python score_szcore.py --cache endtoend_probs.npz
```

CUDA GPU and about 15 GB RAM. Training ~40 min, evaluation ~25 min.

`--max-segments` bounds memory: each segment is ~94 KB, so 24 patients x 10,000 needs
~15 GB. Every seizure is kept regardless; only background is subsampled.

## What's here

**Pipeline**

| File | Purpose |
|---|---|
| `edf_to_npz.py` | EDF to segmented `.npz` in one pass, skipping the CSV intermediate |
| `repair_labels.py` | Recomputes labels from corrected annotations (see below) |
| `train_memory_efficient.py` | Patient-level K-fold CV, bounded memory |
| `model.py`, `config.py` | EEG-Conformer, 120,962 parameters |

**Evaluation** -- the part worth reusing

| File | Purpose |
|---|---|
| `evaluate_end_to_end.py` | Full pipeline with nothing selected using test data |
| `evaluate_temporal.py` | Event-level scoring and temporal smoothing |
| `score_szcore.py` | SzCORE event rules vs strict scoring, side by side |
| `calibrate_per_patient.py` | Unsupervised per-patient thresholds |
| `finetune_per_patient.py` | Personalisation with a strict chronological split |
| `selective_personalise.py` | Personalise only the patients the base model fails |

**Foundation-model comparison**

| File | Purpose |
|---|---|
| `train_cbramod.py` | Fine-tunes CBraMod on identical folds |
| `edf_to_cbramod.py` | Converts to 200 Hz / uV-100, verified to match the main pipeline |

**Experiments** -- `train_natural_prior.py` (class-prior ablation) and `build_memmap.py`
(full dataset as one float16 memmap). Result files are indexed in
[`cv_results/README.md`](cv_results/README.md).

## The label bug

The original annotation parser took the first number on each line:

```python
match = re.search(r'(\d+)', line)   # "Seizure 1 Start Time: 1724 seconds"
                                    # captures "1", the index -- not 1724
```

CHB-MIT writes single-seizure files as `Seizure Start Time:` and multi-seizure files as
`Seizure 1 Start Time:`. On the second form, the regex captured the seizure index. Every
such seizure was stored as starting *and* ending at second 1, so no samples were ever
labelled.

**18 of 24 patients lost their entire ground truth.** The six whose files happened to use
the unnumbered form parsed correctly, which is why the problem stayed hidden -- and why
the model still reported 99% accuracy while detecting nothing (99% of windows are
non-seizure).

Fixing it took the dataset from 564 to 5,408 seizure segments and moved AUC from
**0.32 to 0.78** -- more than every subsequent experiment combined.

`repair_labels.py` verifies against known-good patients before writing anything.

## What did not work

Eleven interventions; three helped.

| | Intervention | Effect |
|---|---|---|
| yes | Repair the seizure labels | AUC 0.32 to 0.78 |
| yes | Temporal smoothing at evaluation | ~3x fewer false alarms, same sensitivity |
| yes | Selective per-patient personalisation | +0.13 to +0.22 event sensitivity |
| no | Synthetic oversampling (SMOTE) | none; removing it was marginally better |
| no | Weighted batch sampling | none |
| no | 2.7x more background EEG | none |
| no | +52% more seizures (68 downloaded) | none |
| no | Training class prior (12% / 3.6% / 1.4%) | none across an 8.6x range |
| no | Pretrained foundation model (CBraMod) | +0.001 paired |
| worse | 2.8x model capacity | AUC 0.79 to 0.76 (overfitting) |
| worse | Rolling threshold recalibration | worse on both sensitivity and false alarms |

Everything in the middle group landed on AUC 0.76-0.79 regardless of setting. That plateau
is a property of this architecture reading raw waveforms, not of any hyperparameter.

**On the imbalance.** It was never the real problem. The data appeared six times more
imbalanced than it was, because the bug had erased most positive labels. The true rate is
1.363%, which is simply what scalp EEG is. Every method aimed at rebalancing it --
synthetic data included -- changed nothing.

## A pretrained foundation model does not help

CBraMod (ICLR 2025, 4.9M parameters, pretrained on 9,000 h of TUEG hospital EEG) was
fine-tuned and compared against the 121k-parameter from-scratch model. The comparison is
**paired**: identical folds, patients, seed, windows and labels, with only the
architecture differing.

| fold | EEG-Conformer | CBraMod | delta |
|---|---|---|---|
| 1 | 0.8353 | 0.8682 | +0.0329 |
| 2 | 0.7630 | 0.7354 | -0.0276 |
| 3 | 0.7939 | 0.7919 | -0.0020 |
| **mean** | **0.7974 +/- 0.0296** | **0.7985 +/- 0.0544** | **+0.0011** |

A mean paired difference of **+0.001**. The folds disagree in both directions and the
spread of differences (0.025) is larger than the difference itself, matching the same-fold
noise measured independently in the class-prior runs. CBraMod was also the less stable of
the two.

**40x the parameters and 9,000 hours of pretraining bought nothing.** This is consistent
with independent benchmarks (EEG-FM-Bench, AdaBrain-Bench, NeuroAtlas), which find these
models frequently match their own random initialisation on seizure tasks.

Note that those benchmarks score at *window* level. No published work reports event-level
sensitivity and false alarms per hour for any foundation model on CHB-MIT.

CHB03 and CHB05 are excluded from this comparison: their local copies predate `raw_data/`
and do not match the PhysioNet recordings (different labels, and a different window count
for CHB05), so they cannot be paired. 22 patients remain.

## Things that fooled the measurements

Each of these produced a confidently wrong number before being caught.

**Counting alarm runs is gameable.** False alarms are counted as contiguous runs of
positive predictions. A threshold low enough to flag *everything* collapses each recording
into one run, which overlaps a real seizure and therefore books almost no false alarms --
scoring ~0 FA/h at 100% sensitivity while being useless. Guard with a flagged-time bound
(`max_flagged` in `calibrate_per_patient.py`).

**AUC cannot see what breaks end to end.** Cases `chb01` and `chb21` are the same child,
and the splits put her on both sides. Fixing it moved AUC by **0.013** and event
sensitivity by **0.084**. Judged on AUC alone the leak looked harmless.

**Patient grouping moves the headline by 8 points.** One reshuffle of which patients sit
in which fold swung CHB06 from 33% to 100% and CHB14 from 43% to 14%, with no other
change. At 3 folds, anything finer than +/-0.08 is noise; leave-one-patient-out is needed
to resolve smaller effects.

**The scorer changes the answer by 3.4x.** Identical predictions score 14.9 FA/h strict
and 4.4 FA/h under SzCORE event rules. Published false-alarm rates are not comparable
unless the scorer is named.

## Limitations

- **Not clinically usable.** Continuous monitoring wants under 1 FA/hour; this is at 4.4.
- **Personalisation is part of the method.** 5 of 24 patients were fine-tuned on one of
  their own earlier seizures, evaluated only on later ones. For a patient with no recorded
  seizure, the base model applies and performance is lower.
- **One hospital, 23 subjects, all paediatric.** No cross-institution validation.
- **Three folds is too few** given the +/-8-point grouping variance.
- **Bimodal by patient.** Roughly a third reach 100% sensitivity at under ~1 FA/h; a
  handful produce most of the false alarms. The average describes almost nobody.

## Data

Not included (205 GB). CHB-MIT is public on PhysioNet, and `edf_to_npz.py --fetch-missing`
downloads the seizure-bearing recordings and converts them directly.

## References

- Song et al., *EEG Conformer: Convolutional Transformer for EEG Decoding and Visualization*, IEEE TNSRE 2022
- Wang et al., *CBraMod: A Criss-Cross Brain Foundation Model for EEG Decoding*, ICLR 2025
- Shoeb, *Application of Machine Learning to Epileptic Seizure Onset Detection*, MIT 2009 (CHB-MIT)
- Dan et al., *SzCORE: A Seizure Community Open-source Research Evaluation framework*, 2024

## License

MIT
