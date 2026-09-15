# Patient-Independent Seizure Detection on CHB-MIT

Detecting epileptic seizures from scalp EEG, tested only on patients the model has never
seen. Every split is by patient and ordered in time. Two models are compared: a
121k-parameter EEG-Conformer trained from scratch, and CBraMod, a pretrained foundation
model.

## Result

| | |
|---|---|
| **Event sensitivity** | **94%** (158 of 168 events) |
| **False alarms** | **5.2 per hour** |
| Evaluated on | 170 h held-out, 24 cases, 166 seizures |
| Cross-validation | leave-one-patient-out, 23 folds |

Scored with [SzCORE](https://epilepsybenchmarks.com) event rules (90 s merging, 30/60 s
tolerance around each seizure). Under a stricter in-house scorer the identical predictions
give **89% at 14.4 FA/h** -- the scorer alone moves false alarms by 2.8x, so it must always
be named.

Both numbers come from leave-one-patient-out. The earlier 3-fold split, which trains each
model on 16 patients rather than 23, gives **83% at 4.4 FA/h** on the same evaluation
harness. Under the strict scorer that split is worse on *both* axes -- 78% at 14.9 FA/h
against 89% at 14.4 -- so this is a real gain, not a shifted operating point. See
[More patients, not more data](#more-patients-not-more-data).

Adding a second hospital's 14 patients trades in the other direction: **34% fewer false
alarms for about two points of sensitivity**. Which operating point is preferable depends
on the use, so both are reported rather than only the flattering one. See
[A second hospital buys false alarms, not sensitivity](#a-second-hospital-buys-false-alarms-not-sensitivity).

5 of 24 patients are personalised on one of their *own earlier* seizures and evaluated
only on later ones. Thresholds are chosen from adaptation data, never from the test period.

**[Full report](report.html)** covers the method, every negative result, and the pitfalls.

### How this compares

| System | Sensitivity | False alarms |
|---|---|---|
| 2025 SzCORE challenge winner *(private, cross-hospital)* | 37% | 1.3 / day |
| Published CHB-MIT, leave-one-patient-out | 75-85% | 1-5 / hour |
| **This work** | **94%** | **5.2 / hour** |
| Rule-based baseline (Gotman, 1982) | 91% | ~10 / hour |

The Gotman row is the useful calibration: **high sensitivity is cheap.** A rule-based
detector from 1982 reaches 91% if you tolerate enough false alarms. Suppressing false
alarms is the entire difficulty, and it is not solved here. This is **not clinically
usable**, which needs under 1 FA/hour.

## More patients, not more data

Every attempt to give the model *more* of the same thing failed: 2.7x more background EEG,
52% more seizures, synthetic oversampling, weighted sampling. All landed on the same AUC
plateau.

Giving it more *patients* is the one axis that moved. Leave-one-patient-out trains each of
23 models on 22 subjects instead of the 3-fold split's 16, changing nothing else:

| | 3-fold (16 train) | LOPO (22 train) |
|---|---|---|
| Event sensitivity (strict) | 0.783 | **0.886** |
| False alarms/h (strict) | 14.93 | **14.42** |
| Event sensitivity (SzCORE) | 0.83 | **0.94** |

Better on both axes under the strict scorer, so this is not a threshold trade. The same
architecture, the same data per patient, the same seed -- six more training subjects.

That reverses the earlier conclusion that the dataset was saturated. It was saturated in
*hours*, not in *people*.

### What 23 folds reveal that 3 folds hide

Per-patient AUC across the 23 folds: **mean 0.803, sd 0.174**, from **0.428 (CHB13)** to
**0.992 (CHB22)**. The 3-fold run reported +/- 0.017. That figure was the spread of three
*averages of eight patients each*; the real per-patient spread is **ten times** larger, and
every ablation in this project was measured against the small number.

Eight of 23 subjects clear AUC 0.9. Four fall below 0.6.

The average describes almost nobody, and two patients show why window-level AUC is the
wrong measure:

- **CHB13** scores 0.428 AUC -- worse than chance at ranking windows -- yet catches
  **11 of 11 seizures** at event level once personalised. Its windows are ordered badly;
  its seizures are still found.
- **CHB09** catches **3 of 3 with zero false alarms** across its entire recording.

A model can be below chance on the metric the literature reports and still find every
seizure in the patient.

## A second hospital buys false alarms, not sensitivity

Siena Scalp EEG adds 14 adult patients from an Italian hospital, taking training from 22
subjects to 36. Both arms hold out the same 23 CHB-MIT subjects one at a time and are
scored on those; Siena patients are pinned into every training set and never held out, so
the arms differ in exactly one thing.

| Training set | Ch | Event sens | FA/h | SzCORE sens |
|---|---|---|---|---|
| CHB-MIT, 22 subjects | 23 | **0.886** | 14.42 | **0.940** |
| CHB-MIT + Siena, 36 subjects | 20 | 0.861 | 11.90 | 0.875 |
| CHB-MIT + Siena, 36 subjects | 23 | 0.831 | **11.83** | 0.881 |

The two effects do **not** compound. Restoring the three channels *and* adding Siena is
worse than adding Siena alone at 20 channels. The CHB-MIT-only model still has the best
sensitivity.

What Siena does buy is the axis that matters clinically:

```
median false alarms/h   11.27 -> 7.42     34% fewer
mean   false alarms/h   14.15 -> 11.98
better on 17 of 24 patients
```

Individually large in places: CHB15 32.11 to 0.94, CHB16 14.18 to 1.49, CHB12 23.85 to
6.88. Sensitivity is cheap and false alarms are the unsolved problem -- a rule-based
detector from 1982 reaches 91% sensitivity if you tolerate enough noise -- so a third
fewer false alarms is not a small thing.

**The sensitivity loss is one patient.** 21 of 24 patients are unchanged or better.
CHB06 alone falls from 1.00 to 0.33, accounting for 6 of the 9 lost events, and CHB06 is
the patient documented below as swinging between 33% and 100% on nothing more than a fold
reshuffle. Excluding it, the difference is **0.879 to 0.861** rather than 0.886 to 0.831.

Honest reading: a trade of roughly two points of sensitivity for a third fewer false
alarms, with the headline distorted by the least stable patient in the dataset. Not the
clean win predicted before it was run.

### Siena needed its annotations rebuilt first

The published seizure lists are hand-written free text and inconsistent in ways that break
a naive parser: `PNO6-1.edf` with a letter O for a zero, `PN11-.edf` with no index,
`16:13.23` with a stray colon, `11.41.04 opure 11.40.43`, recordings crossing midnight,
and one seizure ending an hour after its own recording stops. `parse_siena.py` computes
offsets against the EDF *header* instead, which is machine-written; two recordings whose
text start times disagree with their headers by 3 and 10 hours resolve correctly that way.

The original download script also guessed filenames from a pattern, but Siena names a
multi-seizure recording after every seizure it contains (`PN10-4.5.6.edf`). Eleven
recordings 404'd silently and two whole patients were missing from its list. Recovering
them took the usable set from 12 patients and 32 seizures to **14 patients and 47
seizures**.

`siena_to_npz.py` rebuilds CHB-MIT's bipolar montage by differencing Siena's unipolar
electrodes, verified exact (`max|delta| 0.00e+00`, including CHB-MIT's own internal
redundancies). It also refuses to convert a partially downloaded EDF: MNE reads a
truncated file without complaint by inferring length from file size, which silently
produces a shorter recording missing the seizures at the end.

## AUC got three decisions backwards

Siena has no FT9 or FT10 electrode, so three CHB-MIT derivations -- `T7-FT9`,
`FT9-FT10`, `FT10-T8` -- cannot be rebuilt from its montage. Merging the cohorts meant
first deciding whether those channels could be dropped.

Measured on window-level AUC, paired on identical folds and seeds, dropping them looked
free: **+0.014** over 9 paired 3-fold runs, and **+0.026** over 23 leave-one-patient-out
folds. Two independent measurements, both slightly favouring the smaller montage.

Both were wrong. Scored on events, which is what a detector is for:

| Training set | Channels | AUC | Event sens | FA/h | SzCORE sens |
|---|---|---|---|---|---|
| CHB-MIT, 22 subjects | 23 | 0.8030 | **0.886** | 14.42 | **0.940** |
| CHB-MIT, 22 subjects | 20 | 0.8289 | 0.777 | 13.92 | 0.810 |
| CHB-MIT + Siena, 36 subjects | 20 | 0.7997 | 0.861 | 11.90 | 0.875 |
| CHB-MIT + Siena, 36 subjects | 23 | -- | 0.831 | 11.83 | 0.881 |

Dropping the three channels cost **0.109 event sensitivity**. AUC rated the same change
+0.026, in the opposite direction.

Then the second decision, on the same table. Adding 14 Siena patients moved AUC by
**-0.029** -- read alone, a reason to abandon the merge. At event level it is **+0.084
sensitivity and 2.0 fewer false alarms per hour**, better on both axes at once.

**AUC was wrong about both changes, in opposite directions.** It is a ranking statistic
over windows in isolation; whether a seizure is caught depends on whether *any* window in
it clears threshold, and whether an alarm is false depends on runs of neighbouring
windows. Neither survives being averaged into a rank.

The per-patient detail is sharper still. The four patients whose AUC fell furthest when
Siena was added were CHB14, CHB13, CHB16 and CHB12, dropping 0.13 to 0.21 to land near
chance. At event level over the same predictions:

| patient | AUC change | event sensitivity |
|---|---|---|
| CHB12 | -0.134 | 0.38 to **0.74** |
| CHB13 | -0.189 | 1.00 to 1.00 |
| CHB14 | -0.211 | 1.00 to 1.00 |
| CHB16 | -0.162 | 0.22 to **0.33** |

Every one of them held or improved on the metric that matters, while AUC said they had
collapsed.

### The third: a foundation model that looked neutral

CBraMod was compared against the from-scratch model on identical folds and patients. On
window-level AUC the two were separated by **+0.001**, which this repo read as "no
benefit". Scored on events, on exactly the same folds:

| | Parameters | Event sens | FA/h | SzCORE sens |
|---|---|---|---|---|
| EEG-Conformer | 121k | **0.927** (152/164) | **15.82** | **0.940** |
| CBraMod | 4.9M | 0.823 (135/164) | 16.97 | 0.843 |

Worse on both axes: **17 fewer seizures caught** and more false alarms. Per patient it is
worse on 5, equal on 13, better on 4, with severe individual losses -- CHB16 from 0.67 to
0.11, CHB12 (39 events, the largest) from 0.92 to 0.59.

"No benefit" was too generous. 40x the parameters and 9,000 h of pretraining make it
**substantially worse** at the task, and AUC reported the gap as one thousandth of a point.

### Why AUC fails here specifically

It ranks windows in isolation. Whether a seizure is caught depends on whether *any* window
inside it clears threshold; whether an alarm is false depends on runs of neighbouring
windows. Neither property survives being averaged into a rank, and both are what a
detector is judged on.

**Consequence for this repo:** earlier versions of this file reported the channel drop as
free and CBraMod as neutral, both on AUC evidence gathered before event-level scoring was
run. Both claims were wrong and are corrected above. Every headline number in this
repository is now event-level.

## Class imbalance is not the lever, and neither is picking better negatives

Seizures are 1.363% of recorded time, so rebalancing the training set is the first thing
anyone suggests. Six variants have now been tested; none work, and one is actively harmful.

### Rebalancing the ratio

Three models trained at 12%, 3.6% and 1.4% seizure on identical folds, an 8.6x range:

| training balance | event sens | FA/h | p99 of output on pure background |
|---|---|---|---|
| 12% | 0.831 | **11.82** | 0.168 |
| 3.6% | **0.861** | 13.69 | 0.021 |
| 1.4% (the true rate) | 0.861 | 14.13 | 0.0039 |

No trend. The best sensitivity and the best false-alarm rate come from different settings,
and the true rate is worst on both.

The last column is why. Training at the natural prior makes the model **40x less
overconfident on background** -- a large, real change. It buys nothing, because the
per-patient calibration simply selects a 40x lower threshold and absorbs it. Thresholding
is rank-based; rebalancing moves confidence, not ranking.

**This generalises.** Any intervention that shifts the model's confidence uniformly is
invisible to this pipeline by construction. Only changes that reorder which windows score
above which can survive calibration. That single fact explains most of the null results in
the table above.

### Selecting hard negatives, which is worse

Rather than trimming background at random, keep the background windows the model actually
fires on and drop the easy ones -- same volume, same class ratio, only *which* negatives
changes. It targets false alarms directly rather than the ratio, so it should have escaped
the calibration argument.

| | event sens | FA/h | window AUC |
|---|---|---|---|
| uniform background | 0.831 | 11.82 | 0.78 |
| 50% hardest background | 0.789 | **21.40** | **0.44** |

Nearly double the false alarms, and window-level AUC below chance.

The mined negatives cluster around seizures:

| | within 1 min of a seizure | within 5 min |
|---|---|---|
| 500 hardest background windows | **13.5%** | **32.5%** |
| 500 random background windows | 3.1% | 15.1% |

Four times the concentration. CHB-MIT annotates the *clinical* seizure, but electrographic
onset precedes it and post-ictal activity follows, so windows adjacent to a seizure look
like seizures because they partly are. Half the training budget went on teaching the model
that seizure-like activity is definitely not a seizure, and it learned exactly that.

**Hard-negative mining assumes clean labels.** Seizure boundaries are a clinical judgement,
not a signal transition, so any method that preferentially selects confusing negatives
preferentially selects mislabelled ones. Random trimming is harmless and useless; selective
trimming is harmful.

## Per-patient thresholds are worth more than any model change except the label fix

The decision threshold is fitted per patient, on that patient's own earlier recording,
never on the test period. Measured against a single global threshold over identical
predictions:

| | event sens | FA/h |
|---|---|---|
| per-patient thresholds | **0.886** | 14.42 |
| best global threshold | 0.747 | 9.96 |

**+0.139 sensitivity**, 23 more seizures out of 166 -- and no global threshold reaches
0.886 at any setting without flagging so much of the recording that the detector stops
detecting.

The calibrated thresholds span **0.036 to 0.998** across 24 patients. Some patients need
the model 28x more certain before alarming. No single number serves both ends.

The cost is real: a patient with no prior recording gets the default and does worse, and
the scheme normalises away exactly the kind of model improvement described above.

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

# 2. train, leave-one-patient-out (23 folds, resumable)
python train_lopo.py --epochs 15 --max-segments 2000

# ...or the quick 3-fold version for iterating, ~40 min
python train_memory_efficient.py --folds 3 --epochs 15 --max-segments 2000 --balance none

# 3. optional: add the Siena cohort (14 more patients)
python parse_siena.py                       # rebuild annotations from EDF headers
python siena_to_npz.py --approximate-ft     # convert to CHB-MIT's montage
python train_lopo.py --tag merged --data-dirs preprocessed_data preprocessed_data_siena23     --always-train PN00 PN01 PN03 PN05 PN06 PN07 PN09 PN10 PN11 PN12 PN13 PN14 PN16 PN17

# 4. the honest end-to-end number
python evaluate_end_to_end.py --target-fa 2.0
python score_szcore.py --cache endtoend_probs.npz
```

CUDA GPU and about 15 GB RAM. 3-fold training ~40 min, LOPO ~9 h, evaluation ~25 min.

`--max-segments` bounds memory: each segment is ~94 KB, so 24 patients x 10,000 needs
~15 GB. Every seizure is kept regardless; only background is subsampled.

## What's here

**Pipeline**

| File | Purpose |
|---|---|
| `edf_to_npz.py` | EDF to segmented `.npz` in one pass, skipping the CSV intermediate |
| `repair_labels.py` | Recomputes labels from corrected annotations (see below) |
| `train_memory_efficient.py` | Patient-level K-fold CV, bounded memory |
| `train_lopo.py` | Leave-one-patient-out, 23 folds, resumable |
| `parse_siena.py` | Siena annotations from EDF headers, not the hand-written text |
| `siena_to_npz.py` | Siena to CHB-MIT montage, with a truncated-download guard |
| `check_domain_shift.py` | How separable the two cohorts are, per channel |
| `train_hard_negatives.py` | Hard-negative mining, with a leakage check on the mining model |
| `train_natural_prior.py` | Training at the true 1.4% class prior |
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

**Experiments** -- `build_memmap.py` (full dataset as one float16 memmap). Result files are indexed in
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

Sixteen interventions; five helped, one traded.

| | Intervention | Effect |
|---|---|---|
| yes | Repair the seizure labels | AUC 0.32 to 0.78 |
| yes | Temporal smoothing at evaluation | ~3x fewer false alarms, same sensitivity |
| yes | Selective per-patient personalisation | +0.13 to +0.22 event sensitivity |
| yes | Six more training patients (LOPO) | +0.10 event sensitivity, fewer false alarms |
| worse | Dropping 3 of 23 channels (FT9/FT10) | -0.109 event sensitivity, though AUC said +0.026 |
| mixed | 14 more patients from a second hospital (Siena) | 34% fewer false alarms, ~2 points less sensitivity |
| no | Synthetic oversampling (SMOTE) | none; removing it was marginally better |
| no | Weighted batch sampling | none |
| no | 2.7x more background EEG | none |
| no | +52% more seizures (68 downloaded) | none |
| no | Training class prior (12% / 3.6% / 1.4%) | none across an 8.6x range, on events as well as AUC |
| worse | Hard-negative mining (50% hardest background) | false alarms 11.8 to 21.4 per hour |
| yes | Per-patient decision thresholds | +0.139 event sensitivity over any global threshold |
| worse | Pretrained foundation model (CBraMod) | -0.104 event sensitivity, more false alarms |
| worse | 2.8x model capacity | AUC 0.79 to 0.76 (overfitting) |
| worse | Rolling threshold recalibration | worse on both sensitivity and false alarms |

Everything in the middle group landed on AUC 0.76-0.79 regardless of setting. That plateau
is a property of this architecture reading raw waveforms, not of any hyperparameter.

**The pattern.** Everything that failed added more *data*. The two things that worked
added more *patients* (LOPO) or more *information about the patient being tested*
(personalisation). Volume was never the binding constraint; subject diversity was.

**On the imbalance.** It was never the real problem. The data appeared six times more
imbalanced than it was, because the bug had erased most positive labels. The true rate is
1.363%, which is simply what scalp EEG is. Every method aimed at rebalancing it --
synthetic data included -- changed nothing.

## A pretrained foundation model makes it worse

CBraMod (ICLR 2025, 4.9M parameters, pretrained on 9,000 h of TUEG hospital EEG) was
fine-tuned and compared against the 121k-parameter from-scratch model. The comparison is
**paired**: identical folds, patients, seed, windows and labels, with only the
architecture differing. Both trainers were checked to produce the same three splits before
anything was run.

On window-level AUC the two are indistinguishable:

| fold | EEG-Conformer | CBraMod | delta |
|---|---|---|---|
| 1 | 0.8353 | 0.8682 | +0.0329 |
| 2 | 0.7630 | 0.7354 | -0.0276 |
| 3 | 0.7939 | 0.7919 | -0.0020 |
| **mean** | **0.7974 +/- 0.0296** | **0.7985 +/- 0.0544** | **+0.0011** |

Scored on events, through the same honest pipeline, the same models are not close:

| | Event sens | FA/h | SzCORE sens | SzCORE FA/h |
|---|---|---|---|---|
| EEG-Conformer, 121k | **0.927** (152/164) | **15.82** | **0.940** | **5.28** |
| CBraMod, 4.9M | 0.823 (135/164) | 16.97 | 0.843 | 5.38 |

**17 fewer seizures caught, and more false alarms.** Worse on 5 patients, equal on 13,
better on 4; the losses are severe where they occur -- CHB16 0.67 to 0.11, CHB02 and CHB11
halving, CHB12 (39 events) 0.92 to 0.59.

40x the parameters and 9,000 hours of pretraining bought a **worse detector**, and AUC
priced the difference at one thousandth of a point.

This is consistent in direction with independent benchmarks (EEG-FM-Bench,
AdaBrain-Bench, NeuroAtlas), which find these models frequently match their own random
initialisation on seizure tasks. Those benchmarks score at *window* level, so they cannot
see the gap measured here. No published work reports event-level sensitivity and false
alarms per hour for any EEG foundation model on CHB-MIT.

The head is mean-pooled (~26k parameters) rather than CBraMod's published flatten head
(~14.7M, three times its own backbone), so the test measures pretrained features rather
than head capacity on 22 patients.

CHB03 and CHB05 are excluded from this comparison: their local copies predate `raw_data/`
and do not match the PhysioNet recordings, so they cannot be paired. 22 patients remain,
which is why the absolute numbers here are not comparable with the 24-patient rows
elsewhere in this file.

## Things that fooled the measurements

Each of these produced a confidently wrong number before being caught.

**Counting alarm runs is gameable.** False alarms are counted as contiguous runs of
positive predictions. A threshold low enough to flag *everything* collapses each recording
into one run, which overlaps a real seizure and therefore books almost no false alarms --
scoring ~0 FA/h at 100% sensitivity while being useless. Guard with a flagged-time bound
(`max_flagged` in `calibrate_per_patient.py`).

This is not a hypothetical. Comparing per-patient against global thresholds, a first pass
without the bound reported *1.000 sensitivity at 0.15 FA/h* from a global threshold of
0.010 -- a perfect score from a detector that alarms continuously. The trap was already
documented here and was walked into anyway, which is the argument for keeping the bound in
the pipeline rather than in the analyst's memory. For reference, the real system flags
8.5% of recorded time against a 1.8% seizure rate, and one patient (CHB12) flags 34%.

**AUC cannot see what breaks end to end.** Cases `chb01` and `chb21` are the same child,
and the splits put her on both sides. Fixing it moved AUC by **0.013** and event
sensitivity by **0.084**. Judged on AUC alone the leak looked harmless.

**Patient grouping moves the headline by 8 points.** One reshuffle of which patients sit
in which fold swung CHB06 from 33% to 100% and CHB14 from 43% to 14%, with no other
change. At 3 folds, anything finer than +/-0.08 is noise.

**Fold-averaged error bars hide the variance that matters.** The 3-fold run reported AUC
+/- 0.017, which reads as a tightly determined result. It is the spread of three averages
of eight patients; the per-patient spread measured by leave-one-patient-out is **0.174**,
ten times larger. Every negative result below was measured against the small number, so
effects under roughly 0.08 AUC were never resolvable at all.

**Absolute numbers do not transfer between runs.** The same architecture scores 0.783
event sensitivity on 24 patients over 3 folds, 0.927 on 22 patients over 3 folds, and
0.886 on 24 patients under leave-one-patient-out. Different patient sets, different test
hours, different fold assignments. That 14-point spread is larger than any effect measured
in this project, so every comparison here is **paired** -- same folds, same patients, one
variable changed -- and rows from different tables should not be read against each other.

**The scorer changes the answer by 3.4x.** Identical predictions score 14.9 FA/h strict
and 4.4 FA/h under SzCORE event rules. Published false-alarm rates are not comparable
unless the scorer is named.

## Limitations

- **Not clinically usable.** Continuous monitoring wants under 1 FA/hour; this is at 5.2.
- **Two patients dominate the false-alarm rate.** CHB12 and CHB13 have 34% and 25% of
  their recordings flagged, against a 1.8% seizure rate, and contribute a large share of
  the pooled 14 FA/h. The average describes almost nobody.
- **Personalisation is part of the method.** 5 of 24 patients were fine-tuned on one of
  their own earlier seizures, evaluated only on later ones. For a patient with no recorded
  seizure, the base model applies and performance is lower.
- **Cross-institution validation is one-directional.** Siena patients are only ever
  training data; the model is never scored on a held-out Siena patient, so nothing here
  says how it performs on adult Italian recordings.
- **The cohorts are trivially separable.** A classifier on band power alone tells CHB-MIT
  from Siena at AUC 0.996, so a merged model can use cohort identity as a feature. Gains
  come from diversity, not transfer.
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
