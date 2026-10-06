# Seizure Detection from Scalp EEG

Finds epileptic seizures in EEG recordings. Cross-patient trained, patient-calibrated:
no test patient is in the training set, but each patient's alarm threshold is set from
their own earlier recording.

**94% of seizures found, 5.2 false alarms per hour (124 per day)**, over 170 hours of
held-out recording. Pooled event sensitivity, SzCORE rules, `szcore_official.txt`.

That number is **patient-calibrated, not patient-independent.** Every patient's
threshold is picked from their own earlier recording, their own first-seizure labels
decide whether to personalise, and 5 of 24 cases are then fine-tuned on one of their
own earlier seizures.

With none of that -- threshold from the training-fold patients only, base model
throughout, no test-patient labels read anywhere -- the same models on the same windows
give **78% at 5.5 false alarms per hour (131 per day)**, in
`szcore_lopo_uncalibrated.txt`.

Good sensitivity. The false-alarm rate is not good enough -- clinical use needs under
1 per hour.

---

## 1. Data in

Raw EEG recordings (`.edf`) from three hospitals, plus human annotations of when
seizures happened.

| dataset | patients | hours | who | seizures |
|---|---|---|---|---|
| CHB-MIT | 23 | 220 | children, Boston | 1.4% of time |
| Siena | 14 | 142 | adults, Italy | 0.6% |
| Helsinki | 79 | 112 | newborns, Finland | 12.4% |

CHB-MIT has 24 recording cases but 23 subjects. chb01 and chb21 are the same child,
recorded 18 months apart. They are always held out together, so a split never puts one
person on both sides. Counts of 24 elsewhere in this file refer to cases.

EEG is a voltage trace per scalp electrode. A seizure is a burst of rhythmic activity
across several of them. Everything else -- chewing, moving, sleep -- is background, and
it is 99% of the recording.

## 2. What we do to it

| step | why |
|---|---|
| Convert to 23 channel pairs (`FP1-F7 = Fp1 - F7`) | differences cancel shared noise |
| Cut into 4-second windows, 50% overlap | fixed size for the model |
| Filter to 0.5-50 Hz | drop drift and mains hum |
| Rescale each channel to mean 0, sd 1 | different machines look comparable |

```
one recording  ->  (1799, 23, 1024) float32   signal
                   (1799,)          int64     0 or 1 per window
```

## 3. Model

**EEG-Conformer**, 120,962 parameters. Small on purpose -- a 4.9M pretrained model
did worse.

```
input   (23, 1024)            23 channels, 4 seconds
        conv over time        waveform shapes
        conv over channels    which electrodes agree
        transformer, 4 layers relates parts of the window
output  2 numbers             seizure probability
```

Trained by holding one patient out at a time, so no test patient is in any training
set. Test-patient data is used after training, though: each patient's alarm threshold
comes from their own earlier recording, their own first-seizure labels decide whether
to personalise, and in 5 of 24 cases the model is then fine-tuned on that patient.
Section 4 and Results say what that is worth.

## 4. Data out

A seizure probability per 4-second window, turned into alarms by three steps:

| step | effect |
|---|---|
| Smooth over neighbouring windows | ~3x fewer false alarms |
| Threshold set per patient, from their own earlier recording | +0.139 sensitivity |
| Require 3 windows in a row | removes single-window blips |

```
CHB09    3 seizures, 3 found,  0.0 false alarms/hour
CHB13   11 seizures, 11 found, 35.1 false alarms/hour
```

Those two rows are the strict scorer, from `szcore_lopo.txt`. Under SzCORE rules the same
predictions give CHB13 7.7 false alarms/hour.

Pooled: how many seizures caught, and how often it cried wolf.

## 5. Run it

```bash
pip install -r requirements.txt
python edf_to_npz.py --fetch-missing                    # download + convert
python train_lopo.py --epochs 15 --max-segments 2000    # ~13 h on a GPU
python evaluate_end_to_end.py --target-fa 2.0           # ~25 min
python score_szcore.py --cache endtoend_probs.npz
```

Other cohorts:

```bash
python parse_siena.py && python siena_to_npz.py --approximate-ft --out preprocessed_data_siena23
python helsinki_to_npz.py --channels 20
```

Needs a CUDA GPU and ~15 GB RAM.

## Results

| system | sensitivity | false alarms |
|---|---|---|
| 2025 SzCORE challenge winner *(private data)* | 37% | 1.3 / day |
| Published CHB-MIT results | 75-85% | 1-5 / hour (24-120 / day) |
| **This work** *(patient-calibrated)* | **94%** | **5.2 / hour, 124 / day** |
| **This work** *(no test-patient data)* | **78%** | **5.5 / hour, 131 / day** |
| Rule-based detector (Gotman, 1982) | 91% | ~10 / hour (~240 / day) |

Scored with [SzCORE](https://epilepsybenchmarks.com) rules, using the official
`timescoring` package. It reproduces `score_szcore.py` exactly on these predictions, so
the re-implementation is sound. A stricter scorer gives **89% at 14.4/hour** on
identical predictions, so the scorer must always be named.

Both rows above are pooled -- every event counted once, so patients with many seizures
weigh more. SzCORE's own convention averages per subject, which gives **97%** and
**86%** instead. Both are in the output files. The pooled number is quoted here because
it is the lower one and the one already published.

| | sensitivity | precision | F1 | FA/day |
|---|---|---|---|---|
| patient-calibrated, pooled | 0.940 | 0.152 | 0.262 | 123.8 |
| patient-calibrated, per subject | 0.970 | 0.202 | 0.308 | 118.9 |
| no test-patient data, pooled | 0.780 | 0.123 | 0.213 | 131.2 |
| no test-patient data, per subject | 0.864 | 0.263 | 0.325 | 122.5 |

**What calibration is worth, measured.** Pooled, per-patient thresholds plus
personalisation are worth **16.1 points of event sensitivity**, 78.0% to 94.0%, and the
uncalibrated run is worse on both axes -- lower sensitivity *and* more alarms, 131.2
against 123.8 per day. Per subject the picture reverses: sensitivity still rises, 86.4%
to 97.0%, and that gain is significant -- paired Wilcoxon p = 0.028 over the 24 cases.
Mean per-subject F1 falls 0.325 to 0.308, but that difference is **not** significant:
p = 0.790, the median difference is +0.010 the other way, and calibration is better in
14 of 24 cases. The negative mean comes from four cases where the training-fold
threshold nearly silences the detector, so CHB19 and CHB20 score precision 1.000 at
0.00 FA/h by barely firing at all. An earlier version of this file called that reversal
a finding. It is not one: read per subject, calibration buys sensitivity and the F1
effect is not measurable in either direction.

**What the calibrated row uses that the uncalibrated one does not.** A threshold from
each patient's own earlier recording; that patient's own first-seizure labels to decide
whether to personalise, which happens for all 24 cases and not only the 5; and
fine-tuning on one of their own earlier seizures for those 5. The uncalibrated row reads
no test-patient labels at all. Both score exactly the same windows: the test period
still starts after each patient's first seizure, which is the one thing the uncalibrated
run borrows, and it fixes where scoring begins rather than what the model or threshold
does.

### Sensitivity at a fixed alarm budget

**These are oracle numbers.** One global threshold is swept over the test predictions
and the best sensitivity inside each budget is kept, so the threshold is chosen knowing
the test result. No deployable system reaches them. They bound what the model's ranking
supports, nothing more.

| budget | patient-calibrated | no test-patient data |
|---|---|---|
| 1 alarm / day | 4.8% | 3.0% |
| 2 / day | 6.0% | 4.2% |
| 5 / day | 8.3% | 17.3% |
| 10 / day | 21.4% | 19.1% |

Pooled. Per-subject rows are in `szcore_official.txt` and
`szcore_lopo_uncalibrated.txt`.

### How this compares

As reported by those sources, not re-run here. SzCORE's own subject-independent CHB-MIT
baselines are **67.1% sensitivity at 2.09 false alarms/day** for XGBoost and **37.0% at
1.66/day** for a random forest. The 2025 SzCORE challenge winner reports **37% at 1.34
false alarms/day**, on different data. Protocols differ, so these are not like-for-like:
those are subject-independent, this work is patient-calibrated, and the challenge number
comes from a held-out hospital nobody could tune against.

The gap that matters is not sensitivity. It is the alarm rate. This work runs at roughly
**90 times** their false-alarm rate. At a comparable budget of 1-2 alarms/day it reaches
3-6%, even with an oracle threshold.

That last row is the calibration: **high sensitivity is cheap.** A 1982 rule-based
detector hits 91% if you tolerate enough noise. Suppressing false alarms is the real
problem, and it is not solved here.

## What worked, and how we know

Every comparison below is **paired**: same folds, same held-out patients, one variable
changed. All are scored on *events* (seizures caught, false alarms per hour), never on
window-level AUC, which got three of these calls backwards.

**One seed, except where stated.** Only the patient-count row has been repeated at more
than one seed, and when it was, the gain moved from +0.304 to a range of 0.03 to 0.31.
So read every other effect size here as a single draw. The nulls are the safer half of
the table -- an effect large enough to beat that much variance would probably have shown
up anyway -- but any single positive number could be smaller or larger than stated. The
CHB-MIT rows are less exposed than the Helsinki one, because leave-one-patient-out folds
are deterministic there (`train_lopo.py` holds out every subject in a fixed order), so a
seed changes only the initialisation and not which patients are tested.

| | what | effect | how we know |
|---|---|---|---|
| **yes** | Fixed a bug in the seizure labels | biggest single change | rerun the identical pipeline before and after the fix: AUC 0.32 to 0.78 |
| **yes** | Smoothing predictions over time | ~3x fewer false alarms | same saved predictions scored with and without smoothing |
| **yes** | A threshold per patient | +0.139 sensitivity | swept a single global threshold over identical predictions; no setting reaches 0.886 at all |
| **yes** | More patients *from the same cohort* | direction yes, size unclear: **+0.22 event sensitivity, range 0.03 to 0.31** across three seeds at a matched ~4.6 FA/h | 8 to 67 Helsinki training patients, three seeds. Positive in 14 of 15 budget-by-seed cells. The seed also picks the held-out babies, so each seed is a near-independent experiment on 12 infants with 158, 104 and 70 events |
| no | More patients from *other* hospitals | no reliable effect | two arms, same held-out patients, extra cohort pinned into training only. Siena: a trade. Helsinki: 0.789 to 0.813 sensitivity but 17.4 to 19.8 FA/h |
| no | Rebalancing the classes (6 ways) | nothing | three models trained at 12% / 3.6% / 1.4% seizure on identical folds: 0.831 / 0.861 / 0.861 sensitivity, no trend |
| no | A pretrained foundation model | worse | same folds and patients, only the architecture differs: 152 of 164 seizures against 135 of 164 |
| worse | Training on the "hard" background only | false alarms nearly doubled | 11.8 to 21.4 FA/h; the mined windows sit within a minute of a seizure 4x as often as random ones |

Two guards run inside the pipeline rather than living in anyone's memory, because both
mistakes were made at least once:

- **a flagged-time bound**, because a threshold low enough to alarm continuously scores
  100% sensitivity at almost no false alarms while being useless
- **a resume check on checkpoints**, because reusing a model trained with a different
  channel count silently answers the wrong question

**More patients helps, but only from the same cohort.** Going from 16 to 22 CHB-MIT
training subjects gained 10 points of sensitivity. Adding patients from *other* hospitals
did not reproduce that:

| added to training | held-out CHB-MIT sensitivity | FA/h |
|---|---|---|
| nothing (control) | 0.789 | 17.4 |
| 14 Siena adults | a trade: fewer false alarms, less sensitivity | |
| 79 Helsinki newborns | 0.813 | 19.8 |

Helsinki gains 2 points of sensitivity for 2.5 more false alarms per hour. At matched
false-alarm rates the difference changes sign depending where you look, which is what no
effect looks like.

**More data of the same kind does not help either.** More hours, more seizures, synthetic
seizures: all nothing.

**Why cross-hospital data may not transfer.** A classifier separates CHB-MIT from Helsinki
at AUC 0.9995 on band power alone -- children against newborns, nine times the seizure
rate, different equipment. Earlier versions of this file argued the diversity would help
regardless. The measurement does not support that.

## How far does patient count go?

The one thing that reliably helps is more patients from the same cohort. CHB-MIT could
not say how far that goes: it has 23 subjects, so 22 is the largest training set it can
ever provide, and two points do not show a curve.

Helsinki has 46 babies with seizures. Training on 8, 16, 32 and 67 patients against a
fixed held-out set of 12, scored on events under false-alarm budgets:

Each cell is sensitivity and **the false-alarm rate actually achieved**. A budget is a
ceiling, and two models under the same ceiling can sit at very different rates, so the
achieved rate has to be printed or the comparison cannot be checked.

| training patients | <= 2 FA/h | <= 5 FA/h | <= 10 FA/h | <= 20 FA/h | <= 40 FA/h |
|---|---|---|---|---|---|
| 8 | 0.025 @ 1.6 | 0.215 @ 4.6 | 0.392 @ 9.9 | 0.519 @ 18.9 | 0.684 @ 30.6 |
| 16 | 0.316 @ 1.6 | 0.361 @ 4.7 | 0.411 @ 9.6 | 0.532 @ 19.9 | 0.646 @ 29.8 |
| 32 | 0.354 @ 1.9 | 0.399 @ 4.3 | 0.437 @ 9.5 | 0.570 @ 19.3 | 0.715 @ 27.1 |
| **67** | **0.468 @ 1.1** | **0.519 @ 4.3** | **0.557 @ 7.9** | **0.652 @ 19.2** | **0.728 @ 26.0** |

Three things follow.

The table above is one seed. Repeating the whole curve at two more seeds shows the
direction is solid and **the magnitude is not**:

| budget | seed 42 | seed 43 | seed 44 | mean | range |
|---|---|---|---|---|---|
| <= 2 FA/h | +0.443 | **-0.019** | +0.286 | +0.237 | 0.462 |
| <= 5 FA/h | +0.304 | +0.029 | +0.314 | +0.216 | 0.285 |
| <= 10 FA/h | +0.165 | +0.058 | +0.314 | +0.179 | 0.257 |
| <= 20 FA/h | +0.133 | +0.087 | +0.343 | +0.187 | 0.256 |
| <= 40 FA/h | +0.044 | +0.096 | +0.057 | **+0.066** | **0.052** |

**More patients helps, and how much is not measurable here.** The gain from 8 to 67 is
positive in 14 of 15 cells, so the direction holds. But at the headline budget the three
seeds give +0.304, +0.029 and +0.314 -- a spread almost as wide as the effect -- and at
the tightest budget one seed is negative. The honest figure is **+0.22 with a range of
0.03 to 0.31**, not the +0.304 an earlier version of this file reported as a point
estimate.

Two things make it this noisy, and both are the design rather than the model. The seed
chooses the held-out babies as well as the initialisation (`helsinki_curve.py:162` passes
it to `split_patients`), so seeds 42 and 43 share only 3 of their 12 test babies. And the
three test sets contain 158, 104 and 70 events, so these are three small experiments
rather than three readings of one.

**The shape is better evidenced than the size.** That the benefit shrinks as the budget
loosens is the most robust thing in the table: at 40 FA/h the mean gain is +0.066 with a
spread of 0.052, against spreads near 0.26 everywhere tighter. More data buys progressively
less as a detector is allowed to alarm more freely, and that holds across every seed.

**It is still climbing at 67** in each seed taken alone, and the curve is monotonic in
training size for seed 42. No ceiling is visible, so the 23 subjects of CHB-MIT are the
shallow end of this curve rather than most of it -- though with this much seed variance,
"still climbing" is a direction and not a slope.

**An earlier version of this table was wrong, and the error was a measurement artifact.**
It swept 50 thresholds, which left the 8-patient model with no operating point between
4.31 and 10.80 FA/h, so its 5 and 10 FA/h cells were the same point read twice. The
8-to-67 gain was reported as +0.405 at 10 FA/h when it was really comparing 4.3 FA/h
against 8.2, and the 40 FA/h cell was reported as -0.006 while comparing 30.7 FA/h
against 22.7. The sweep now draws 600 thresholds from the data, every cell prints its
achieved rate, and the script flags any comparison whose rates differ by more than 25%.
The headline was overstated 2.5x; the shape of the curve was not.

Subsets are nested and the seizure-bearing to background-only mix is held constant, so the
curve measures adding patients rather than resampling them.

## One detector per population

Three separate models, each trained and scored only on its own cohort. Not a merge --
merging was tested and did not work.

With the full pipeline, where the data supports it:

| cohort | seizures caught | FA/h | FA/day | events |
|---|---|---|---|---|
| CHB-MIT, children | 94.0% | 5.2 | 124 | 168 |
| Siena, adults | 93.9% | 5.2 | 124 | 33 |
| Helsinki, newborns | *cannot run* | | | |

Both rows are patient-calibrated. Helsinki cannot run at all because these newborns have
no seizure-free baseline to calibrate on.

With one global threshold for all three, so the method is held fixed and only the
population changes:

| cohort | 5 FA/h | 10 FA/h | 20 FA/h | events |
|---|---|---|---|---|
| **Siena, adults** | **0.727** | **0.879** | **0.970** | 33 |
| CHB-MIT, children | 0.524 | 0.741 | 0.867 | 166 |
| Helsinki, newborns | 0.519 | 0.557 | 0.646 | 158 |

**Siena wins on 13 training patients.** That contradicts the patient-count curve above,
which predicted 13 patients would land near the bottom. Adult epilepsy-monitoring data
appears easier in ways that outweigh having fewer subjects: long recordings, clean
seizure morphology, less movement artifact than children produce.

**Read Siena's row with care.** 33 events against CHB-MIT's 168. Two missed seizures move
it by 6 points, so the ordering between Siena and CHB-MIT is not established. Helsinki's
158 events make its row the firmest of the three, and it is clearly the hardest cohort.

### Per-patient thresholds cannot be used on Helsinki's newborns

The full pipeline does not run on Helsinki at all. Thresholds are calibrated on a
seizure-free stretch before a patient's first seizure, and these recordings often do not
have one: monitoring started *because* seizures were already suspected. Of 12 held-out babies, four
have zero background windows before their first seizure -- HEL07's begins 28 seconds in --
and one has no usable split.

That component is worth +0.139 on CHB-MIT, the second-largest positive result here. It
depends on a recording protocol that monitoring started for suspected seizures does not
follow, which is a limit on the method rather than on the model. It is not true of every
newborn: babies cooled for hypoxic-ischemic encephalopathy are often monitored before their
first seizure (Bernardo et al., Epilepsia 2025).

Siena has the opposite profile: 12 of its 14 patients have thousands of background windows
before their first seizure. PN07 and PN11 have a single seizure each, so nothing remains
to test on once one is spent on adaptation, and they are skipped -- as CHB03 and CHB05 are
on CHB-MIT.

## Three things that fooled us

These were found the hard way, by getting them wrong first. **None is a new discovery** --
each was already established in the literature, and the references are given so a reader
can go to the stronger source. They are kept here because the failure modes are concrete,
measured on this pipeline, and worth knowing before trusting any number in a seizure paper,
including this one.

**AUC got three decisions backwards.** It ranks single windows, but a detector is judged
on catching a *seizure* and on false alarms -- both properties of runs of windows.
Measured on AUC, dropping three channels looked free (it cost 0.109 sensitivity), adding
Siena looked harmful (it helped), a foundation model looked equal (it is worse). Every
number here is now event-level.

> Established work: **NeuroAtlas** (42 datasets, 260,000 h) makes event-level sensitivity
> over 0.1-100 FA/h its primary metric and shows window AUROC and event sensitivity rank
> models differently, Spearman rho 0.61 to 0.81 across seven cohorts -- with the weakest
> agreement on Helsinki and Siena, two of the three used here. The **SzCORE Challenge**
> found that across 28 algorithms, "nearly all strongly overestimated their performance"
> against an independent test set.

**A 3-fold screen overstated a result by 6x.** The Helsinki cohort was screened at 3 folds
and gained +0.145 sensitivity, +0.27 at 10 FA/h. At 23 folds the same comparison gives
+0.024, and the low-false-alarm advantage disappears. Fold assignment alone swings results
by about +/-0.08 here, which was already documented, and the screen's result sat inside
that. Screens are for deciding what to run properly, not for reporting.

**Experts disagree about 47% of seizure time.** Helsinki's three doctors annotated all 79
babies independently. So "when did the seizure start" is a judgement, not a fact -- which
is why training on the most confusing background made things worse. Those windows sit
next to seizures because they partly *are* seizures.

> Established work: **Stevenson et al. 2015** measured interobserver agreement for neonatal
> seizure detection directly. **Abdi et al. 2025** compare consensus strategies -- unanimous,
> majority, any -- against class imbalance and rater count, and recommend reporting
> practice for neonatal detectors.

### What this project does not claim

The cohorts being near-perfectly separable (AUC 0.9995) is reported more sharply elsewhere,
at AUROC 1.000 from frozen foundation-model embeddings. The patient-count curve is a small
version of a scaling study run on 332 patients and 52,959 hours. Per-patient failure
analysis is covered by the SzCORE Challenge across 28 algorithms and 65 subjects, which
found hard seizures are shorter (median 48 s against 118 s) and that 23% of subjects scored
F1 = 0 for every top-5 algorithm.

One observation here is about recording protocol rather than modelling: **per-patient
threshold calibration is unavailable when EEG is ordered because seizures are already
suspected**, as in Helsinki, since there is no seizure-free baseline to calibrate on. That
is described above.

## What is left

Sixteen interventions, twelve of them null or negative, is enough to say the easy
directions are exhausted. What remains:

**More patients from the same cohort.** The only lever still climbing, and the only one
with a measured curve. CHB-MIT is out of patients at 23, so this needs data we do not
have -- a larger paediatric epilepsy-monitoring corpus, or TUH for adults.

**Longer temporal context, done properly.** Windows are classified in 4-second isolation,
which discards what separates a seizure from a transient artifact. A cheap version was
tested here, learning a filter over the probability sequence, and did not beat the
existing moving average at matched false alarms. Systems reporting false alarms per *day*
use 60-80 s of context inside the model rather than after it, which is a different and
untried change.

**Nothing on the model.** Capacity, architecture and a 4.9M-parameter pretrained backbone
were all tested. All neutral or worse.

The honest summary is that this project's results are about data and measurement, not
architecture, and the remaining data lever requires patients nobody here has.

## Limitations

- **Not clinically usable** -- 5.2 false alarms/hour, 124 per day, against the under-1
  per hour needed. Subject-independent baselines elsewhere report 1-2 per day.
- **Two patients dominate the flagged time.** CHB12 and CHB13 have 34% and 25% of their
  recordings flagged. The average describes almost nobody.
- **False alarms concentrate in a few patients.** CHB15 and CHB06 produce 35% of all
  strict false alarms between them. CHB09 produces none.
- **5 of 24 cases are personalised** on one of their own earlier seizures. A patient
  with no recorded seizure does worse.
- **Test-patient labels are read for all 24 cases**, not only the 5. The decision whether
  to personalise is made by checking the base model against that patient's own first
  seizure. Removing it costs 16.1 points of pooled sensitivity, measured in
  `szcore_lopo_uncalibrated.txt`.
- **Per-patient thresholds need a seizure-free baseline**, which recordings started for
  suspected seizures often do not have. The method's second-biggest win is unavailable on
  Helsinki.
- **Siena's own-cohort result rests on 33 events.** CHB-MIT has 168, Helsinki 158.

## Files

| | |
|---|---|
| `edf_to_npz.py`, `parse_siena.py`, `siena_to_npz.py`, `helsinki_to_npz.py` | recordings to windowed arrays |
| `model.py`, `config.py` | the EEG-Conformer |
| `train_lopo.py` | one model per held-out patient |
| `train_memory_efficient.py` | faster 3-fold version, for screening ideas |
| `evaluate_end_to_end.py` | full pipeline, thresholds from each patient's own recording |
| `uncalibrated_lopo.py` | the same folds with thresholds from training patients only |
| `score_official.py` | scoring with the official `timescoring` package |
| `evaluate_temporal.py`, `score_szcore.py` | event-level and SzCORE scoring |
| `calibrate_per_patient.py`, `finetune_per_patient.py` | thresholds and personalisation |
| `check_domain_shift.py` | how separable two cohorts are |

Results indexed in [`cv_results/README.md`](cv_results/README.md).

## Data

Not included. CHB-MIT and Siena are on PhysioNet, Helsinki on Zenodo (CC-BY-4.0). The
scripts above download and convert them.

## References

**Methods and data**

Song et al., *EEG Conformer*, IEEE TNSRE 2022 | Wang et al., *CBraMod*, ICLR 2025 |
Shoeb, MIT 2009 (CHB-MIT) | Detti et al., 2020 (Siena) | Stevenson et al.,
*A dataset of neonatal EEG recordings with seizure annotations*, Sci Data 2019 (Helsinki)

**Evaluation and prior findings this work reproduces**

- Dan et al., *SzCORE: A Seizure Community Open-source Research Evaluation framework*, 2024
- *Quantifying the Generalization Gap in Seizure Detection: the SzCORE Challenge*,
  arXiv 2505.18191 -- 28 algorithms, best 37% sensitivity at 1.34 FP/day
- *NeuroAtlas: Benchmarking Foundation Models for Clinical EEG*, arXiv 2605.14698 --
  event-level sensitivity at fixed FA/h as the primary metric
- Pale et al., *Scaling convolutional neural networks achieves expert level seizure
  detection in neonatal EEG*, npj Digital Medicine 2025 -- 332 patients, 52,959 h
- Stevenson et al., *Interobserver agreement for neonatal seizure detection using
  multichannel EEG*, Ann Clin Transl Neurol 2015
- *Generalization or mirage? Data leakage and reported performance in neonatal EEG
  seizure detection models*, BioData Mining 2025

## License

MIT
