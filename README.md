# Seizure Detection from Scalp EEG

Finds epileptic seizures in EEG recordings, tested only on patients the model has
never seen.

**94% of seizures found, 5.2 false alarms per hour**, over 170 hours of held-out
recording.

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

Trained by holding one patient out at a time. Nothing about the test patient is used
in training.

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
| Published CHB-MIT results | 75-85% | 1-5 / hour |
| **This work** | **94%** | **5.2 / hour** |
| Rule-based detector (Gotman, 1982) | 91% | ~10 / hour |

Scored with [SzCORE](https://epilepsybenchmarks.com) rules. A stricter scorer gives
**89% at 14.4/hour** on identical predictions, so the scorer must always be named.

That last row is the calibration: **high sensitivity is cheap.** A 1982 rule-based
detector hits 91% if you tolerate enough noise. Suppressing false alarms is the real
problem, and it is not solved here.

## What worked, and how we know

Every comparison below is **paired**: same folds, same held-out patients, one variable
changed. All are scored on *events* (seizures caught, false alarms per hour), never on
window-level AUC, which got three of these calls backwards.

| | what | effect | how we know |
|---|---|---|---|
| **yes** | Fixed a bug in the seizure labels | biggest single change | rerun the identical pipeline before and after the fix: AUC 0.32 to 0.78 |
| **yes** | Smoothing predictions over time | ~3x fewer false alarms | same saved predictions scored with and without smoothing |
| **yes** | A threshold per patient | +0.139 sensitivity | swept a single global threshold over identical predictions; no setting reaches 0.886 at all |
| **yes** | More patients *from the same cohort* | **+0.405** event sensitivity at 10 FA/h | 8 to 67 Helsinki training patients, fixed held-out test set. CHB-MIT gave +0.10 from 16 to 22, but it has only 23 subjects and cannot show more of the curve |
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
fixed held-out set of 12, scored on events at matched false-alarm rates:

| training patients | 5 FA/h | 10 FA/h | 20 FA/h | 40 FA/h |
|---|---|---|---|---|
| 8 | 0.152 | 0.152 | 0.519 | 0.684 |
| 16 | 0.361 | 0.411 | 0.513 | 0.646 |
| 32 | -- | 0.418 | 0.551 | 0.703 |
| **67** | **0.519** | **0.557** | **0.646** | 0.677 |

Two things follow, and the second is the useful one.

**It is still climbing at 67.** No ceiling is visible, so the 23 subjects of CHB-MIT are
the shallow end of this curve rather than most of it.

**The benefit is entirely at low false-alarm rates.** Going from 8 patients to 67 is worth
**+0.405 at 10 FA/h** and **-0.006 at 40 FA/h**. When the detector is already alarming
every 90 seconds, more patients buy nothing; the gain appears exactly where a detector
would have to operate to be useful. Every earlier measurement in this project was taken
at one operating point, which would have hidden this.

Subsets are nested and the seizure-bearing to background-only mix is held constant, so the
curve measures adding patients rather than resampling them.

## One detector per population

Three separate models, each trained and scored only on its own cohort. Not a merge --
merging was tested and did not work.

With the full pipeline, where the data supports it:

| cohort | seizures caught | FA/h | events |
|---|---|---|---|
| CHB-MIT, children | 94.0% | 5.2 | 168 |
| Siena, adults | 93.9% | 5.2 | 33 |
| Helsinki, newborns | *cannot run* | | |

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

### Per-patient thresholds cannot be used on newborns

The full pipeline does not run on Helsinki at all. Thresholds are calibrated on a
seizure-free stretch before a patient's first seizure, and NICU recordings do not have
one: monitoring starts *because* the baby is already seizing. Of 12 held-out babies, four
have zero background windows before their first seizure -- HEL07's begins 28 seconds in --
and one has no usable split.

That component is worth +0.139 on CHB-MIT, the second-largest positive result here. It
depends on a recording protocol the neonatal setting does not follow, which is a limit on
the method rather than on the model.

Siena has the opposite profile: 12 of its 14 patients have thousands of background windows
before their first seizure. PN07 and PN11 have a single seizure each, so nothing remains
to test on once one is spent on adaptation, and they are skipped -- as CHB03 and CHB05 are
on CHB-MIT.

## Three things that fooled us

**AUC got three decisions backwards.** It ranks single windows, but a detector is judged
on catching a *seizure* and on false alarms -- both properties of runs of windows.
Measured on AUC, dropping three channels looked free (it cost 0.109 sensitivity), adding
Siena looked harmful (it helped), a foundation model looked equal (it is worse). Every
number here is now event-level.

**A 3-fold screen overstated a result by 6x.** The Helsinki cohort was screened at 3 folds
and gained +0.145 sensitivity, +0.27 at 10 FA/h. At 23 folds the same comparison gives
+0.024, and the low-false-alarm advantage disappears. Fold assignment alone swings results
by about +/-0.08 here, which was already documented, and the screen's result sat inside
that. Screens are for deciding what to run properly, not for reporting.

**Experts disagree about 47% of seizure time.** Helsinki's three doctors annotated all 79
babies independently. So "when did the seizure start" is a judgement, not a fact -- which
is why training on the most confusing background made things worse. Those windows sit
next to seizures because they partly *are* seizures.

Full detail and every negative result: **[report.html](report.html)**.

## Limitations

- **Not clinically usable** -- 5.2 false alarms/hour against the under-1 needed.
- **Two patients dominate the false alarms.** CHB12 and CHB13 have 34% and 25% of their
  recordings flagged. The average describes almost nobody.
- **5 of 24 patients are personalised** on one of their own earlier seizures. A patient
  with no recorded seizure does worse.
- **Per-patient thresholds need a seizure-free baseline**, which neonatal recordings do
  not have. The method's second-biggest win is unavailable in the NICU.
- **Siena's own-cohort result rests on 33 events.** CHB-MIT has 168, Helsinki 158.

## Files

| | |
|---|---|
| `edf_to_npz.py`, `parse_siena.py`, `siena_to_npz.py`, `helsinki_to_npz.py` | recordings to windowed arrays |
| `model.py`, `config.py` | the EEG-Conformer |
| `train_lopo.py` | one model per held-out patient |
| `train_memory_efficient.py` | faster 3-fold version, for screening ideas |
| `evaluate_end_to_end.py` | full pipeline, nothing chosen using test data |
| `evaluate_temporal.py`, `score_szcore.py` | event-level and SzCORE scoring |
| `calibrate_per_patient.py`, `finetune_per_patient.py` | thresholds and personalisation |
| `check_domain_shift.py` | how separable two cohorts are |

Results indexed in [`cv_results/README.md`](cv_results/README.md).

## Data

Not included. CHB-MIT and Siena are on PhysioNet, Helsinki on Zenodo (CC-BY-4.0). The
scripts above download and convert them.

## References

Song et al., *EEG Conformer*, IEEE TNSRE 2022 | Wang et al., *CBraMod*, ICLR 2025 |
Shoeb, MIT 2009 (CHB-MIT) | Stevenson et al., 2019 (Helsinki) | Dan et al., *SzCORE*, 2024

## License

MIT
