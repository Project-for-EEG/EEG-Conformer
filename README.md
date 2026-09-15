# Seizure Detection from Scalp EEG

Finds epileptic seizures in EEG recordings. Tested only on patients the model has
never seen.

**Result: 94% of seizures found, 5.2 false alarms per hour**, over 170 hours of
held-out recording.

That sensitivity is good. The false-alarm rate is not: clinical monitoring needs under
1 per hour. See [Where this stands](#where-this-stands).

---

## 1. What goes in

Raw EEG recordings (`.edf` files) from three hospitals:

| dataset | patients | hours | who | seizure rate |
|---|---|---|---|---|
| CHB-MIT | 23 | 220 | children, Boston | 1.4% |
| Siena | 14 | 142 | adults, Italy | 0.6% |
| Helsinki | 79 | 112 | newborns, Finland | 12.4% |

EEG is a set of voltage traces, one per electrode on the scalp, sampled hundreds of
times a second. A seizure looks like a burst of rhythmic activity across several
electrodes. Everything else -- chewing, moving, loose electrodes, ordinary sleep --
is background, and it is 99% of the recording.

Each dataset also ships human annotations saying when seizures happened.

## 2. What we do to it

Four steps turn a recording into something a model can read.

**Pick 23 channels.** EEG is recorded as one trace per electrode. We convert to the
standard *bipolar montage*: 23 differences between neighbouring electrode pairs, such
as `FP1-F7 = Fp1 - F7`. Differences cancel noise shared by both electrodes.

**Cut into 4-second windows.** Each window is 1024 samples at 256 Hz, overlapping the
previous one by half. A window is labelled *seizure* if more than half of it falls
inside an annotated seizure.

**Filter.** Keep 0.5 to 50 Hz, discarding slow drift and electrical noise.

**Normalise.** Each channel of each window is rescaled to mean 0, standard deviation 1,
so recordings from different machines look comparable.

Output of this stage: arrays of shape `(windows, 23, 1024)` saved as `.npz`.

```
one recording  ->  (1799, 23, 1024) float32   the signal
                   (1799,)          int64     0 or 1 per window
```

## 3. The model

**EEG-Conformer** -- 120,962 parameters. Small by modern standards, and deliberately so:
we tested a 4.9M-parameter pretrained model and it did worse.

```
input   (23, 1024)   23 channels, 4 seconds
        convolution over time      finds waveform shapes
        convolution over channels  finds which electrodes agree
        transformer, 4 layers      relates parts of the window
output  2 numbers    probability of seizure vs not
```

Trained by holding one patient out at a time: train on everyone else, test on that one
person, repeat. Nothing about the test patient is ever used in training.

## 4. What comes out

The model gives a seizure probability for every 4-second window. Three steps turn that
into usable alarms:

**Smooth** over neighbouring windows. A single odd window is noise; a real seizure lasts
tens of seconds. Roughly 3x fewer false alarms.

**Threshold, per patient.** Each patient gets their own cutoff, set from their own
earlier recording. Patients differ enormously -- the fitted thresholds range from 0.036
to 0.998. Worth +0.139 sensitivity over any single shared threshold.

**Require a run.** At least 3 consecutive windows must clear the threshold before an
alarm fires.

Final output, per patient:

```
CHB09    3 seizures, 3 found, 0.00 false alarms/hour
CHB13   11 seizures, 11 found, 35.1 false alarms/hour
```

and pooled: **how many seizures were caught, and how often it cried wolf.**

## 5. How to run it

```bash
pip install -r requirements.txt

# get CHB-MIT and convert it
python edf_to_npz.py --fetch-missing

# train, holding out one patient at a time
python train_lopo.py --epochs 15 --max-segments 2000

# score it honestly
python evaluate_end_to_end.py --target-fa 2.0
python score_szcore.py --cache endtoend_probs.npz
```

Needs a CUDA GPU and about 15 GB RAM. Training takes ~13 h, evaluation ~25 min.

To add the other cohorts:

```bash
python parse_siena.py && python siena_to_npz.py --approximate-ft
python helsinki_to_npz.py --channels 20
```

## Results

| measure | value |
|---|---|
| Seizures found | **94%** (158 of 168) |
| False alarms | **5.2 per hour** |
| Tested on | 170 h, 24 patients, none seen in training |

Scored with [SzCORE](https://epilepsybenchmarks.com) event rules. A stricter scorer gives
**89% at 14.4 per hour** on the identical predictions -- the scorer alone moves false
alarms by 2.8x, so it must always be named.

### How this compares

| system | sensitivity | false alarms |
|---|---|---|
| 2025 SzCORE challenge winner *(private data)* | 37% | 1.3 / day |
| Published CHB-MIT results | 75-85% | 1-5 / hour |
| **This work** | **94%** | **5.2 / hour** |
| Rule-based detector (Gotman, 1982) | 91% | ~10 / hour |

That last row is the useful calibration. **High sensitivity is cheap** -- a rule-based
detector from 1982 hits 91% if you tolerate enough noise. Suppressing false alarms is the
actual difficulty, and it is not solved here.

## What we learned

Sixteen things were tried. Four helped.

| | what | effect |
|---|---|---|
| **yes** | Fixing a bug in the seizure labels | the single biggest change |
| **yes** | Smoothing predictions over time | ~3x fewer false alarms |
| **yes** | A threshold per patient | +0.139 sensitivity |
| **yes** | More patients | the only reliable lever |
| no | Rebalancing the classes (6 ways) | nothing |
| no | A pretrained foundation model | worse |
| worse | Training on the "hard" background only | false alarms nearly doubled |

**More patients is the one thing that works.** Going from 16 to 22 training patients
gained 10 points of sensitivity. Adding 79 newborns from a third hospital gained 15 more
-- and the gain was largest at low false-alarm rates, where it matters.

**More data of the same kind does not.** More hours, more seizures, synthetic seizures:
all measured, all nothing.

**The cohorts do not need to resemble each other.** A classifier tells CHB-MIT from
Helsinki at AUC 0.9995 -- children versus newborns, different countries, nine times the
seizure rate. It helped anyway. The benefit is diversity, not similarity.

## Two things that fooled us

**AUC got three decisions backwards.** AUC ranks individual windows, but a detector is
judged on whether it catches a *seizure* and how often it cries wolf -- both properties of
runs of windows, not single ones. Measured on AUC, dropping three channels looked free
(it cost 0.109 sensitivity), adding Siena looked harmful (it helped), and a foundation
model looked equal (it is clearly worse). Every number here is now event-level.

**Experts disagree about seizures far more than expected.** Helsinki has three doctors
who annotated all 79 babies independently. They disagree on **47% of the seizure time.**
So "when did the seizure start" is a judgement, not a fact -- which is why training on
the most confusing background made things worse: those windows sit next to seizures
because they partly *are* seizures.

Full detail, including every negative result: **[report.html](report.html)**.

## Where this stands

- **Not clinically usable.** Needs under 1 false alarm/hour; this is at 5.2.
- **Two patients dominate the false alarms.** CHB12 and CHB13 have 34% and 25% of their
  recordings flagged. The average describes almost nobody.
- **Personalisation is part of the method.** 5 of 24 patients were fine-tuned on one of
  their own earlier seizures. A patient with no recorded seizure does worse.
- **Only CHB-MIT patients are ever scored.** Siena and Helsinki are training data only,
  so nothing here says how it performs on adults or newborns.

## Files

**Data preparation**

| file | what it does |
|---|---|
| `edf_to_npz.py` | CHB-MIT recordings to windowed arrays |
| `parse_siena.py`, `siena_to_npz.py` | same for Siena |
| `helsinki_to_npz.py` | same for Helsinki |
| `repair_labels.py` | rebuilds labels after the annotation fix |

**Training**

| file | what it does |
|---|---|
| `model.py`, `config.py` | the EEG-Conformer |
| `train_lopo.py` | one model per patient held out |
| `train_memory_efficient.py` | faster 3-fold version, for screening ideas |

**Evaluation** -- the part worth reusing

| file | what it does |
|---|---|
| `evaluate_end_to_end.py` | the full pipeline, nothing chosen using test data |
| `evaluate_temporal.py` | event-level scoring and smoothing |
| `score_szcore.py` | strict vs SzCORE scoring, side by side |
| `calibrate_per_patient.py` | per-patient thresholds |
| `finetune_per_patient.py` | personalisation, chronological split |
| `check_domain_shift.py` | how separable two cohorts are |

Result files are indexed in [`cv_results/README.md`](cv_results/README.md).

## The label bug

Worth knowing because it explains the project's history. The original annotation parser
took the first number on each line:

```python
match = re.search(r'(\d+)', line)   # "Seizure 1 Start Time: 1724 seconds"
                                    # captures "1", the index -- not 1724
```

CHB-MIT writes single-seizure files as `Seizure Start Time:` and multi-seizure files as
`Seizure 1 Start Time:`. On the second form the parser captured the seizure *number*, so
every such seizure was recorded as starting and ending at second 1.

**18 of 24 patients lost their entire ground truth.** The model still reported 99%
accuracy, because 99% of windows are not seizures. Fixing it took the dataset from 564 to
5,408 seizure windows and moved AUC from **0.32 to 0.78** -- more than every later
experiment combined.

## Data

Not included. CHB-MIT and Siena are public on PhysioNet, Helsinki on Zenodo
(CC-BY-4.0); the scripts above download and convert them.

## References

- Song et al., *EEG Conformer*, IEEE TNSRE 2022
- Wang et al., *CBraMod*, ICLR 2025
- Shoeb, *Application of Machine Learning to Epileptic Seizure Onset Detection*, MIT 2009
- Stevenson et al., *A dataset of neonatal EEG recordings with seizures annotations*, 2019
- Dan et al., *SzCORE*, 2024

## License

MIT
