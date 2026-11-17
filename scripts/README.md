# Scripts

The shell scripts behind the results. They moved here from the top folder.
Each is kept as it was run. The only change is in the eleven that had the
line `cd "$(dirname "$0")"`. It is now `cd "$(dirname "$0")/../.."`, so
they still run in the top folder.

Run them from the top folder, e.g. `bash scripts/runs/run_seeds.sh`.
`cv_folds_run.sh`, `fixed_split_seeds.sh`, `run_seeds.sh`, `score_seeds.sh`
and `fetch_siena_missing.sh` have no `cd` line, so they must be started there.

## data/ -- downloads and conversion

| script | what it does |
|---|---|
| `fetch_helsinki.sh` | downloads the Helsinki recordings and annotation files from Zenodo, listed in `hels.json` |
| `fetch_siena_missing.sh` | downloads the Siena files the first download missed or cut short, and the seizure lists for PN16 and PN17 |
| `siena_pipeline.sh` | waits for the Siena download, then parses and converts it and checks the domain shift |

## runs/ -- the runs behind the results

| script | what it ran | committed output |
|---|---|---|
| `arm_c.sh` | Siena merge at 23 channels, FT9/FT10 approximated from F9/F10 | none of its own; it stopped at fold 15 of 23, and its 14 finished folds are in the `arm_c_resume.sh` file |
| `arm_c_resume.sh` | resumes `arm_c.sh` from its saved folds | `cv_results/lopo23siena_20260901_043857.json` |
| `cbramod_gap.sh` | EEG-Conformer against CBraMod at event level, on the same 22 patients and folds | `cv_results/patient_cv_results_20260901_054040.json` (the EEG-Conformer training only) |
| `channel_ablation.sh` | 23 against 20 channels, seeds 7 and 13 | `cv_results/patient_cv_results_20260828_050852.json`, `_060915`, `_071314`, `_081329` |
| `cv_folds_run.sh` | Helsinki patient-count curve, 4-fold cross-validation | `cv_results/helsinki_curve_20261007_*.json`, `cv_results/helsinki_curve_events_20261007_*.json` |
| `fixed_split_seeds.sh` | Helsinki curve at seeds 43 and 44, test split fixed at seed 42 | `cv_results/helsinki_curve_20261006_064341.json`, `_075753`; `cv_results/helsinki_curve_events_20261006_064624.json`, `_080037` |
| `helsinki_screen.sh` | 3-fold screen: CHB-MIT alone against CHB-MIT plus Helsinki, 20 channels | `cv_results/patient_cv_results_20260915_054533.json`, `_114950` |
| `lopo_helsinki.sh` | leave-one-patient-out with and without Helsinki, 20 channels, `--max-segments 500` | `cv_results/lopoA__20260915_174606.json`, `cv_results/lopoB__20260916_084401.json` |
| `merge_experiment.sh` | leave-one-patient-out with and without Siena, 20 channels | `cv_results/lopo20_20260830_105704.json`, `cv_results/lopo20siena_20260831_042156.json` |
| `prior_eventlevel.sh` | event-level scores for the three class-prior models | none committed |
| `run_seeds.sh` | Helsinki curve at seeds 43 and 44; the seed also picks the test babies | `cv_results/helsinki_curve_20261006_021158.json`, `_032627` |
| `score_seeds.sh` | event scores for those two seeds | `cv_results/helsinki_curve_events_20261006_033052.json`, `_033257` |
| `siena_only.sh` | Siena-only leave-one-patient-out | `cv_results/sienaonly__20260917_065906.json` |

`arm_c_resume.sh`, `cbramod_gap.sh`, `helsinki_screen.sh`,
`lopo_helsinki.sh` and `siena_only.sh` also score events with
`evaluate_end_to_end.py` and `score_szcore.py`. Those scores went to logs and
caches, which are not committed. For these scripts, the JSON files listed
hold the training results only.
