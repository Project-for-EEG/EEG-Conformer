# Result files

Each backs a claim made in the top-level README.

| file | AUC | what it shows |
|---|---|---|
| `lopo_20260827_122021.json` | 0.8030 +/- 0.1744 | **the headline run**: leave-one-patient-out, 23 folds, 23 channels |
| `lopo20_20260830_105704.json` | 0.8289 +/- 0.1226 | LOPO at 20 channels, the matched control for the merge |
| `lopo20siena_20260831_042156.json` | 0.7997 | LOPO at 20 channels with Siena's 14 patients added |
| `lopo23siena_20260901_043857.json` | -- | LOPO at 23 channels with Siena added, the F9/F10 approximation |
| `patient_cv_results_20260901_054040.json` | -- | EEG-Conformer on the 22 CBraMod folds, event-level arm |
| `patient_cv_results_20260828_040131.json` | 0.7703 +/- 0.0349 | 20 channels, seed 42 |
| `patient_cv_results_20260828_050852.json` | 0.7793 | 23 channels, seed 7 |
| `patient_cv_results_20260828_060915.json` | 0.7962 | 20 channels, seed 7 |
| `patient_cv_results_20260828_071314.json` | 0.7317 | 23 channels, seed 13 |
| `patient_cv_results_20260828_081329.json` | 0.7654 | 20 channels, seed 13 -- these five are the 9 paired folds behind the channel ablation |
| `patient_cv_results_20260824_230658.json` | 0.7795 +/- 0.0168 | the earlier headline: leak-free, 24 patients, 3-fold |
| `patient_cv_results_20260825_043725.json` | 0.7974 +/- 0.0296 | paired baseline on 22 patients, the EEG-Conformer column of the CBraMod table |
| `cbramod_20260825_054045.json` | 0.7985 +/- 0.0544 | CBraMod on the identical folds |
| `patient_cv_results_20260727_215252.json` | 0.7779 +/- 0.0300 | SMOTE **on** |
| `patient_cv_results_20260727_234217.json` | 0.8003 +/- 0.0056 | SMOTE **off** -- together these are the "synthetic data: no effect" claim |
| `patient_cv_results_20260817_120642.json` | 0.7927 +/- 0.0219 | weighted sampling off: no effect |
| `patient_cv_results_20260728_021408.json` | 0.7887 +/- 0.0327 | 2.7x more background: no effect |
| `patient_cv_results_20260817_171644.json` | 0.7911 +/- 0.0596 | +52% more seizures: no effect |
| `patient_cv_results_20260817_132308.json` | 0.7574 +/- 0.0064 | 2.8x model capacity: worse |
| `hardneg_20260914_145408.json` | 0.4404 +/- 0.0847 | hard-negative mining: 21.4 FA/h against 11.8 baseline |
| `prior12_20260818_124735.json` | 0.7629 +/- 0.0253 | training prior 12% |
| `prior036_20260818_142312.json` | 0.7411 +/- 0.0428 | training prior 3.6% |
| `natural_prior_20260818_052538.json` | 0.7639 +/- 0.0317 | training prior 1.4% -- the three together are the "class prior: no effect across 8.6x" claim, since confirmed at event level (0.831 / 0.861 / 0.861 sensitivity at 11.8 / 13.7 / 14.1 FA/h) |

**AUC in this table is not the result.** It is kept because these runs were originally
compared on it, and because three of the project's findings are that it misleads: it
rated dropping three channels as +0.026 when the event cost was -0.109, rated the Siena
merge at -0.029 when the event gain was +0.084, and rated CBraMod at +0.001 when it is
0.104 worse at catching seizures. Event-level numbers live in the top-level README.

Note that AUC is a ranking measure, not the deployable number. The headline
83% sensitivity at 4.4 FA/h is event-level and comes from
`evaluate_end_to_end.py` plus `score_szcore.py`.
