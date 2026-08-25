# Result files

Each backs a claim made in the top-level README.

| file | AUC | what it shows |
|---|---|---|
| `patient_cv_results_20260824_230658.json` | 0.7795 +/- 0.0168 | **the headline run**: leak-free, 24 patients, 3-fold |
| `patient_cv_results_20260825_043725.json` | 0.7974 +/- 0.0296 | paired baseline on 22 patients, the EEG-Conformer column of the CBraMod table |
| `cbramod_20260825_054045.json` | 0.7985 +/- 0.0544 | CBraMod on the identical folds |
| `patient_cv_results_20260727_215252.json` | 0.7779 +/- 0.0300 | SMOTE **on** |
| `patient_cv_results_20260727_234217.json` | 0.8003 +/- 0.0056 | SMOTE **off** -- together these are the "synthetic data: no effect" claim |
| `patient_cv_results_20260817_120642.json` | 0.7927 +/- 0.0219 | weighted sampling off: no effect |
| `patient_cv_results_20260728_021408.json` | 0.7887 +/- 0.0327 | 2.7x more background: no effect |
| `patient_cv_results_20260817_171644.json` | 0.7911 +/- 0.0596 | +52% more seizures: no effect |
| `patient_cv_results_20260817_132308.json` | 0.7574 +/- 0.0064 | 2.8x model capacity: worse |
| `prior12_20260818_124735.json` | 0.7629 +/- 0.0253 | training prior 12% |
| `prior036_20260818_142312.json` | 0.7411 +/- 0.0428 | training prior 3.6% |
| `natural_prior_20260818_052538.json` | 0.7639 +/- 0.0317 | training prior 1.4% -- the three together are the "class prior: no effect across 8.6x" claim |

Note that AUC is a ranking measure, not the deployable number. The headline
83% sensitivity at 4.4 FA/h is event-level and comes from
`evaluate_end_to_end.py` plus `score_szcore.py`.
