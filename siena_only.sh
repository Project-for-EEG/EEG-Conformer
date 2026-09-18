#!/bin/sh
# Siena on its own: trained and scored only on adults.
#
# The third of three single-cohort models. CHB-MIT (children) and Helsinki
# (newborns) are both measured; this completes the table.
#
# Leave-one-patient-out over 14 adults, so 13 train per fold. That is the
# smallest training set of the three, and the patient-count curve measured on
# Helsinki says 13 is deep in the shallow end -- 8 patients scored 0.152 at
# 10 FA/h against 0.557 for 67. A poor result here is expected and is the same
# finding seen from the other side, not a separate failure.
#
# Siena is also the hardest cohort by class balance: 0.55% seizure against
# CHB-MIT's 1.4% and Helsinki's 12.4%.
#
# --max-segments 2000 matches the CHB-MIT headline run. Memory is not a
# constraint here: 13 patients at 2000 is 2.4 GB against the 16.5 GB that made
# a Helsinki fold thrash.
#
# PN07 and PN11 have a single seizure each, so nothing remains to test on after
# one is spent on adaptation. They are skipped, as CHB03 and CHB05 are on
# CHB-MIT.
set -e
cd "$(dirname "$0")"

python -u train_lopo.py --epochs 15 --max-segments 2000 --tag sienaonly_ \
    --data-dirs preprocessed_data_siena23 \
    > logs/siena_only_train.log 2>&1
echo "training done"

python -u evaluate_end_to_end.py --target-fa 2.0 \
    --ckpt-glob "lopo/sienaonly__*.pt" --data-root preprocessed_data_siena23 \
    --cache endtoend_siena_only.npz > logs/siena_only_eval.log 2>&1
grep -A4 personalised logs/siena_only_eval.log | tail -4

python -u score_szcore.py --cache endtoend_siena_only.npz 2>&1 | grep -A3 POOLED
echo "SIENA ONLY DONE"
