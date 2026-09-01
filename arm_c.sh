#!/bin/sh
# Arm C: both wins at once -- 23 channels and 36 training subjects.
#
# The 20-channel experiment established two things separately. Dropping the
# FT9/FT10 derivations cost 0.109 event sensitivity (0.886 -> 0.777). Adding
# Siena's 14 patients gained 0.084 (0.777 -> 0.861). The two effects are close
# in size and opposite in sign, which is why the 20-channel merge landed just
# below the plain 23-channel CHB-MIT model despite training on 14 more people.
#
# Siena has no FT9/FT10 electrode, but it does place F9 and F10 immediately
# anterior to them, in all 41 recordings. Substituting those gives an
# approximation of the three missing derivations -- inexact, but the measured
# alternatives are worse: dropping costs 0.109, and zero-filling would hand the
# model a constant channel that identifies the cohort outright.
#
# Control is the existing 23-channel CHB-MIT-only LOPO at 0.886 sensitivity /
# 14.42 FA/h, which used the same folds, seed, epochs and segment cap. Only the
# training set differs.
set -e
cd "$(dirname "$0")"

SIENA="PN00 PN01 PN03 PN05 PN06 PN07 PN09 PN10 PN11 PN12 PN13 PN14 PN16 PN17"

echo "=== waiting for the 23-channel Siena conversion"
while ! grep -q "written to" logs/siena23_convert.log 2>/dev/null; do
    sleep 120
done
echo "    conversion done"

echo "=== how separable are the cohorts at 23 channels?"
python -u check_domain_shift.py --siena preprocessed_data_siena23 \
    --keep-ft --per-channel 2>&1 | tee logs/domain_shift_23.log

echo
echo "=== ARM C: 36 training subjects, 23 channels"
python -u train_lopo.py --epochs 15 --max-segments 2000 --tag lopo23siena \
    --data-dirs preprocessed_data preprocessed_data_siena23 \
    --always-train $SIENA \
    > logs/merge_armC.log 2>&1
echo "    training done"

echo "=== event-level evaluation"
python -u evaluate_end_to_end.py --target-fa 2.0 \
    --ckpt-glob "lopo/lopo23siena_*.pt" --cache endtoend_armC.npz \
    > logs/eval_armC.log 2>&1
grep -A4 personalised logs/eval_armC.log | tail -5

python -u score_szcore.py --cache endtoend_armC.npz 2>&1 | grep -A3 POOLED

echo "ARM C DONE"
