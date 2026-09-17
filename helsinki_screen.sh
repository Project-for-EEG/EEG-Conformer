#!/bin/sh
# Does a third cohort -- 79 neonates from Helsinki -- help on CHB-MIT patients?
#
# Both arms hold out the same CHB-MIT patients in the same three folds and are
# scored on them. Helsinki patients are pinned into every training set and
# never held out, so the arms differ in exactly one thing: whether Helsinki was
# in the training data.
#
# 20 channels in both arms. Helsinki has no FT9/FT10 electrode and, unlike
# Siena, no F9/F10 either, so the three CHB-MIT channels that use them cannot
# be reconstructed. Verification showed the nearest substitutes are an exact
# negative of channel 1 and an exact duplicate of channel 13 -- two of three
# carrying no information the model does not already see. A 23-channel arm is
# deferred until this one shows Helsinki is worth anything at all.
#
# This is the cheap tier: 3 folds with event-level scoring, about 9 hours,
# rather than leave-one-patient-out at 4 to 5 days. Screening on events rather
# than AUC is the correction to how the channel decision was mishandled --
# AUC called that one backwards twice.
set -e
cd "$(dirname "$0")"

HELS=$(cat helsinki_patients.txt)
COMMON="--folds 3 --epochs 15 --max-segments 2000 --balance none --drop-channels 19 20 21"

echo "=== CONTROL: CHB-MIT only, 20 channels"
python -u train_memory_efficient.py $COMMON --ckpt-prefix hctl_ \
    > logs/hels_control_train.log 2>&1
python -u evaluate_end_to_end.py --target-fa 2.0 --ckpt-glob "hctl_*.pt" \
    --cache endtoend_hels_control.npz > logs/hels_control_eval.log 2>&1
grep -A4 personalised logs/hels_control_eval.log | tail -4

echo
echo "=== TEST: CHB-MIT + 79 Helsinki neonates, 20 channels"
python -u train_memory_efficient.py $COMMON --ckpt-prefix htest_ \
    --data-dirs preprocessed_data preprocessed_data_helsinki20 \
    --always-train $HELS \
    > logs/hels_test_train.log 2>&1
python -u evaluate_end_to_end.py --target-fa 2.0 --ckpt-glob "htest_*.pt" \
    --cache endtoend_hels_test.npz > logs/hels_test_eval.log 2>&1
grep -A4 personalised logs/hels_test_eval.log | tail -4

echo
echo "=== SzCORE"
for c in endtoend_hels_control endtoend_hels_test; do
    echo "--- $c"
    python -u score_szcore.py --cache $c.npz 2>&1 | grep -A3 POOLED
done
echo "HELSINKI SCREEN DONE"
