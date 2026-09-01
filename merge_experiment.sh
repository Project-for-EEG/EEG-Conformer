#!/bin/sh
# Does adding 14 Siena patients help on CHB-MIT patients?
#
# The only intervention that has ever moved this project's headline is more
# training *subjects*: going from 16 to 22 gained +0.10 event sensitivity,
# while every attempt to add more hours, more seizures, or synthetic data
# changed nothing. Siena adds 14 more people, taking training from 22 subjects
# to 36.
#
# Both arms hold out the same 23 CHB-MIT subjects, one at a time, and are
# scored on those held-out patients. Siena patients are pinned into every
# training set and never held out, so the arms differ in exactly one thing:
# whether Siena was in the training data. Same folds, same seed, same channels,
# same evaluation.
#
# 20 channels in both arms. Siena has no FT9/FT10 electrode, so CHB-MIT
# channels 19-21 cannot be derived from it; measured over 9 paired folds,
# dropping them costs nothing (+0.014, CI [-0.012, +0.040]). Arm A must drop
# them too, or the channel change would be confounded with the patient change.
#
# Distinct tags keep the two arms' checkpoints apart, and train_lopo refuses to
# resume into a checkpoint written with a different channel count or training
# set size, so a stale cache cannot quietly answer the wrong question.
set -e
cd "$(dirname "$0")"

SIENA="PN00 PN01 PN03 PN05 PN06 PN07 PN09 PN10 PN11 PN12 PN13 PN14 PN16 PN17"
COMMON="--drop-channels 19 20 21 --epochs 15 --max-segments 2000"

echo "=== ARM A: 22 training subjects, CHB-MIT only, 20 channels"
python -u train_lopo.py $COMMON --tag lopo20 \
    > logs/merge_armA.log 2>&1
echo "    arm A done"

echo "=== ARM B: 36 training subjects, CHB-MIT + Siena, 20 channels"
python -u train_lopo.py $COMMON --tag lopo20siena \
    --data-dirs preprocessed_data preprocessed_data_siena \
    --always-train $SIENA \
    > logs/merge_armB.log 2>&1
echo "    arm B done"

echo "MERGE EXPERIMENT DONE"
