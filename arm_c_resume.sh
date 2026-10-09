#!/bin/sh
# Resume Arm C from its saved folds, then evaluate.
#
# The first attempt died at fold 15 of 23 with no traceback -- terminated from
# outside rather than crashed. train_lopo.py caches each fold as a checkpoint
# and refuses to reuse one written with a different channel count or training
# set size, so resuming picks up the 14 completed folds and recomputes nothing.
set -e
cd "$(dirname "$0")"

SIENA="PN00 PN01 PN03 PN05 PN06 PN07 PN09 PN10 PN11 PN12 PN13 PN14 PN16 PN17"

python -u train_lopo.py --epochs 15 --max-segments 2000 --tag lopo23siena \
    --data-dirs preprocessed_data preprocessed_data_siena23 \
    --always-train $SIENA \
    >> logs/merge_armC.log 2>&1
echo "training done"

python -u evaluate_end_to_end.py --target-fa 2.0 \
    --ckpt-glob "lopo/lopo23siena_*.pt" --cache endtoend_armC.npz \
    > logs/eval_armC.log 2>&1
echo "eval done"
grep -A4 personalised logs/eval_armC.log | tail -5
python -u score_szcore.py --cache endtoend_armC.npz 2>&1 | grep -A3 POOLED
echo "ARM C DONE"
