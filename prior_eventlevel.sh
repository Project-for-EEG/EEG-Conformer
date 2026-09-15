#!/bin/sh
# Does training class balance affect FALSE ALARMS?
#
# Three models were trained at 12%, 3.6% and 1.4% seizure in the training set,
# an 8.6x range, on identical folds. They were compared on window-level AUC,
# scored 0.76-0.79 throughout, and recorded as "no effect". That is the same
# metric that has since been caught calling three other decisions backwards.
#
# The checkpoints also stored the 99th percentile of model output on pure
# background, and that did not stay flat at all:
#
#   training prior     fold 1     fold 2
#   12%                0.168      0.032
#   3.6%               0.021      0.0012
#   1.4%               0.0039     0.00021
#
# A model trained at 12% puts 1% of ordinary EEG above 0.168; at the true 1.4%
# it puts the same 1% above 0.0039, forty times lower. Thresholding is
# rank-based so AUC cannot see this, but it is precisely what sets the floor on
# false alarms per hour. Event-level scoring can see it.
#
# No retraining: these checkpoints already exist.
set -e
cd "$(dirname "$0")"

for tag in prior12 prior036 natural; do
    echo "=== $tag"
    python -u evaluate_end_to_end.py --target-fa 2.0 \
        --ckpt-glob "${tag}_fold*.pt" --cache endtoend_${tag}_ev.npz \
        > logs/eval_${tag}.log 2>&1
    grep -A4 personalised logs/eval_${tag}.log | tail -4
    python -u score_szcore.py --cache endtoend_${tag}_ev.npz 2>&1 | grep -A3 POOLED
    echo
done
echo "PRIOR EVENT-LEVEL DONE"
