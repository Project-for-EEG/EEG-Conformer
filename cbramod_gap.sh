#!/bin/sh
# Close the CBraMod gap: does the foundation model help at EVENT level?
#
# The existing verdict -- "+0.001, no benefit" -- is window-level AUC on 3
# folds. Since it was written, AUC has been caught calling two other decisions
# backwards: it rated dropping three channels as +0.026 when the true event
# cost was -0.109, and rated the Siena merge at -0.029 when the true event gain
# was +0.084. The CBraMod conclusion rests on exactly the evidence type that
# failed twice, so it is currently the least supported claim in the repo.
#
# This scores both architectures the way a detector is actually judged: event
# sensitivity and false alarms per hour, through the same honest pipeline
# (chronological split, unsupervised calibration, selective personalisation).
#
# The comparison is paired. Both trainers were verified to produce identical
# 3-fold splits over the same 22 patients (CHB03 and CHB05 excluded, their
# local copies predate raw_data/ and do not match PhysioNet). CBraMod reads
# cbramod_data at 200 Hz in uV/100; EEG-Conformer reads preprocessed_data at
# 256 Hz z-scored. Feeding either model the other's format is silently wrong
# rather than an error, which is why --data-root is explicit.
set -e
cd "$(dirname "$0")"

echo "waiting for Arm C to finish its evaluation"
while [ ! -f endtoend_armC.npz ]; do sleep 300; done
echo "  Arm C done"

echo "=== EEG-Conformer, same 22 patients, same folds"
python -u train_memory_efficient.py --folds 3 --epochs 15 --max-segments 2000 \
    --balance none --exclude CHB03 CHB05 --ckpt-prefix conf22_ \
    > logs/conf22_train.log 2>&1
echo "  trained"

echo "=== event-level: EEG-Conformer"
python -u evaluate_end_to_end.py --target-fa 2.0 --ckpt-glob "conf22_*.pt" \
    --cache endtoend_conf22.npz > logs/eval_conf22.log 2>&1
grep -A4 personalised logs/eval_conf22.log | tail -4

echo "=== event-level: CBraMod"
python -u evaluate_end_to_end.py --target-fa 2.0 --ckpt-glob "cbramod_fold*.pt" \
    --data-root cbramod_data --cache endtoend_cbramod.npz \
    > logs/eval_cbramod.log 2>&1
grep -A4 personalised logs/eval_cbramod.log | tail -4

echo
echo "=== SzCORE, both"
for c in endtoend_conf22 endtoend_cbramod; do
    echo "--- $c"
    python -u score_szcore.py --cache $c.npz 2>&1 | grep -A3 POOLED
done
echo "CBRAMOD GAP DONE"
