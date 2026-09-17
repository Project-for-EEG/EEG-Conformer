#!/bin/sh
# Leave-one-patient-out, with and without the Helsinki neonatal cohort.
#
# The 3-fold screen showed +0.145 event sensitivity from adding 79 newborns,
# and +0.27 at 10 false alarms per hour. This is the same question at
# publishable resolution: 23 folds, one CHB-MIT subject held out at a time.
#
# --max-segments 500 rather than the usual 2000, and both arms use it. At 2000
# a single fold with Helsinki pinned needs 16.5 GB against 17.1 GB of system
# RAM; measured directly, it drove available memory to zero and 7.5 GB into
# swap. That thrashing is the likely cause of fold times swinging between 46
# and 83 minutes, of a benchmark setting running 18x slow, and of an earlier
# overnight run dying at fold 15. At 500 a fold is 5.2 GB with 5.5 GB free.
#
# The cap limits background only; every seizure window is kept regardless. More
# background was tested directly once before and did nothing.
#
# Running the control at 500 as well is the point: comparing a new 500-segment
# result against the published 2000-segment number would confound the cohort
# with the cap.
set -e
cd "$(dirname "$0")"

HELS=$(cat helsinki_patients.txt)
COMMON="--epochs 15 --max-segments 500 --drop-channels 19 20 21"

echo "=== CONTROL: CHB-MIT only, 23 folds"
python -u train_lopo.py $COMMON --tag lopoA_ \
    > logs/lopoA_train.log 2>&1
python -u evaluate_end_to_end.py --target-fa 2.0 --ckpt-glob "lopo/lopoA__*.pt" \
    --cache endtoend_lopoA.npz > logs/lopoA_eval.log 2>&1
grep -A4 personalised logs/lopoA_eval.log | tail -4

echo
echo "=== TEST: CHB-MIT + 79 Helsinki neonates, 23 folds"
python -u train_lopo.py $COMMON --tag lopoB_ \
    --data-dirs preprocessed_data preprocessed_data_helsinki20 \
    --always-train $HELS \
    > logs/lopoB_train.log 2>&1
python -u evaluate_end_to_end.py --target-fa 2.0 --ckpt-glob "lopo/lopoB__*.pt" \
    --cache endtoend_lopoB.npz > logs/lopoB_eval.log 2>&1
grep -A4 personalised logs/lopoB_eval.log | tail -4

echo
echo "=== SzCORE"
for c in endtoend_lopoA endtoend_lopoB; do
    echo "--- $c"
    python -u score_szcore.py --cache $c.npz 2>&1 | grep -A3 POOLED
done
echo "LOPO HELSINKI DONE"
