#!/bin/sh
# Two extra seeds for the Helsinki patient-count curve, so the +0.304 headline
# gets an uncertainty rather than being a single draw. Seed 42 is already done
# and keeps its checkpoints; these write under their own tags.
#
# Note that --seed also chooses the held-out babies (helsinki_curve.py:162
# passes it to split_patients), so this varies BOTH the test set and the
# training run. That is the stricter question -- would the result hold with
# different babies as well as a different initialisation -- and the intervals
# will be wider than training noise alone.
set -e
for S in 43 44; do
  echo "=== seed $S ==="
  python -u helsinki_curve.py --seed "$S" --tag "helscurve${S}_"
done
echo "=== both seeds done ==="
