#!/bin/sh
# Variance decomposition for the Helsinki patient-count curve.
#
# Seeds 42/43/44 moved the held-out babies AND the initialisation together, and
# gave a spread of 0.285 on the 8-to-67 gain. That spread mixes two causes:
# which 12 babies are tested, and where training happens to land. They need
# opposite remedies, so they have to be separated.
#
# Here the split is PINNED at 42, so all runs are scored on the same 12 babies
# that seed 42 used, and only the initialisation varies. Whatever spread
# survives is training noise. The rest of the original 0.285 is patient
# sampling.
set -e
for S in 43 44; do
  echo "=== training seed $S, split pinned at 42 ==="
  python -u helsinki_curve.py --seed "$S" --split-seed 42 --tag "helsfix${S}_"
  python -u helsinki_curve_events.py --seed "$S" --split-seed 42 --tag "helsfix${S}_"
done
echo "=== fixed-split seeds done ==="
