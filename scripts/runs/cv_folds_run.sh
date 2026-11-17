#!/bin/sh
# 4-fold cross-validation of the Helsinki patient-count curve.
#
# Replaces a single fixed 12-baby test set. The decomposition showed that set
# was the limiting instrument: the 8-to-67 gain moved 0.285 across three draws
# of it, but only 0.057 when the babies were pinned and only the training seed
# varied. ~65% of the variance was which babies got tested.
#
# Every scorable baby is now in exactly one test fold, so each is scored once
# by a model that never saw it. Folds are stratified by reviewer count.
set -e
for F in 0 1 2 3; do
  echo "=== fold $F ==="
  python -u helsinki_curve.py --fold "$F" --tag "helscv${F}_"
  python -u helsinki_curve_events.py --fold "$F" --tag "helscv${F}_"
done
echo "=== all folds done ==="
