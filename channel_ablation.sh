#!/bin/sh
# Is dropping the three FT9/FT10 derivations free?
#
# Siena has no FT9 or FT10 electrode, so CHB-MIT channels 19 (T7-FT9), 20
# (FT9-FT10) and 21 (FT10-T8) cannot be reconstructed from its montage. The
# merge has to either drop them from both cohorts or zero-fill them in Siena,
# and a constant-zero channel would tell the model exactly which dataset a
# window came from. Dropping is only the right answer if those channels carry
# little unique signal.
#
# One paired 3-fold comparison put the cost at -0.009 AUC with a 95% interval
# of [-0.047, +0.029] -- consistent with zero, but far too wide to act on. The
# limit is the fold count, not the run: at 3 folds the patient grouping alone
# swings results by ~0.08.
#
# Seeds 42, 7 and 13 reshuffle which patients share a fold. Each seed runs both
# channel counts on identical folds and identical background subsamples, so the
# comparison stays paired; nine paired folds narrow the interval by about 1.7x.
#
# Seed 42 is already done for both, so only 7 and 13 are run here.
set -e
cd "$(dirname "$0")"

for seed in 7 13; do
    for chans in 23 20; do
        echo "=== seed $seed, $chans channels"
        if [ "$chans" = "20" ]; then
            drop="--drop-channels 19 20 21"
        else
            drop=""
        fi
        python -u train_memory_efficient.py \
            --folds 3 --epochs 15 --max-segments 2000 --balance none \
            --seed "$seed" $drop > "logs/chanabl_s${seed}_c${chans}.log" 2>&1
        echo "    done"
    done
done
echo "ALL DONE"
