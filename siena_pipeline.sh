#!/bin/sh
# Wait for the Siena downloads, then re-parse and convert.
#
# Ordering matters here for a reason that already bit once. siena_seizures.json
# records which annotated seizures have a file on disk; the first run was made
# while 12 recordings were still missing, so 15 seizures were written off as
# "no matching EDF". Converting against that stale JSON would produce correct
# looking .npz files with those seizures silently labelled as background.
#
# So: wait for every file to pass the EDF record-count check, re-parse, then
# convert. The completeness check is the same one siena_to_npz.py applies per
# file -- a truncated download keeps a valid header, and MNE reads it without
# complaint as a shorter recording.
set -e
cd "$(dirname "$0")"

echo "waiting for downloads to complete"
while :; do
    incomplete=$(python -c "
from pathlib import Path
from siena_to_npz import edf_is_complete
bad = [f.name for f in sorted(Path('additional_data/siena').glob('*/*.edf'))
       if not edf_is_complete(f)[0]]
print(len(bad), ' '.join(bad[:4]))
")
    n=$(echo "$incomplete" | cut -d' ' -f1)
    if [ "$n" = "0" ]; then
        break
    fi
    echo "  $incomplete"
    sleep 300
done

echo
echo "=== all downloads complete; re-parsing annotations"
python -u parse_siena.py

echo
echo "=== converting to CHB-MIT format"
python -u siena_to_npz.py 2>&1 | grep -v RuntimeWarning | grep -v "raw = mne"

echo
echo "=== domain shift against CHB-MIT"
python -u check_domain_shift.py

echo "SIENA PIPELINE DONE"
