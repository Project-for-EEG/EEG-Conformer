#!/bin/sh
# Fetch the Siena recordings the original download script missed.
#
# download_siena.ps1 guessed filenames from a pattern (PN12-1.edf, PN12-2.edf)
# but Siena names a multi-seizure recording after every seizure it contains
# (PN12-1.2.edf, PN10-4.5.6.edf, PN10-7.8.9.edf). The guessed names 404'd
# silently, so 11 recordings were never downloaded, and PN16/PN17 were absent
# from the script's patient list entirely -- two whole patients lost to a
# hardcoded guess. PN14-2.edf downloaded only 119 MB of 438 MB and is
# truncated, so it is resumed.
#
# The authoritative list is the RECORDS file inside the dataset itself.
#
# Downloads run 4 at a time. A single PhysioNet stream sustains ~110 KB/s from
# here, which is 14 hours for 7.4 GB; the limit is per-connection, not per-host,
# so four streams finish in about a quarter of that.
BASE=https://physionet.org/files/siena-scalp-eeg/1.0.0
OUT=additional_data/siena
JOBS=4

FILES="PN10/PN10-7.8.9.edf PN12/PN12-1.2.edf PN12/PN12-4.edf
PN13/PN13-2.edf PN13/PN13-3.edf PN14/PN14-2.edf
PN14/PN14-3.edf PN14/PN14-4.edf PN16/PN16-1.edf
PN16/PN16-2.edf PN17/PN17-1.edf PN17/PN17-2.edf"

for f in $FILES; do
    mkdir -p "$OUT/$(dirname "$f")"
done

# -C - resumes a partial file, which is exactly PN14-2's situation and also
# makes the whole script safe to re-run after an interruption
echo "$FILES" | tr ' ' '\n' | grep . | \
    xargs -P "$JOBS" -I{} sh -c \
    'curl -sS -L -C - --retry 5 --retry-delay 5 --retry-all-errors \
        -o "'"$OUT"'/{}" "'"$BASE"'/{}" && echo "done {}"'

for p in PN16 PN17; do
    curl -sS -L -o "$OUT/$p/Seizures-list-$p.txt" "$BASE/$p/Seizures-list-$p.txt"
done

echo "--- sizes on disk"
for f in $FILES; do
    printf '%-24s %s\n' "$f" "$(du -h "$OUT/$f" 2>/dev/null | cut -f1)"
done
echo "ALL DONE"
