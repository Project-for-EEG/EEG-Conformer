#!/usr/bin/env bash
# v3_download.sh -- download the CHB-MIT files that are not in C:\EEG\raw_data
# into RAWFULL (test plan v3, step S4; D3 = yes, D5 = C:, user's OK 2026-11-16).
#
# Source: PhysioNet's open-data mirror. physionet.org redirects every EDF
# there (HTTP 302). Each file is kept only if its SHA-256 equals the value in
# PhysioNet's SHA256SUMS.txt. A file is written as <name>.part and renamed
# only after its checksum matches, so a stopped run can simply be restarted.
# Nothing in C:\EEG\raw_data is written. Stops before C: falls under 10 GB free.
#
# Usage (Git Bash):  bash v3_download.sh
set -u

S3=https://physionet-open.s3.amazonaws.com/chbmit/1.0.0
LOCAL=/c/EEG/raw_data
RAWFULL=/c/chbmit_full
LOGDIR=/c/EEG_v3/download
PROCS=3
MIN_FREE=$((10 * 1000 * 1000 * 1000))   # 10 GB, plan stop rule
HEADROOM=$((300 * 1000 * 1000))          # room for one more file

mkdir -p "$RAWFULL" "$LOGDIR"
LOG="$LOGDIR/download_log.txt"
export S3 RAWFULL LOGDIR LOG MIN_FREE HEADROOM

ts() { date '+%Y-%m-%d %H:%M:%S'; }
export -f ts

echo "$(ts) START $0 procs=$PROCS" >> "$LOG"

# 1. Checksum list (not itself in the list) and the file list.
curl -sS --fail --retry 5 --retry-delay 15 --retry-all-errors \
     -o "$RAWFULL/SHA256SUMS.txt" "$S3/SHA256SUMS.txt" || { echo "$(ts) FAIL SHA256SUMS.txt" >> "$LOG"; exit 1; }
tr -d '\r' < "$RAWFULL/SHA256SUMS.txt" > "$LOGDIR/sums.txt"
echo "$(ts) SHA256SUMS.txt $(wc -l < "$LOGDIR/sums.txt") lines, sha256 $(sha256sum "$RAWFULL/SHA256SUMS.txt" | cut -d' ' -f1)" >> "$LOG"

# 2. Job list: every EDF in RECORDS that is not local, plus all small
#    metadata files (summaries, .seizures, RECORDS*, SUBJECT-INFO, ANNOTATORS).
#    The PDF is skipped.
awk '{print $2}' "$LOGDIR/sums.txt" | grep -v '\.edf$' | grep -v '\.pdf$' > "$LOGDIR/jobs_meta.txt"
awk '{print $2}' "$LOGDIR/sums.txt" | grep '\.edf$' | sort | while read -r f; do
    [ -f "$LOCAL/$f" ] || echo "$f"
done > "$LOGDIR/jobs_edf.txt"
echo "$(ts) jobs: $(wc -l < "$LOGDIR/jobs_edf.txt") EDF, $(wc -l < "$LOGDIR/jobs_meta.txt") metadata" >> "$LOG"

get() {
    rel="$1"
    dest="$RAWFULL/$rel"
    want=$(awk -v f="$rel" '$2 == f {print $1}' "$LOGDIR_SUMS")
    if [ -z "$want" ]; then echo "$(ts) NOSUM $rel" >> "$LOG"; return 1; fi
    if [ -f "$dest" ] && [ "$(sha256sum "$dest" | cut -d' ' -f1)" = "$want" ]; then
        echo "$(ts) HAVE $rel" >> "$LOG"; return 0
    fi
    avail=$(df -B1 --output=avail /c | tail -1)
    if [ "$avail" -lt $((MIN_FREE + HEADROOM)) ]; then
        echo "$(ts) STOP-LOWDISK $rel avail=$avail" >> "$LOG"; return 255
    fi
    mkdir -p "$(dirname "$dest")"
    url="$S3/$(printf '%s' "$rel" | sed 's/+/%2B/g')"
    for attempt in 1 2; do
        t0=$(date +%s)
        curl -sS --fail --retry 5 --retry-delay 15 --retry-all-errors \
             --connect-timeout 30 -C - -o "$dest.part" "$url" 2>> "$LOGDIR/curl_errors.txt"
        rc=$?
        have=$( [ -f "$dest.part" ] && sha256sum "$dest.part" | cut -d' ' -f1 )
        if [ $rc -eq 0 ] && [ "$have" = "$want" ]; then
            mv -f "$dest.part" "$dest"
            echo "$(ts) OK $rel $(stat -c %s "$dest") B $(( $(date +%s) - t0 )) s" >> "$LOG"
            return 0
        fi
        echo "$(ts) RETRY $rel attempt=$attempt curl_rc=$rc sha_ok=no" >> "$LOG"
        rm -f "$dest.part"   # our own partial file, restart from zero
    done
    echo "$(ts) FAIL $rel" >> "$LOG"
    return 1
}
export -f get
export LOGDIR_SUMS="$LOGDIR/sums.txt"

# 3. Metadata first (small), then EDFs, PROCS at a time.
cat "$LOGDIR/jobs_meta.txt" | xargs -P "$PROCS" -I{} bash -c 'get "$1"' _ {}
cat "$LOGDIR/jobs_edf.txt"  | xargs -P "$PROCS" -I{} bash -c 'get "$1"' _ {}

# 4. Summary.
n_jobs=$(cat "$LOGDIR/jobs_meta.txt" "$LOGDIR/jobs_edf.txt" | wc -l)
n_ok=0; missing=""
while read -r f; do
    if [ -f "$RAWFULL/$f" ]; then n_ok=$((n_ok + 1)); else missing="$missing $f"; fi
done < <(cat "$LOGDIR/jobs_meta.txt" "$LOGDIR/jobs_edf.txt")
echo "$(ts) END present $n_ok of $n_jobs; missing:${missing:- none}" >> "$LOG"
echo "present $n_ok of $n_jobs; missing:${missing:- none}"
[ "$n_ok" -eq "$n_jobs" ]
