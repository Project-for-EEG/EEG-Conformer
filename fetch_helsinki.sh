#!/bin/sh
# Fetch the Helsinki neonatal EEG corpus from Zenodo.
#
#   Stevenson et al., "A dataset of neonatal EEG recordings with seizures
#   annotations", doi 10.5281/zenodo.4940267, CC-BY-4.0.
#
# 79 newborns, 4.34 GB, no registration. Three independent expert annotators
# scored every recording, which no other cohort here provides: annotations_2017
# _A_fixed.csv, _B.csv and _C.csv. Where the three disagree is a direct measure
# of how fuzzy a seizure boundary is, and that fuzziness is what made
# hard-negative mining fail on CHB-MIT.
#
# Downloads run 4 at a time; a single Zenodo stream is slower than the link.
# curl -C - resumes, so the script is safe to re-run after an interruption --
# and PN14-2.edf in the Siena set proved that a partial EDF reads without
# complaint as a shorter recording, so completeness is checked afterwards
# rather than assumed.
set -e
cd "$(dirname "$0")"

OUT=additional_data/helsinki
API=https://zenodo.org/api/records/4940267
JOBS=4

mkdir -p "$OUT"

echo "listing files"
python - "$OUT" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
rec = json.loads(Path("hels.json").read_text())
lines = []
for f in rec["files"]:
    lines.append("%s\t%s" % (f["links"]["self"], f["key"]))
Path(out / "_manifest.tsv").write_text("\n".join(lines) + "\n")
print("  %d files, %.2f GB"
      % (len(lines), sum(f["size"] for f in rec["files"]) / 1e9))
PY

echo "downloading with $JOBS streams"
cut -f1,2 "$OUT/_manifest.tsv" | \
    xargs -P "$JOBS" -I{} sh -c '
        url=$(printf "%s" "{}" | cut -f1)
        name=$(printf "%s" "{}" | cut -f2)
        curl -sS -L -C - --retry 5 --retry-delay 5 --retry-all-errors \
             -o "'"$OUT"'/$name" "$url" && echo "  done $name"'

echo
echo "verifying sizes against the Zenodo record"
python - "$OUT" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
rec = json.loads(Path("hels.json").read_text())
bad = 0
for f in rec["files"]:
    p = out / f["key"]
    if not p.exists():
        print("  MISSING  %s" % f["key"])
        bad += 1
    elif p.stat().st_size != f["size"]:
        print("  SHORT    %s  %.1f of %.1f MB"
              % (f["key"], p.stat().st_size / 1e6, f["size"] / 1e6))
        bad += 1
if bad:
    print("  %d files incomplete; re-run this script to resume" % bad)
    raise SystemExit(1)
print("  all %d files complete" % len(rec["files"]))
PY

echo "HELSINKI DOWNLOAD DONE"
