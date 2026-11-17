"""v3_check_download.py -- check the CHB-MIT download (test plan v3, step S4).

Read-only. Run with: python -I v3_check_download.py
Stdlib plus v3_edf.read_header/complete (both stdlib only).

Checks
  D1  every RECORDS entry is in exactly one place: local raw_data or RAWFULL
  D2  every file in RAWFULL (EDF and metadata) has the SHA-256 of PhysioNet's
      SHA256SUMS.txt, recomputed here in Python
  D3  every RAWFULL EDF is complete: bytes = 256*(ns+1) + n_records*sum(spr)*2;
      files in EXCLUDED (plan section 3) are listed with their result, not counted
  D4  no .part files and no files that are not in SHA256SUMS.txt
  D5  C: keeps at least 10 GB free
  Report only: local raw_data EDFs against SHA256SUMS.txt; hours and GB.
"""
import hashlib
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v3_edf  # noqa: E402

LOCAL = Path(r'C:\EEG\raw_data')
RAWFULL = Path(r'C:\chbmit_full')
OUT = Path(r'C:\EEG_v3\download\download_checks.txt')
MIN_FREE = 10 * 1000 ** 3
# Plan section 3, Exclusions (user's OK, 2026-11-16). D2 still hashes these files.
EXCLUDED = {'chb17/chb17b_69.edf': 'PhysioNet\'s copy: EEG in the first 1462 of 3600 records and part of '
                                   'the next, all zero after'}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    lines, fails = [], []

    def out(s):
        lines.append(s)
        print(s, flush=True)

    sums = {}
    for ln in (RAWFULL / 'SHA256SUMS.txt').read_text(encoding='ascii').splitlines():
        if ln.strip():
            h, name = ln.split(None, 1)
            sums[name.strip()] = h
    out('SHA256SUMS.txt entries %d, sha256 %s' % (len(sums), sha256(RAWFULL / 'SHA256SUMS.txt')))

    records = [r.strip() for r in (RAWFULL / 'RECORDS').read_text(encoding='ascii').splitlines() if r.strip()]
    local_records = [r.strip() for r in (LOCAL / 'RECORDS.txt').read_text(encoding='ascii').splitlines() if r.strip()]
    same = sorted(records) == sorted(local_records)
    out('RECORDS entries %d (unique %d); same set as local RECORDS.txt: %s' % (len(records), len(set(records)), same))
    if not same:
        fails.append('RECORDS differs from local RECORDS.txt')

    # D1
    both, neither, n_local, n_full = [], [], 0, 0
    for r in records:
        a, b = (LOCAL / r).is_file(), (RAWFULL / r).is_file()
        n_local += a
        n_full += b
        if a and b:
            both.append(r)
        if not a and not b:
            neither.append(r)
    ok = not both and not neither and n_local + n_full == len(records)
    out('D1 %s  local %d + RAWFULL %d = %d of %d; in both %s; in neither %s'
        % ('PASS' if ok else 'FAIL', n_local, n_full, n_local + n_full, len(records), both or '[]', neither or '[]'))
    if not ok:
        fails.append('D1')

    # D2, D4
    files = sorted(p for p in RAWFULL.rglob('*') if p.is_file())
    bad_sha, not_listed, parts = [], [], []
    n_checked, total_bytes = 0, 0
    for p in files:
        rel = p.relative_to(RAWFULL).as_posix()
        if rel == 'SHA256SUMS.txt':
            continue
        if rel.endswith('.part'):
            parts.append(rel)
            continue
        if rel not in sums:
            not_listed.append(rel)
            continue
        n_checked += 1
        total_bytes += p.stat().st_size
        if sha256(p) != sums[rel]:
            bad_sha.append(rel)
    ok2 = not bad_sha
    out('D2 %s  %d files hashed, %d mismatches %s' % ('PASS' if ok2 else 'FAIL', n_checked, len(bad_sha), bad_sha or ''))
    if not ok2:
        fails.append('D2')
    ok4 = not parts and not not_listed
    out('D4 %s  .part files %s; files not in SHA256SUMS %s' % ('PASS' if ok4 else 'FAIL', parts or '[]', not_listed or '[]'))
    if not ok4:
        fails.append('D4')

    # D3
    incomplete, excluded, hours = [], [], 0.0
    edfs = [p for p in files if p.suffix == '.edf']
    for p in edfs:
        ok_c, on_disk, expected = v3_edf.complete(p)
        rel = p.relative_to(RAWFULL).as_posix()
        if rel in EXCLUDED:
            excluded.append('%s (complete %s, %d of %d B; %s)' % (rel, ok_c, on_disk, expected, EXCLUDED[rel]))
            continue
        if not ok_c:
            incomplete.append('%s (%d of %d B)' % (p.name, on_disk, expected))
            continue
        h = v3_edf.read_header(p)
        hours += h['n_records'] * h['record_duration'] / 3600.0
    ok3 = not incomplete
    out('D3 %s  %d RAWFULL EDFs, %d excluded, %d incomplete %s'
        % ('PASS' if ok3 else 'FAIL', len(edfs), len(excluded), len(incomplete), incomplete or ''))
    if not ok3:
        fails.append('D3')
    out('INFO D3 excluded, not counted: %s' % (excluded or '[]'))
    out('INFO RAWFULL: %.2f GB in %d listed files; EDF hours by header %.2f h (%d EDFs, excluded not counted)'
        % (total_bytes / 1e9, n_checked, hours, len(edfs) - len(excluded)))

    # D5
    free = shutil.disk_usage('C:\\').free
    ok5 = free >= MIN_FREE
    out('D5 %s  C: free %.2f GB (limit 10 GB)' % ('PASS' if ok5 else 'FAIL', free / 1e9))
    if not ok5:
        fails.append('D5')

    # Report only: local EDFs against PhysioNet's checksums
    local_edfs = sorted(LOCAL.rglob('*.edf'))
    mism, local_hours, no_header = [], 0.0, []
    for p in local_edfs:
        rel = p.relative_to(LOCAL).as_posix()
        try:
            h = v3_edf.read_header(p)
            local_hours += h['n_records'] * h['record_duration'] / 3600.0
        except ValueError:
            no_header.append(rel)
        if rel not in sums:
            mism.append(rel + ' (not listed)')
        elif sha256(p) != sums[rel]:
            ok_c, on_disk, expected = v3_edf.complete(p)
            mism.append('%s (complete %s, %d of %d B)' % (rel, ok_c, on_disk, expected))
    out('INFO local raw_data: %d EDFs, %d differ from SHA256SUMS: %s' % (len(local_edfs), len(mism), mism or '[]'))
    out('INFO hours by header: RAWFULL %.2f h + local %.2f h = %.2f h in %d EDFs (excluded not counted)%s'
        % (hours, local_hours, hours + local_hours, len(edfs) - len(excluded) + len(local_edfs) - len(no_header),
           '; local headers unreadable, not counted: %s' % no_header if no_header else ''))

    out('OVERALL %s: %d FAIL %s' % ('PASS' if not fails else 'FAIL', len(fails), fails or ''))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text('\n'.join(lines) + '\n', encoding='ascii')
    return 0 if not fails else 1


if __name__ == '__main__':
    sys.exit(main())
