"""
v3 allow-list of missing inputs for CHB-MIT, from the summary montage blocks.

Reads every raw_data/chbNN/chbNN-summary.txt, gives each "File Name:" entry the
latest "Channels in EDF Files:" or "Channels changed:" block above it, runs the
exact matcher (v3_edf.match_targets) on the block labels and lists every target
that cannot be built. For the local EDFs it also checks that the header labels
equal the block labels.

Writes <root>/allowlist_summary.txt and, when nothing needs a person to look,
<root>/allowlist_chbmit.tsv, and prints the SHA-256 of the tsv. Reads text and
EDF headers only.

Stops (exit 1, no tsv, any old tsv in <root> removed) on a matcher error, on a
block that misses a target other than the 3 FT inputs, or on a header whose
labels differ from its block: those need a hand-reviewed decision, not an
automatic allow-list row (test plan section 4, rules 5 and 7).

Run with cwd C:\\EEG:
  C:/Users/user/anaconda3/python.exe -I experiments/v3/v3_allowlist.py [--root D:/EEG_v3]
      [--edf-root DIR ...]   EDF folders (chbNN/*.edf) for the header check; default raw_data
"""
import argparse
import hashlib
import re
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, r'C:\EEG')
sys.path.insert(0, str(Path(__file__).resolve().parent))
import v3_edf  # noqa: E402

RAW = Path(r'C:\EEG\raw_data')
N_CASES, N_FILES = 24, 676

BLOCK_RE = re.compile(r'^(Channels in EDF Files|Channels changed)\s*:')
CHAN_RE = re.compile(r'^Channel\s+(\d+)\s*:\s*(.*)$')
FILE_RE = re.compile(r'^File Name\s*:\s*(\S+)')


def parse_summary(path):
    """Blocks of one summary: list of dicts with line, kind, labels, files."""
    blocks, cur, reading = [], None, False
    for no, line in enumerate(path.read_text(encoding='latin-1').splitlines(), 1):
        s = line.strip()
        m = BLOCK_RE.match(s)
        if m:
            cur = {'case': path.parent.name, 'line': no, 'kind': m.group(1), 'labels': [], 'files': []}
            blocks.append(cur)
            reading = True
            continue
        m = CHAN_RE.match(s)
        if m:
            if not reading:
                raise ValueError('%s:%d channel line outside a block' % (path, no))
            if int(m.group(1)) != len(cur['labels']) + 1:
                raise ValueError('%s:%d channel number out of order' % (path, no))
            cur['labels'].append(m.group(2).strip())
            continue
        if reading and (s == '' or set(s) == {'*'}) and not cur['labels']:
            continue                                     # separator under the block title
        reading = False
        m = FILE_RE.match(s)
        if m:
            if cur is None:
                raise ValueError('%s:%d file before any channel block' % (path, no))
            cur['files'].append(m.group(1))
    return blocks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=r'D:\EEG_v3')
    ap.add_argument('--edf-root', action='append', default=None,
                    help='folder holding chbNN/*.edf for the header check (repeatable); default raw_data')
    args = ap.parse_args()
    edf_roots = [Path(p) for p in (args.edf_root or [str(RAW)])]
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)

    from download_and_convert import TARGET_CHANNELS
    assert TARGET_CHANNELS == v3_edf.TARGETS, 'target list differs from the repo'

    summaries = sorted(RAW.glob('chb*/chb*-summary.txt'))
    assert len(summaries) == N_CASES, len(summaries)
    blocks = []
    for p in summaries:
        blocks.extend(parse_summary(p))
    file_block = {}
    for bi, b in enumerate(blocks):
        for f in b['files']:
            assert f not in file_block, 'file named twice: %s' % f
            file_block[f] = bi
    assert len(file_block) == N_FILES, len(file_block)

    rows, errors, lines, non_ft_files = [], [], [], []
    lines.append('CHB-MIT allow-list from summary montage blocks')
    lines.append('%d summaries, %d blocks, %d files' % (len(summaries), len(blocks), len(file_block)))
    lines.append('')
    for bi, b in enumerate(blocks):
        res = v3_edf.match_targets(b['labels'])
        missing = [r for r in res if r['kind'] == 'missing']
        errs = [r for r in res if r['kind'] == 'error']
        built = [r for r in res if r['kind'] in ('direct', 'minus', 'derived')]
        b['missing'] = [r['target'] for r in missing]
        lines.append('block %d  %s  %s-summary.txt:%d  "%s"' % (bi, b['case'], b['case'], b['line'], b['kind']))
        lines.append('  files (%d): %s' % (len(b['files']), ' '.join(b['files'])))
        lines.append('  labels (%d): %s' % (len(b['labels']), ' | '.join(b['labels'])))
        lines.append('  buildable %d/23: direct %d, minus %d, derived %d'
                     % (len(built), sum(r['kind'] == 'direct' for r in res),
                        sum(r['kind'] == 'minus' for r in res), sum(r['kind'] == 'derived' for r in res)))
        lines.append('  missing (%d): %s' % (len(missing), ' '.join(b['missing']) or '-'))
        for r in errs:
            lines.append('  ERROR %s: %s' % (r['target'], r['reason']))
            errors.append('%s block %d %s: %s' % (b['case'], bi, r['target'], r['reason']))
        for f in b['files']:
            for r in missing:
                rows.append((f, r['target'], r['reason']))
            if any(t not in v3_edf.FT_TARGETS for t in b['missing']):
                non_ft_files.append(f)
        lines.append('')

    # header labels of the EDFs found under the EDF roots against their block
    local = sorted(p for r in edf_roots for p in r.glob('chb*/*.edf'))
    diffs, unlisted = [], []
    for p in local:
        if p.name not in file_block:
            unlisted.append(p.name)
            continue
        hl = v3_edf.read_header(p)['labels']
        bl = blocks[file_block[p.name]]['labels']
        if hl != bl:
            diffs.append('%s header %s vs block %s' % (p.name, hl, bl))

    n_allowed_files = len({r[0] for r in rows})
    lines.append('SUMMARY')
    lines.append('allow-listed rows %d in %d files' % (len(rows), n_allowed_files))
    lines.append('files whose block misses a target other than %s: %d'
                 % (', '.join(v3_edf.FT_TARGETS), len(non_ft_files)))
    for f in non_ft_files:
        lines.append('  %s: %s' % (f, ' '.join(blocks[file_block[f]]['missing'])))
    lines.append('matcher errors: %d' % len(errors))
    for e in errors:
        lines.append('  ' + e)
    lines.append('EDF roots %s' % ' '.join(str(r) for r in edf_roots))
    lines.append('EDFs %d: header labels differ from block in %d, not in any summary %d'
                 % (len(local), len(diffs), len(unlisted)))
    for d in diffs:
        lines.append('  DIFF ' + d)
    for u in unlisted:
        lines.append('  UNLISTED ' + u)

    tsv = root / 'allowlist_chbmit.tsv'
    stop = []
    if errors:
        stop.append('%d matcher errors' % len(errors))
    if non_ft_files:
        stop.append('%d files miss a target other than the 3 FT inputs' % len(non_ft_files))
    if diffs:
        stop.append('%d headers differ from their block' % len(diffs))
    if stop:
        if tsv.exists():
            tsv.unlink()                                 # never leave an old tsv behind
        lines.append('STOPPED, no allowlist_chbmit.tsv written: %s; needs a hand-reviewed decision'
                     % '; '.join(stop))
        (root / 'allowlist_summary.txt').write_text('\n'.join(lines) + '\n', encoding='ascii')
        for ln in lines[lines.index('SUMMARY'):]:
            print(ln)
        sys.exit(1)

    with open(tsv, 'w', encoding='ascii', newline='\n') as f:
        f.write('file\ttarget\treason\n')
        for r in rows:
            f.write('%s\t%s\t%s\n' % r)
    sha = hashlib.sha256(tsv.read_bytes()).hexdigest()
    lines.append('allowlist_chbmit.tsv SHA-256 %s' % sha)
    (root / 'allowlist_summary.txt').write_text('\n'.join(lines) + '\n', encoding='ascii')
    for ln in lines[lines.index('SUMMARY'):]:
        print(ln)
    print('SHA-256 %s  %s' % (sha, tsv))
    sys.exit(0)


if __name__ == '__main__':
    main()
