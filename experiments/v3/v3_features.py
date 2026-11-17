"""
v3 per-window features from the raw EDF, in two channel modes. CPU only.

  --mode old    exactly experiments/trees/trees_features.py: MNE read, the repo's
                find_matching_channel on raw.ch_names, zero-fill, float32.
  --mode exact  the exact matcher (v3_edf.match_targets) on the raw header labels;
                a target that is not matched or derived is zero only when it is in
                the allow-list, else the run stops naming file and target.

Both modes then do what trees_features.py does: 4 s windows, 2 s step, per-window
0.5-50 Hz filtfilt, and the same features() function, imported unchanged from
experiments/trees/trees_features.py. Labels are rebuilt from summary times on the
EDF sample axis; the stored .npz labels are kept as labels_stored.

--records local (default) reads the 152 local files. --records full reads every
RECORDS entry except EXCLUDED: a local file (and its v2 .npz) where there is one,
else the file in RAWFULL. --check-only runs the pre-flight (completeness and, in
exact mode, the channel map of every file), writes the log and stops.

Run with cwd C:\\EEG and plain python (not -I: v2's environment is the user site), e.g.
  C:/Users/user/anaconda3/python.exe experiments/v3/v3_features.py --mode exact --records full --out C:/EEG_v3/feats_full_exact
--out must be a new folder: files already there are overwritten.
Writes only into --out: <CASE>__<stem>_labeled.npz, chanmap.tsv, v3_features_log.txt.
"""
import argparse
import hashlib
import importlib.metadata as md
import os
import sys
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(r'C:\EEG')
HERE = Path(__file__).resolve().parent
# numba (used inside MNE) caches into --out, never next to its package; set before MNE loads
if __name__ in ('__main__', '__mp_main__'):
    _pre = argparse.ArgumentParser(add_help=False)
    _pre.add_argument('--out')
    _o = _pre.parse_known_args()[0].out
    if _o:
        os.environ['NUMBA_CACHE_DIR'] = str(Path(_o) / 'numba_cache')
TREES_CODE = REPO / 'experiments' / 'trees'
for p in (str(TREES_CODE), str(REPO), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402
import pywt  # noqa: E402
import scipy  # noqa: E402
from scipy import signal  # noqa: E402

import v3_edf  # noqa: E402
from edf_to_npz import (TARGET_CHANNELS, find_matching_channel, annotations_for,  # noqa: E402
                        SAMPLING_RATE, WINDOW, STEP, mne)
import trees_features as TF  # noqa: E402  (features, NAMES, CHUNK, ba used as is)

PREP = REPO / 'preprocessed_data'
RAW = REPO / 'raw_data'
CHB05_06 = Path(r'C:\EEG_archive\2026-11-12\b3889703-629c-4c5a-b583-69f722743ff4'
                r'\scratchpad\trees\dl_chb05_06\chb05_06.edf')
RAWFULL = Path(r'C:\chbmit_full')
# Test plan section 3, "Damaged file" (user's OK, 2026-11-16): never read
EXCLUDED = {'chb17/chb17b_69.edf'}
N_RECORDS = 686
ALLOWLIST = Path(r'C:\EEG_v3\allowlist_chbmit.tsv')
# SHA-256 of allowlist_chbmit.tsv as first written, 2026-11-14, before any model
# output was read; v3_check_loader.py fails on any other value
ALLOWLIST_SHA256 = 'fe8ba91263828d632759acd208f5c4827015647cfcf5ae187902f9b3b1edd760'
LABELS_NOTE = ("keys: 'labels' = labels rebuilt from summary times (both modes); "
               "'labels_stored' = the v2 .npz labels, which TREES feats call 'labels'")
DESIGNED_PAIRS = {(TARGET_CHANNELS.index('T8-P8-0'), TARGET_CHANNELS.index('T8-P8-1')),
                  (TARGET_CHANNELS.index('T7-P7'), TARGET_CHANNELS.index('P7-T7'))}
UNIT_SCALE = {'uV': 1e-6, 'UV': 1e-6, '\xb5V': 1e-6, 'mV': 1e-3, 'V': 1.0, 'nV': 1e-9}
CORR_CHUNK = 262144

assert TARGET_CHANNELS == v3_edf.TARGETS, 'target list differs from the repo'


class LoaderStop(Exception):
    pass


# ---------------------------------------------------------------- jobs and allow-list

def list_jobs(records='local'):
    """'local': the 152 local files, listed as trees_features.py lists them.
    'full': also every other RECORDS entry except EXCLUDED, read from RAWFULL (no .npz)."""
    jobs = []
    for c in sorted(d.name for d in PREP.iterdir() if d.is_dir()):
        for f in sorted((PREP / c).glob('*.npz')):
            stem = f.stem[:-len('_labeled')]
            edf = RAW / c.lower() / (stem + '.edf')
            if stem == 'chb05_06':
                edf = CHB05_06                       # the local copy is truncated
            assert edf.exists(), edf
            jobs.append({'case': c, 'stem': stem, 'edf': str(edf), 'npz': str(f)})
    if records == 'local':
        return jobs
    assert records == 'full', records
    rec = (RAWFULL / 'RECORDS').read_text(encoding='ascii').split()
    assert len(rec) == len(set(rec)) == N_RECORDS and EXCLUDED <= set(rec), 'unexpected RECORDS'
    have = {(j['case'], j['stem']) for j in jobs}
    for rel in rec:
        if rel in EXCLUDED:
            continue
        case_dir, name = rel.split('/')
        case, stem = case_dir.upper(), name[:-len('.edf')]
        if (case, stem) in have:
            continue
        edf = RAWFULL / case_dir / name
        assert edf.exists(), edf
        jobs.append({'case': case, 'stem': stem, 'edf': str(edf), 'npz': ''})
    jobs.sort(key=lambda j: (j['case'], j['stem']))
    assert len(jobs) == N_RECORDS - len(EXCLUDED), len(jobs)
    return jobs


def out_name(case, stem):
    return '%s__%s_labeled.npz' % (case, stem)


def load_allowlist(path):
    lines = Path(path).read_text(encoding='ascii').splitlines()
    assert lines[0] == 'file\ttarget\treason', 'bad allow-list header: %s' % path
    rows = set()
    for ln in lines[1:]:
        f, t, _ = ln.split('\t')
        rows.add((f, t))
    return rows


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------- channel map

def check_mne_names(labels, ch_names, where):
    """Header labels and MNE names agree in count and order; MNE only renames duplicates."""
    if len(labels) != len(ch_names):
        raise LoaderStop('%s: %d header labels but %d MNE channels' % (where, len(labels), len(ch_names)))
    count = Counter(labels)
    for i, (lab, name) in enumerate(zip(labels, ch_names)):
        if count[lab] == 1:
            ok = name == lab
        else:
            ok = name.startswith(lab + '-') and name[len(lab) + 1:].isalnum()
        if not ok:
            raise LoaderStop('%s: signal %d header label %r but MNE name %r' % (where, i, lab, name))


def exact_map(labels, fname, allow):
    """Exact matcher plus allow-list: 23 dicts; raises LoaderStop on step 7."""
    res = v3_edf.match_targets(labels)
    for r in res:
        if r['kind'] == 'error':
            raise LoaderStop('%s: target %s: %s' % (fname, r['target'], r['reason']))
        if r['kind'] == 'missing':
            if (fname, r['target']) not in allow:
                raise LoaderStop('%s: target %s not matched, derived or allow-listed (%s)'
                                 % (fname, r['target'], r['reason']))
            r['kind'] = 'missing-allowed'
            r['src'] = 'MISSING-ALLOWED'
    return res


def _build(edf_path, case, stem, mode, allow):
    """Read one EDF and build the 23 inputs as float32 (23, n), plus map and stats."""
    hdr = v3_edf.read_header(edf_path)
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    names = list(raw.ch_names)
    data = raw.get_data()
    del raw
    n = data.shape[1]
    ch = np.zeros((len(TARGET_CHANNELS), n), dtype=np.float32)
    kinds, srcs, ia, ib = [], [], [], []
    if mode == 'old':
        for i, target in enumerate(TARGET_CHANNELS):
            idx = find_matching_channel(target, names)
            if idx is not None:
                ch[i] = data[idx]                    # float64 to float32, as the whole-array cast
                kinds.append('old')
                srcs.append(names[idx])
                ia.append(idx)
            else:
                kinds.append('zero')
                srcs.append('ZERO')
                ia.append(-1)
            ib.append(-1)
    else:
        check_mne_names(hdr['labels'], names, edf_path)
        for i, r in enumerate(exact_map(hdr['labels'], stem + '.edf', allow)):
            k = r['kind']
            if k == 'direct':
                ch[i] = data[r['a']]
            elif k == 'minus':
                ch[i] = -data[r['a']]                # float64, then cast
            elif k == 'derived':
                ch[i] = data[r['a']] - data[r['b']]  # float64, then cast
            kinds.append(k)
            srcs.append(r['src'])
            ia.append(r['a'])
            ib.append(r['b'])

    # clipping: share of samples at the header physical min or max (report only)
    clip = {}
    for j in sorted({x for x in ia + ib if x >= 0}):
        scale = UNIT_SCALE.get(hdr['dimension'][j])
        dspan = hdr['digital_max'][j] - hdr['digital_min'][j]
        if scale is None or dspan == 0:
            clip[j] = float('nan')
            continue
        tol = 0.5 * abs(hdr['physical_max'][j] - hdr['physical_min'][j]) / dspan * scale
        x = data[j]
        at = (np.abs(x - hdr['physical_max'][j] * scale) <= tol) | (np.abs(x - hdr['physical_min'][j] * scale) <= tol)
        clip[j] = float(np.mean(at))
    del data
    return {'ch': ch, 'kind': kinds, 'src': srcs, 'a': ia, 'b': ib, 'hdr': hdr,
            'mne_names': names, 'clip': clip}


def build_channels(edf_path, case, stem, mode, allowlist=None):
    """(float32 array (23, n), chan_src) for one EDF; allowlist is a path or a set."""
    if mode == 'exact':
        allow = allowlist if isinstance(allowlist, set) else load_allowlist(allowlist or ALLOWLIST)
    else:
        allow = set()
    b = _build(edf_path, case, stem, mode, allow)
    return b['ch'], b['src']


def neighbour_corr(ch, built):
    """|r| > 0.98 between built, non-constant slots over the whole file, designed pairs excluded."""
    idx = [i for i in range(ch.shape[0]) if built[i] and ch[i].min() != ch[i].max()]
    if len(idx) < 2:
        return []
    n = ch.shape[1]
    mean = np.zeros(len(idx))
    for j in range(0, n, CORR_CHUNK):                    # two passes: means, then centred products
        mean += ch[idx, j:j + CORR_CHUNK].astype(np.float64).sum(1)
    mean /= n
    cov = np.zeros((len(idx), len(idx)))
    for j in range(0, n, CORR_CHUNK):
        x = ch[idx, j:j + CORR_CHUNK].astype(np.float64) - mean[:, None]
        cov += x @ x.T
    sd = np.sqrt(np.diag(cov))
    out = []
    for p in range(len(idx)):
        for q in range(p + 1, len(idx)):
            i, k = idx[p], idx[q]
            if (i, k) in DESIGNED_PAIRS or (k, i) in DESIGNED_PAIRS:
                continue
            r = cov[p, q] / (sd[p] * sd[q])
            if abs(r) > 0.98:
                out.append('%s~%s=%.3f' % (TARGET_CHANNELS[i], TARGET_CHANNELS[k], r))
    return out


# ---------------------------------------------------------------- one file

def one(job):
    case, stem, edf_path, npz_path, out_dir, mode, allow_path = (
        job['case'], job['stem'], job['edf'], job['npz'], job['out'], job['mode'], job['allow'])
    t0 = time.time()
    ok_c, on_disk, expected = v3_edf.complete(edf_path)
    if not ok_c:
        raise LoaderStop('%s: incomplete, %d bytes on disk, %d expected' % (edf_path, on_disk, expected))
    allow = load_allowlist(allow_path) if mode == 'exact' else set()
    b = _build(edf_path, case, stem, mode, allow)
    ch = b['ch']
    n_samples = ch.shape[1]
    starts = list(range(0, n_samples - WINDOW + 1, STEP))

    seizures = annotations_for(case.lower()).get(Path(edf_path).name, [])
    sample_labels = np.zeros(n_samples, dtype=np.int64)
    for s in seizures:
        lo = int(s['start'] * SAMPLING_RATE)
        hi = min(int(s['end'] * SAMPLING_RATE), n_samples)
        if hi > lo:
            sample_labels[lo:hi] = 1
    labels = np.array([1 if np.mean(sample_labels[s:s + WINDOW]) > 0.5 else 0
                       for s in starts], dtype=np.int64)

    ref_seg = ref_lab = None
    if npz_path and Path(npz_path).exists():
        with np.load(npz_path) as d:
            ref_lab = d['labels']
            if len(ref_lab) == len(starts):
                ref_seg = d['segments']
    b_, a_ = TF.ba()
    feats, worst = [], 0.0
    for j in range(0, len(starts), TF.CHUNK):
        st = starts[j:j + TF.CHUNK]
        W = np.stack([ch[:, s:s + WINDOW] for s in st])          # float32
        F = signal.filtfilt(b_, a_, W, axis=-1)
        if ref_seg is not None:
            mean = np.mean(F, axis=-1, keepdims=True)
            std = np.std(F, axis=-1, keepdims=True) + 1e-8
            Z = ((F - mean) / std).astype(np.float32)
            worst = max(worst, float(np.max(np.abs(Z - ref_seg[j:j + TF.CHUNK]))))
        feats.append(TF.features(F))
    feats = np.concatenate(feats)
    del ref_seg

    built = [k in ('old', 'direct', 'minus', 'derived') for k in b['kind']]
    flat_share = np.mean(np.isnan(feats[:, :, 0]), axis=0)
    flat = ['%s=%.2f' % (TARGET_CHANNELS[i], flat_share[i]) for i in range(len(TARGET_CHANNELS))
            if built[i] and flat_share[i] > 0.5]
    labs = b['hdr']['labels']
    clip = [('%s@%d=%.3f%%' % (labs[j], j, 100 * v)) if v == v else ('%s@%d=noscale' % (labs[j], j))
            for j, v in sorted(b['clip'].items()) if not v <= 0.001]
    corr = neighbour_corr(ch, built)
    del ch

    if ref_lab is None:
        nlab, status = -1, 'NO-STORED'
    elif len(ref_lab) != len(starts):
        nlab, status = -1, 'COUNT-MISMATCH'
    else:
        nlab = int(np.sum(labels != ref_lab))
        status = 'OK' if worst < 1e-4 else 'DZ-DIFF'

    save = {'feats': feats, 'labels': labels.astype(np.int8), 'names': np.array(TF.NAMES),
            'chan_src': np.array(b['src']), 'chan_kind': np.array(b['kind']),
            'chan_a': np.array(b['a'], dtype=np.int32), 'chan_b': np.array(b['b'], dtype=np.int32),
            'raw_labels': np.array(labs), 'mne_names': np.array(b['mne_names']),
            'mode': np.array(mode), 'edf': np.array(edf_path)}
    if ref_lab is not None and len(ref_lab) == len(starts):
        save['labels_stored'] = ref_lab.astype(np.int8)
    out_path = Path(out_dir) / out_name(case, stem)
    tmp = out_path.with_name(out_path.stem + '.tmp.npz')
    np.savez(tmp, **save)
    os.replace(tmp, out_path)

    k = b['kind']
    line = ('%-6s %-24s windows %6d/%6s  max|dz| %s  label-diffs %s  complete %s (%d B)  '
            'derived %d  minus %d  missing-allowed %d  zero %d  flat [%s]  clip [%s]  corr [%s]  %4.0f s  %s'
            % (case, stem, len(starts), len(ref_lab) if ref_lab is not None else '-',
               '%.2e' % worst if status in ('OK', 'DZ-DIFF') else '-',
               nlab if nlab >= 0 else '-', 'yes' if ok_c else 'NO', on_disk,
               k.count('derived'), k.count('minus'), k.count('missing-allowed'), k.count('zero'),
               ' '.join(flat), ' '.join(clip), ' '.join(corr), time.time() - t0, status))
    rows = ['%s\t%d\t%s\t%s\t%s\t%d\t%d' % (out_name(case, stem)[:-4], i, TARGET_CHANNELS[i], k[i],
                                            b['src'][i], b['a'][i], b['b'][i])
            for i in range(len(TARGET_CHANNELS))]
    return {'case': case, 'stem': stem, 'line': line, 'status': status, 'rows': rows,
            'isolated': sys.flags.isolated}


# ---------------------------------------------------------------- main

def versions():
    def meta(name):
        try:
            return md.version(name)
        except md.PackageNotFoundError:
            return 'none'
    return ['python %s' % sys.executable,
            'python version %s' % sys.version.split()[0],
            'isolated (-I) %d' % sys.flags.isolated,
            'numpy module %s metadata %s' % (np.__version__, meta('numpy')),
            'scipy module %s metadata %s' % (scipy.__version__, meta('scipy')),
            'mne module %s metadata %s' % (mne.__version__, meta('mne')),
            'pywt module %s metadata(PyWavelets) %s file %s' % (pywt.__version__, meta('PyWavelets'), pywt.__file__),
            'features() from %s sha256 %s' % (TF.__file__, sha256(TF.__file__))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--mode', required=True, choices=('old', 'exact'))
    ap.add_argument('--procs', type=int, default=4)
    ap.add_argument('--only', default='', help='comma list of stems, e.g. chb12_27,chb13_40')
    ap.add_argument('--cases', default='', help='comma list of cases, e.g. CHB01,CHB12')
    ap.add_argument('--allowlist', default=str(ALLOWLIST))
    ap.add_argument('--records', default='local', choices=('local', 'full'))
    ap.add_argument('--check-only', action='store_true', help='pre-flight only, no extraction')
    args = ap.parse_args()
    if Path.cwd().resolve() != REPO.resolve():
        sys.exit('run with cwd C:\\EEG (annotations_for reads raw_data relative to it)')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    jobs = list_jobs(args.records)
    if args.cases:
        want = {c.strip().upper() for c in args.cases.split(',') if c.strip()}
        jobs = [j for j in jobs if j['case'] in want]
    if args.only:
        want = {s.strip() for s in args.only.split(',') if s.strip()}
        unknown = want - {j['stem'] for j in jobs}
        if unknown:
            sys.exit('--only names not in the job list: %s' % ' '.join(sorted(unknown)))
        jobs = [j for j in jobs if j['stem'] in want]

    head = versions() + ['mode %s' % args.mode, 'args %s' % vars(args), 'jobs %d' % len(jobs), LABELS_NOTE]
    if args.records == 'full':
        head.append('records full: %d local, %d from %s; excluded %s'
                    % (sum(1 for j in jobs if j['npz']), sum(1 for j in jobs if not j['npz']), RAWFULL,
                       sorted(EXCLUDED)))
    allow = set()
    if args.mode == 'exact':
        allow = load_allowlist(args.allowlist)
        sha = sha256(args.allowlist)
        head.append('allow-list %s sha256 %s rows %d, pinned value %s'
                    % (args.allowlist, sha, len(allow), 'matches' if sha == ALLOWLIST_SHA256 else
                       'DIFFERS (pinned %s)' % ALLOWLIST_SHA256))
    log_path = out / 'v3_features_log.txt'

    def write_log(lines):
        log_path.write_text('\n'.join(lines) + '\n', encoding='ascii', errors='backslashreplace')

    # pre-flight: completeness of every file, and in exact mode the full channel map
    problems = []
    for j in jobs:
        ok_c, on_disk, expected = v3_edf.complete(j['edf'])
        if not ok_c:
            problems.append('INCOMPLETE %s: %d bytes on disk, %d expected' % (j['edf'], on_disk, expected))
        if args.mode == 'exact':
            try:
                exact_map(v3_edf.read_header(j['edf'])['labels'], j['stem'] + '.edf', allow)
            except LoaderStop as e:
                problems.append('STOP %s' % e)
            except ValueError as e:                  # header unreadable, already INCOMPLETE
                problems.append('STOP %s: header unreadable: %s' % (j['edf'], e))
    if problems:
        write_log(head + problems + ['stopped before extraction: %d problems' % len(problems)])
        print('\n'.join(problems))
        sys.exit(1)
    head.append('pre-flight: %d files complete%s' % (len(jobs), ', channel maps OK' if args.mode == 'exact' else ''))
    for h in head:
        print(h, flush=True)
    if args.check_only:
        write_log(head + ['check only: stopped before extraction'])
        print('check only: stopped before extraction')
        sys.exit(0)

    for j in jobs:
        j.update(out=str(out), mode=args.mode, allow=args.allowlist)
    t0, results = time.time(), []
    try:
        with Pool(args.procs) as pool:
            for r in pool.imap_unordered(one, jobs):
                results.append(r)
                print(r['line'], flush=True)
    except Exception as e:
        lines = sorted(r['line'] for r in results)
        write_log(head + lines + ['STOPPED: %s: %s' % (type(e).__name__, e)])
        print('STOPPED: %s: %s' % (type(e).__name__, e))
        sys.exit(1)

    results.sort(key=lambda r: (r['case'], r['stem']))
    bad = [r for r in results if r['status'] in ('COUNT-MISMATCH',)
           or (args.mode == 'old' and r['status'] == 'DZ-DIFF')]
    tally = Counter(r['status'] for r in results)
    lines = head + ['workers isolated (-I): %s' % sorted({r['isolated'] for r in results})]
    lines += [r['line'] for r in results]
    lines.append('%d recordings, status %s, %d not OK, %.1f min'
                 % (len(results), dict(sorted(tally.items())), len(bad), (time.time() - t0) / 60))
    write_log(lines)
    with open(out / 'chanmap.tsv', 'w', encoding='ascii', errors='backslashreplace', newline='\n') as f:
        f.write('file\tslot\ttarget\tkind\tchan_src\ta\tb\n')
        for r in results:
            f.write('\n'.join(r['rows']) + '\n')
    print(lines[-1])
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
