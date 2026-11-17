"""
v3 loader checks before any result is read (test plan sections 4 and 9).

  C2(i)    old mode reproduces TREES feats on the first unaffected file of CHB01, CHB12, CHB24
  C2(ii)   old vs exact mode: only the 6 known files change; exact feats of the others vs TREES;
           old feats of the 6 vs TREES (old mode is v2's matcher there too)
  C2(iii)  the 6 rebuilt files, checked from MNE data and against epilepsy2bids (chb12_27-29)
  C4       rebuilt labels differ from the stored ones only in chb03_01, chb03_02, chb05_06, chb05_13
  complete all read EDFs complete; allow  local allow-list rows are the 3 FT inputs of the 6 files,
           and the allow-list SHA-256 equals the pinned value

Reads finished v3_features.py folders (--old, --exact); C2(iii) reads the EDFs.
Writes --out only and exits 1 on any FAIL (3 when nothing failed but something is UNRUN).
INFO lines are never counted. With --partial only the files present in the folders are
checked (for small tests); a C2(i) file absent then counts as UNRUN.

Run with cwd C:\\EEG:
  C:/Users/user/anaconda3/python.exe -I experiments/v3/v3_check_loader.py --old D:/EEG_v3/feats_old --exact D:/EEG_v3/feats_exact
"""
import argparse
import contextlib
import io
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
# numba (via epilepsy2bids) must not cache next to its package; set before any import of it
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument('--out', default=r'D:\EEG_v3\checks\loader_checks.txt')
os.environ['NUMBA_CACHE_DIR'] = str(Path(_pre.parse_known_args()[0].out).parent / 'numba_cache')
for p in (r'C:\EEG', str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402

import v3_edf  # noqa: E402
import v3_features as VF  # noqa: E402

AFFECTED = ['chb12_27', 'chb12_28', 'chb12_29', 'chb13_40', 'chb16_18', 'chb19_01']
E2B_FILES = ['chb12_27', 'chb12_28', 'chb12_29']
C4_FILES = ['chb03_01', 'chb03_02', 'chb05_06', 'chb05_13']
C2I_CASES = ['CHB01', 'CHB12', 'CHB24']
TREES_FEATS = Path(r'C:\EEG_archive\2026-11-12\b3889703-629c-4c5a-b583-69f722743ff4'
                   r'\scratchpad\trees\feats')
ENV_TOL = 1e-6
E2B_TOL = 1e-6
N_LOCAL = 152
SYN = {'T3': 'T7', 'T4': 'T8', 'T5': 'P7', 'T6': 'P8'}


class Report:
    def __init__(self):
        self.lines, self.fail, self.unrun = [], 0, 0

    def add(self, s=''):
        self.lines.append(s)
        print(s, flush=True)

    def result(self, name, status, detail=''):
        self.fail += status == 'FAIL'
        self.unrun += status == 'UNRUN'
        self.add('%-10s %-8s %s' % (name, status, detail))


def compare_feats(a, b):
    """('PASS'|'PASS-ENV'|'FAIL', max relative difference, note)."""
    if a.shape != b.shape:
        return 'FAIL', float('nan'), 'shape %s vs %s' % (a.shape, b.shape)
    if np.array_equal(a, b, equal_nan=True):
        return 'PASS', 0.0, 'array_equal'
    na, nb = np.isnan(a), np.isnan(b)
    if not np.array_equal(na, nb):
        return 'FAIL', float('nan'), 'NaN pattern differs in %d cells' % int(np.sum(na != nb))
    x, y = a[~na].astype(np.float64), b[~nb].astype(np.float64)
    if not np.array_equal(np.isinf(x), np.isinf(y)) or not np.array_equal(x[np.isinf(x)], y[np.isinf(y)]):
        return 'FAIL', float('nan'), 'inf pattern differs'
    fin = np.isfinite(x)
    x, y = x[fin], y[fin]
    den = np.maximum(np.abs(x), np.abs(y))
    diff = np.abs(x - y)
    rel = np.where(den > 0, diff / np.where(den > 0, den, 1), 0.0)
    m = float(rel.max()) if rel.size else 0.0
    ncell = int(np.sum(diff > 0))
    if m <= ENV_TOL:
        return 'PASS-ENV', m, 'not equal, %d cells differ, max rel %.3e <= %.0e' % (ncell, m, ENV_TOL)
    return 'FAIL', m, '%d cells differ, max rel %.3e > %.0e; %s' % (ncell, m, ENV_TOL, by_feature(a, b))


def by_feature(a, b):
    """Diagnosis only: per feature, cells over the tolerance, their largest |value|, max abs diff."""
    out, rest = [], 0.0
    for k, name in enumerate(VF.TF.NAMES):
        x, y = a[..., k].astype(np.float64), b[..., k].astype(np.float64)
        f = np.isfinite(x) & np.isfinite(y)
        x, y = x[f], y[f]
        den = np.maximum(np.abs(x), np.abs(y))
        d = np.abs(x - y)
        rel = np.where(den > 0, d / np.where(den > 0, den, 1), 0.0)
        over = rel > ENV_TOL
        if over.any():
            out.append('%s %d cells over (|value| <= %.1e there), max rel %.1e, max abs %.1e'
                       % (name, int(over.sum()), float(den[over].max()), float(rel.max()), float(d.max())))
        elif rel.size:
            rest = max(rest, float(rel.max()))
    return 'by feature: %s; all other features max rel %.2e' % ('; '.join(out), rest)


def load(folder, case, stem):
    p = Path(folder) / VF.out_name(case, stem)
    if not p.exists():
        return None
    with np.load(p) as d:
        return {k: d[k] for k in d.files}


def trees(case, stem):
    with np.load(TREES_FEATS / VF.out_name(case, stem)) as d:
        return d['feats'], d['labels']


def vs_trees(rep, tag, case, stem, mine):
    tf, tl = trees(case, stem)
    st, m, note = compare_feats(mine['feats'], tf)
    if 'labels_stored' not in mine or not np.array_equal(mine['labels_stored'].astype(np.int64), tl.astype(np.int64)):
        st, note = 'FAIL', note + '; labels_stored differ from TREES labels'
    rep.result(tag, st, '%s vs TREES: %s' % (stem, note))
    return st, m


# ---------------------------------------------------------------- C2(i)

def c2i(rep, args, jobs):
    rep.add('--- C2(i): old mode vs TREES feats, first unaffected file of %s' % ', '.join(C2I_CASES))
    for c in C2I_CASES:
        stem = next(j['stem'] for j in jobs if j['case'] == c and j['stem'] not in AFFECTED)
        mine = load(args.old, c, stem)
        if mine is None:
            rep.result('C2(i)', 'FAIL' if not args.partial else 'UNRUN', '%s not in %s' % (stem, args.old))
            continue
        vs_trees(rep, 'C2(i)', c, stem, mine)


# ---------------------------------------------------------------- C2(ii)

def c2ii(rep, args, jobs):
    rep.add('--- C2(ii): old vs exact mode; only %s may change' % ', '.join(AFFECTED))
    changed, n, tally, worst, tally_old = [], 0, {}, 0.0, {}
    for j in jobs:
        o, e = load(args.old, j['case'], j['stem']), load(args.exact, j['case'], j['stem'])
        if o is None or e is None:
            if not args.partial:
                rep.result('C2(ii)', 'FAIL', '%s missing from %s' % (j['stem'], 'old' if o is None else 'exact'))
            continue
        n += 1
        same_map = (all(k == 'old' for k in o['chan_kind']) and all(k == 'direct' for k in e['chan_kind'])
                    and np.array_equal(o['chan_a'], e['chan_a']))
        same_feats = np.array_equal(o['feats'], e['feats'], equal_nan=True)
        same_labels = np.array_equal(o['labels'], e['labels'])
        if not same_labels:
            rep.result('C2(ii)', 'FAIL', '%s labels differ between modes' % j['stem'])
        if not same_map:
            changed.append(j['stem'])
        if j['stem'] in AFFECTED:
            if same_map or same_feats:
                rep.result('C2(ii)', 'FAIL', '%s left the changed set (map same %s, feats same %s)'
                           % (j['stem'], same_map, same_feats))
            else:
                diff = ['%s: %s -> %s' % (VF.TARGET_CHANNELS[i], o['chan_src'][i], e['chan_src'][i])
                        for i in range(len(VF.TARGET_CHANNELS)) if o['chan_src'][i] != e['chan_src'][i]]
                rep.result('C2(ii)', 'PASS', '%s changed as expected, %d slots: %s'
                           % (j['stem'], len(diff), '; '.join(diff)))
            # old mode must still be v2 on the files the fix changes
            st, m = vs_trees(rep, 'C2(ii)old', j['case'], j['stem'], o)
            tally_old[st] = tally_old.get(st, 0) + 1
            continue
        if not same_map or not same_feats:
            rep.result('C2(ii)', 'FAIL', '%s entered the changed set (map same %s, feats same %s)'
                       % (j['stem'], same_map, same_feats))
            continue
        st, m = vs_trees(rep, 'C2(ii)', j['case'], j['stem'], e)
        tally[st] = tally.get(st, 0) + 1
        worst = max(worst, m) if m == m else worst
    want = [s for s in AFFECTED if any(j['stem'] == s for j in jobs)]
    if not args.partial and n != N_LOCAL:
        rep.result('C2(ii)', 'FAIL', 'compared %d files, expected %d' % (n, N_LOCAL))
    present_aff = [s for s in want if exists(args.old, case_of(jobs, s)) and exists(args.exact, case_of(jobs, s))]
    st = 'PASS' if sorted(changed) == sorted(present_aff) else 'FAIL'
    rep.result('C2(ii)', st, 'changed set %s; expected %s; %d files compared; unaffected vs TREES %s, max rel %.3e; '
               'affected, old mode vs TREES %s'
               % (sorted(changed), sorted(present_aff), n, tally, worst, tally_old))


def case_of(jobs, stem):
    return next(j for j in jobs if j['stem'] == stem)


def exists(folder, job):
    return (Path(folder) / VF.out_name(job['case'], job['stem'])).exists()


# ---------------------------------------------------------------- C2(iii)

def chk_norm(label):
    """Check-side name normaliser, kept apart from the matcher on purpose."""
    s = label.upper().replace(' ', '')
    if s.startswith('EEG'):
        s = s[3:]
    if s == '01':
        s = 'O1'
    if s.endswith('-CS2'):
        s = s[:-4]
    return '-'.join(SYN.get(t, t) for t in s.split('-'))


def chk_ref(label):
    """Reference form of one raw label: 'bare' (no '-'), '-CS2', or 'other'."""
    s = label.upper().replace(' ', '')
    if s.startswith('EEG'):
        s = s[3:]
    if '-' not in s:
        return 'bare'
    if s.endswith('-CS2') and s.count('-') == 1:
        return '-CS2'
    return 'other'


def c2iii(rep, args, jobs):
    rep.add('--- C2(iii): rebuilt slots of the 6 files from MNE data; chb12_27-29 vs epilepsy2bids')
    allow = VF.load_allowlist(args.allowlist)
    T = VF.TARGET_CHANNELS
    i_t7p7, i_p7t7, i_t0, i_t1 = T.index('T7-P7'), T.index('P7-T7'), T.index('T8-P8-0'), T.index('T8-P8-1')
    for stem in AFFECTED:
        j = next(j for j in jobs if j['stem'] == stem)
        ch, src = VF.build_channels(j['edf'], j['case'], stem, 'exact', allow)
        raw = VF.mne.io.read_raw_edf(j['edf'], preload=True, verbose=False)
        names = list(raw.ch_names)
        d = raw.get_data()
        del raw
        errs, kinds = [], {'direct': 0, 'derived': 0, 'minus': 0, 'missing': 0}

        def idx(label):
            if '#' in label:
                lab, k = label.rsplit('#', 1)
                name = '%s-%s' % (lab, k)
            else:
                name = label
            hits = [i for i, x in enumerate(names) if x == name]
            if len(hits) != 1:
                raise ValueError('%s: MNE name %r found %d times' % (stem, name, len(hits)))
            return hits[0]

        for i, t in enumerate(T):
            A, B = v3_edf.target_pair(t)
            s = str(src[i])
            try:
                if s == 'MISSING-ALLOWED':
                    kinds['missing'] += 1
                    if t not in v3_edf.FT_TARGETS:
                        errs.append('%s missing but not an FT input' % t)
                    if (stem + '.edf', t) not in allow:
                        errs.append('%s missing but not allow-listed' % t)
                    if np.any(ch[i] != 0):
                        errs.append('%s missing but not zero' % t)
                elif s.startswith('minus '):
                    kinds['minus'] += 1
                    a = idx(s[6:])
                    if t != 'P7-T7' or chk_norm(s[6:]) != 'T7-P7':
                        errs.append('%s: unexpected minus source %s' % (t, s))
                    if not np.array_equal(ch[i], (-d[a]).astype(np.float32)):
                        errs.append('%s != -(%s) cast to float32' % (t, s[6:]))
                elif ' minus ' in s:
                    kinds['derived'] += 1
                    la, lb = s.split(' minus ')
                    a, b = idx(la), idx(lb)
                    if chk_norm(la) != A or chk_norm(lb) != B:
                        errs.append('%s: sources %s, %s are not electrodes %s, %s' % (t, la, lb, A, B))
                    if chk_ref(la) != chk_ref(lb) or chk_ref(la) == 'other':
                        errs.append('%s: sources %s, %s do not share one reference (%s, %s)'
                                    % (t, la, lb, chk_ref(la), chk_ref(lb)))
                    if not np.array_equal(ch[i], (d[a] - d[b]).astype(np.float32)):
                        errs.append('%s != (%s - %s) cast to float32' % (t, la, lb))
                else:
                    kinds['direct'] += 1
                    a = idx(s)
                    if chk_norm(s.split('#')[0]) != A + '-' + B:
                        errs.append('%s: direct source %s has another name' % (t, s))
                    if not np.array_equal(ch[i], d[a].astype(np.float32)):
                        errs.append('%s != %s cast to float32' % (t, s))
                if s != 'MISSING-ALLOWED' and not np.any(ch[i] != 0):
                    errs.append('%s built but all zero' % t)
            except ValueError as e:
                errs.append(str(e))
        # both T8-P8 slots equal the file's T8-P8 (the label if present, else the derived one)
        if not np.array_equal(ch[i_t0], ch[i_t1]):
            errs.append('T8-P8-0 != T8-P8-1')
        t8 = [k for k, x in enumerate(names) if x == 'T8-P8']
        if len(t8) == 1 and not np.array_equal(ch[i_t0], d[t8[0]].astype(np.float32)):
            errs.append('T8-P8 slots != the EDF T8-P8')
        p7_ok = np.array_equal(ch[i_p7t7], -ch[i_t7p7])
        p7 = 'P7-T7 = -(T7-P7) exactly: %s' % p7_ok
        if not p7_ok:
            errs.append('P7-T7 slot != -(T7-P7 slot)')
        ft = [T[i] for i in range(len(T)) if not np.any(ch[i] != 0)]
        if sorted(ft) != sorted(v3_edf.FT_TARGETS):
            errs.append('zero slots %s, expected exactly %s' % (ft, list(v3_edf.FT_TARGETS)))
        rep.result('C2(iii)', 'FAIL' if errs else 'PASS',
                   '%s: direct %d, derived %d, minus %d, missing-allowed %d; zero slots %s; T8-P8 %s; %s%s'
                   % (stem, kinds['direct'], kinds['derived'], kinds['minus'], kinds['missing'], ft,
                      'single label' if len(t8) == 1 else 'derived (no T8-P8 label)', p7,
                      ('; ' + '; '.join(errs)) if errs else ''))
        if stem in E2B_FILES:
            e2b(rep, args, j, ch)
        del d, ch


def e2b(rep, args, j, ch):
    """chb12_27-29 against epilepsy2bids, as chbmit_convert2bids.py:57-60 loads them."""
    stem = j['stem']
    try:
        from epilepsy2bids.eeg import Eeg
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            eeg = Eeg.loadEdf(j['edf'], Eeg.Montage.UNIPOLAR, Eeg.ELECTRODES_10_20)
            eeg.standardize(256, Eeg.ELECTRODES_10_20, 'bipolar')
    except Exception as e:  # noqa: BLE001
        rep.result('C2(iii)e2b', 'UNRUN', '%s: epilepsy2bids could not load it: %s: %s'
                   % (stem, type(e).__name__, e))
        return
    said = buf.getvalue().strip().replace('\n', ' | ')
    T = VF.TARGET_CHANNELS
    banana = set(T[:18])                                     # the 18 double-banana slots
    errs, parts, worst, seen = [], [], 0.0, []
    if len(eeg.channels) != 18:
        errs.append('epilepsy2bids gave %d channels, expected 18' % len(eeg.channels))
    if eeg.data.shape[1] != ch.shape[1]:
        errs.append('length %d vs %d' % (eeg.data.shape[1], ch.shape[1]))
    else:
        for k, name in enumerate(eeg.channels):
            try:
                a, b = name.split('-')
                tgt = '%s-%s' % (SYN.get(a.upper(), a.upper()), SYN.get(b.upper(), b.upper()))
                slot = T.index('T8-P8-0') if tgt == 'T8-P8' else T.index(tgt)
            except ValueError:
                errs.append('epilepsy2bids channel %r pairs with no target' % name)
                continue
            if T[slot] not in banana:
                errs.append('epilepsy2bids channel %r pairs with %s, not a double-banana slot' % (name, T[slot]))
                continue
            seen.append(T[slot])
            ref = eeg.data[k].astype(np.float64)                 # uV
            mine = ch[slot].astype(np.float64) * 1e6             # V to uV
            mx = float(np.max(np.abs(ref)))
            dif = float(np.max(np.abs(mine - ref)))
            ratio = dif / mx if mx > 0 else float('inf')
            worst = max(worst, ratio)
            parts.append('%s->%s %.2e/%.0f' % (name, T[slot], dif, mx))
            if mx == 0:
                errs.append('%s is all zero in epilepsy2bids' % name)
            elif ratio > E2B_TOL:
                errs.append('%s max diff %.3e uV = %.3e of max %.1f uV' % (name, dif, ratio, mx))
        if len(seen) != len(set(seen)) or set(seen) != banana:
            errs.append('compared slots %s, expected each of the 18 double-banana slots once' % sorted(seen))
    rep.result('C2(iii)e2b', 'FAIL' if errs else 'PASS',
               '%s: %d channels, max |diff|/max|value| %.3e (tol %.0e); epilepsy2bids said: %s; '
               'pairs name->slot maxdiff_uV/max_uV: %s%s'
               % (stem, len(eeg.channels), worst, E2B_TOL, said or 'nothing', ', '.join(parts),
                  ('; ' + '; '.join(errs)) if errs else ''))


# ---------------------------------------------------------------- C4, completeness, allow-list

def c4(rep, args, jobs):
    rep.add('--- C4: rebuilt labels vs stored labels; only %s may differ, by 1 window' % ', '.join(C4_FILES))
    seen, diffs, n = [], {}, 0
    for j in jobs:
        e = load(args.exact, j['case'], j['stem'])
        if e is None:
            if not args.partial:
                rep.result('C4', 'FAIL', '%s missing from %s' % (j['stem'], args.exact))
            continue
        n += 1
        seen.append(j['stem'])
        if 'labels_stored' not in e:
            rep.result('C4', 'FAIL', '%s has no labels_stored (window count mismatch?)' % j['stem'])
            continue
        k = int(np.sum(e['labels'] != e['labels_stored']))
        if k:
            diffs[j['stem']] = k
    want = {s: 1 for s in C4_FILES if s in seen}
    st = 'PASS' if diffs == want else 'FAIL'
    rep.result('C4', st, '%d files; differing %s; expected %s%s'
               % (n, diffs, want, '' if n == N_LOCAL else ' (partial: %s absent)'
                  % [s for s in C4_FILES if s not in seen]))


def completeness(rep, args, jobs, present):
    rep.add('--- Completeness: every read EDF complete')
    bad, n = [], 0
    for j in jobs:
        if args.partial and j['stem'] not in present:
            continue
        n += 1
        ok, on_disk, exp = v3_edf.complete(j['edf'])
        if not ok:
            bad.append('%s %d/%d' % (j['edf'], on_disk, exp))
    rep.result('complete', 'FAIL' if bad or (not args.partial and n != N_LOCAL) else 'PASS',
               '%d files read, %d incomplete %s' % (n, len(bad), bad))
    # information only: the local chb05_06 is truncated today; a complete copy later is no fault
    ctrl = Path(r'C:\EEG\raw_data\chb05\chb05_06.edf')
    if ctrl.exists():
        ok, on_disk, exp = v3_edf.complete(ctrl)
        rep.result('complete', 'INFO', 'control, not counted: local %s complete %s (%d of %d B)'
                   % (ctrl, ok, on_disk, exp))


def allow_check(rep, args, jobs):
    rep.add('--- Allow-list: local files with allow-listed targets')
    allow = VF.load_allowlist(args.allowlist)
    local = {j['stem'] + '.edf' for j in jobs}
    got = {}
    for f, t in allow:
        if f in local:
            got.setdefault(f[:-4], set()).add(t)
    ok = sorted(got) == sorted(AFFECTED) and all(v == set(v3_edf.FT_TARGETS) for v in got.values())
    rep.result('allow', 'PASS' if ok else 'FAIL', '%s: local files %s'
               % (args.allowlist, {k: sorted(v) for k, v in sorted(got.items())}))
    sha = VF.sha256(args.allowlist)
    rep.result('allow', 'PASS' if sha == args.allow_sha256 else 'FAIL',
               'sha256 %s, pinned %s' % (sha, args.allow_sha256))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--old', default=r'D:\EEG_v3\feats_old')
    ap.add_argument('--exact', default=r'D:\EEG_v3\feats_exact')
    ap.add_argument('--allowlist', default=str(VF.ALLOWLIST))
    ap.add_argument('--allow-sha256', default=VF.ALLOWLIST_SHA256,
                    help='expected SHA-256 of the allow-list (pinned before any model output was read)')
    ap.add_argument('--out', default=r'D:\EEG_v3\checks\loader_checks.txt')
    ap.add_argument('--parts', default='c2i,c2ii,c2iii,c4,complete,allow')
    ap.add_argument('--partial', action='store_true', help='check only the files present (small tests)')
    args = ap.parse_args()
    if Path.cwd().resolve() != VF.REPO.resolve():
        sys.exit('run with cwd C:\\EEG')
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    parts = [p.strip() for p in args.parts.split(',') if p.strip()]
    jobs = VF.list_jobs()
    assert len(jobs) == N_LOCAL, len(jobs)
    present = {j['stem'] for j in jobs if exists(args.old, j) or exists(args.exact, j)}

    rep = Report()
    for h in VF.versions():
        rep.add(h)
    rep.add('args %s' % vars(args))
    rep.add('files present: %d (%s)' % (len(present), 'partial' if args.partial else 'full'))
    if 'c2i' in parts:
        c2i(rep, args, jobs)
    if 'c2ii' in parts:
        c2ii(rep, args, jobs)
    if 'c2iii' in parts:
        c2iii(rep, args, jobs)
    if 'c4' in parts:
        c4(rep, args, jobs)
    if 'complete' in parts:
        completeness(rep, args, jobs, present)
    if 'allow' in parts:
        allow_check(rep, args, jobs)
    verdict = 'FAIL' if rep.fail else ('UNRUN items' if rep.unrun else 'PASS')
    rep.add('OVERALL %s: %d FAIL, %d UNRUN, parts %s%s'
            % (verdict, rep.fail, rep.unrun, parts,
               ' (PARTIAL: %d of %d files)' % (len(present), N_LOCAL) if args.partial else ''))
    Path(args.out).write_text('\n'.join(rep.lines) + '\n', encoding='ascii', errors='backslashreplace')
    sys.exit(1 if rep.fail else (3 if rep.unrun else 0))


if __name__ == '__main__':
    main()
