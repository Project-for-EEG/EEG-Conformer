"""
v3 check C2(ii) on full records (test plan section 4): old mode vs exact mode, same run.

  Plan: "files outside the 31 NOMAP files keep identical channel maps and feats;
  a file entering or leaving that list stops the work." Here that is a FAIL.

Reads two finished `v3_features.py --records full` folders (--old, --exact): every
<CASE>__<stem>_labeled.npz, chanmap.tsv and v3_features_log.txt. Reads no EDF data
and changes nothing in either folder. Writes --out only, and refuses to overwrite it.
The job list (v3_features.list_jobs) checks that every EDF of the feature run is
still in place, so the checker stops if one has moved.

Checks, each PASS or FAIL (INFO lines are never counted):
  setup     the 31 NOMAP stems (plan D4) equal the files of the pinned allow-list
  logs      both logs: same versions and features() hash, full records, 685 jobs,
            0 not OK; per-file case and status: OK / NO-STORED (old), DZ-DIFF only
            on the 6 local NOMAP files (exact); exact log says the allow-list
            SHA-256 matches. The hashes of the loader code are INFO only: the logs
            do not record them.
  files     each folder holds exactly the 685 expected .npz, no .tmp.npz left
  C2(ii)    per file: same EDF, labels and window count in both modes, labels as
            long as feats; each stored map equals its matcher re-run on the stored
            labels (old: find_matching_channel on the MNE names; exact: the exact
            matcher and allow-list on the header labels); outside NOMAP identical
            map and feats; inside NOMAP map and feats both differ; window total
            equals the header pre-check
  NOMAP     per NOMAP file, against the header pre-check: exact kinds and the set of
            slots whose map entry changed are as predicted; slots with an unchanged
            source, or zero in both modes, keep identical feats; both T8-P8 slots are one signal; P7-T7 is
            T7-P7 reversed; the 3 FT slots are allow-listed and all NaN. Rebuilt
            slots over half NaN, and P7-T7 feats against T7-P7's, are INFO only
            (plan rule 6: a flat channel is reported, not a stop).
  chanmap   chanmap.tsv agrees with the .npz channel fields

Exit 0 PASS, 1 any FAIL, 3 no FAIL but --only made it partial.

Run with cwd C:\\EEG and plain python (not -I: v2's environment is the user site):
  C:/Users/user/anaconda3/python.exe experiments/v3/v3_check_full.py --old C:/EEG_v3/feats_full_old --exact C:/EEG_v3/feats_full_exact
"""
import argparse
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
OUT_DEFAULT = r'C:\EEG_v3\checks\full_c2ii.txt'
# numba (inside MNE, imported by v3_features) must not cache next to its package
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument('--out', default=OUT_DEFAULT)
os.environ['NUMBA_CACHE_DIR'] = str(Path(_pre.parse_known_args()[0].out).parent / 'numba_cache')
for p in (r'C:\EEG', str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402

import v3_edf  # noqa: E402
import v3_features as VF  # noqa: E402

# Test plan D4 (test_plan_v3.txt:156-159): 31 files, the first 6 local
NOMAP = (['chb12_27', 'chb12_28', 'chb12_29', 'chb13_40', 'chb16_18', 'chb19_01', 'chb13_04']
         + ['chb13_%02d' % k for k in range(5, 17)]
         + ['chb13_18', 'chb13_24', 'chb13_30', 'chb13_36', 'chb13_37', 'chb13_38', 'chb13_39',
            'chb13_47', 'chb15_01', 'chb16_19', 'chb17c_13', 'chb18_01'])
LOCAL_NOMAP = NOMAP[:6]
CS2_FILES = ('chb12_27', 'chb12_28', 'chb12_29')
# Exact-mode kinds per NOMAP file, from the header pre-check
# (C:\EEG_v3\full_records_precheck\review_pc_maps_out.txt:21-51)
KINDS_CS2 = {'derived': 20, 'missing-allowed': 3}
KINDS_OTHER = {'direct': 19, 'minus': 1, 'missing-allowed': 3}
# slots whose source changes: all 23 in the CS2 files, these 6 in the other 28
MOVED_OTHER = ('T8-P8-0', 'P7-T7', 'T7-FT9', 'FT9-FT10', 'FT10-T8', 'T8-P8-1')
# loader code whose hashes the feature logs do not record (INFO only)
CODE_FILES = (HERE / 'v3_features.py', HERE / 'v3_edf.py', Path(r'C:\EEG\edf_to_npz.py'),
              Path(r'C:\EEG\download_and_convert.py'))
N_FILES = VF.N_RECORDS - len(VF.EXCLUDED)                     # 685
T = VF.TARGET_CHANNELS
I_T80, I_T81 = T.index('T8-P8-0'), T.index('T8-P8-1')
I_T7P7, I_P7T7 = T.index('T7-P7'), T.index('P7-T7')
I_FT = [T.index(t) for t in v3_edf.FT_TARGETS]
I_SKEW = list(VF.TF.NAMES).index('skew')
STATUS_RE = re.compile(r'^(CHB\d+)\s+(\S+)\s+windows\s.*\s(OK|NO-STORED|DZ-DIFF|COUNT-MISMATCH)$')
SUMMARY_RE = re.compile(r'^(\d+) recordings, status (\{.*\}), (\d+) not OK, [\d.]+ min$')
# windows over all 685 files from the headers, for comparison only
# (review_pc_maps_out.txt:64)
PRECHECK_WINDOWS = 1766780

assert len(NOMAP) == len(set(NOMAP)) == 31, len(NOMAP)


class Report:
    def __init__(self):
        self.lines, self.fail = [], 0

    def add(self, s=''):
        self.lines.append(s)
        print(s, flush=True)

    def result(self, name, status, detail=''):
        assert status in ('PASS', 'FAIL', 'INFO'), status
        self.fail += status == 'FAIL'
        self.add('%-8s %-5s %s' % (name, status, detail))


def load(folder, job):
    p = Path(folder) / VF.out_name(job['case'], job['stem'])
    if not p.exists():
        return None
    with np.load(p) as d:
        return {k: d[k] for k in d.files}


def load_chan(folder, job):
    """Only the channel fields (np.load reads keys lazily, so feats are not read)."""
    p = Path(folder) / VF.out_name(job['case'], job['stem'])
    if not p.exists():
        return None
    with np.load(p) as d:
        return {k: d[k] for k in ('chan_kind', 'chan_src', 'chan_a', 'chan_b')}


def s_list(a):
    return [str(x) for x in np.asarray(a).tolist()]


# ---------------------------------------------------------------- setup

def check_setup(rep, args, jobs):
    rep.add('--- setup: NOMAP list against the allow-list and the job list')
    allow = VF.load_allowlist(args.allowlist)
    files = sorted({f[:-len('.edf')] for f, _ in allow})
    rep.result('setup', 'PASS' if files == sorted(NOMAP) else 'FAIL',
               'allow-list files %d, NOMAP %d, only in allow-list %s, only in NOMAP %s'
               % (len(files), len(NOMAP), sorted(set(files) - set(NOMAP)), sorted(set(NOMAP) - set(files))))
    sha = VF.sha256(args.allowlist)
    rep.result('setup', 'PASS' if sha == VF.ALLOWLIST_SHA256 else 'FAIL',
               'allow-list sha256 %s, pinned %s' % (sha, VF.ALLOWLIST_SHA256))
    stems = {j['stem'] for j in jobs}
    local = {j['stem'] for j in jobs if j['npz']}
    bad = [s for s in NOMAP if s not in stems]
    rep.result('setup', 'PASS' if not bad and len(jobs) == N_FILES else 'FAIL',
               'jobs %d (expected %d), local %d, NOMAP not in jobs %s' % (len(jobs), N_FILES, len(local), bad))
    rep.result('setup', 'PASS' if sorted(s for s in NOMAP if s in local) == sorted(LOCAL_NOMAP) else 'FAIL',
               'local NOMAP files %s, expected %s' % (sorted(s for s in NOMAP if s in local), LOCAL_NOMAP))


# ---------------------------------------------------------------- logs

def read_log(folder):
    p = Path(folder) / 'v3_features_log.txt'
    if not p.exists():
        return None
    return p.read_text(encoding='ascii', errors='backslashreplace').splitlines()


def check_logs(rep, args, jobs):
    rep.add('--- logs: both runs finished, same code, per-file status')
    logs = {'old': read_log(args.old), 'exact': read_log(args.exact)}
    for m, lines in logs.items():
        if lines is None:
            rep.result('logs', 'FAIL', '%s: no v3_features_log.txt' % m)
    if None in logs.values():
        return
    head = {}
    for m, lines in logs.items():
        k = next((i for i, x in enumerate(lines) if x.startswith('mode ')), None)
        head[m] = lines[:k] if k is not None else None
        ok = (k is not None and lines[k] == 'mode %s' % m
              and any(x.startswith('args ') and "'records': 'full'" in x and "'mode': '%s'" % m in x
                      and "'only': ''" in x and "'cases': ''" in x for x in lines)
              and ('jobs %d' % N_FILES) in lines)
        rep.result('logs', 'PASS' if ok else 'FAIL',
                   '%s: mode line, args (records full, no --only/--cases), jobs %d' % (m, N_FILES))
        last = SUMMARY_RE.match(lines[-1]) if lines else None
        ok = bool(last) and int(last.group(1)) == N_FILES and int(last.group(3)) == 0
        rep.result('logs', 'PASS' if ok else 'FAIL', '%s: last line %r' % (m, lines[-1] if lines else ''))
    same = head['old'] is not None and head['old'] == head['exact']
    rep.result('logs', 'PASS' if same else 'FAIL',
               'versions and features() sha256 equal in both logs (%d lines)%s'
               % (len(head['old'] or []), '' if same else ': old %s exact %s' % (head['old'], head['exact'])))
    if same:
        for x in head['old']:
            rep.result('logs', 'INFO', x)
    for p in CODE_FILES:
        rep.result('logs', 'INFO', 'not in the logs: %s sha256 %s, modified %s'
                   % (p, VF.sha256(p), time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(p.stat().st_mtime))))
    sha_ok = any(x.startswith('allow-list ') and 'pinned value matches' in x for x in logs['exact'])
    rep.result('logs', 'PASS' if sha_ok else 'FAIL', 'exact: allow-list line says the pinned SHA-256 matches')

    local = {j['stem'] for j in jobs if j['npz']}
    for m, lines in logs.items():
        st = {}
        dup = []
        for x in lines:
            g = STATUS_RE.match(x)
            if g:
                if g.group(2) in st:
                    dup.append(g.group(2))
                st[g.group(2)] = '%s %s' % (g.group(1), g.group(3))
        want = {}
        for j in jobs:
            s = j['stem']
            if s not in local:
                want[s] = '%s NO-STORED' % j['case']
            elif m == 'exact' and s in LOCAL_NOMAP:
                want[s] = '%s DZ-DIFF' % j['case']
            else:
                want[s] = '%s OK' % j['case']
        if args.only:
            st = {s: v for s, v in st.items() if s in want}
        wrong = sorted('%s=%s (want %s)' % (s, st.get(s, 'absent'), w) for s, w in want.items() if st.get(s) != w)
        extra = sorted(set(st) - set(want))
        rep.result('logs', 'PASS' if not wrong and not extra and not dup else 'FAIL',
                   '%s: per-file status %s; wrong case or status %s; not a job %s; repeated %s'
                   % (m, dict(sorted(Counter(v.split()[1] for v in st.values()).items())), wrong, extra, dup))


# ---------------------------------------------------------------- inventory

def check_files(rep, args, jobs):
    rep.add('--- files: exactly the expected .npz in each folder')
    want = {VF.out_name(j['case'], j['stem']) for j in jobs}
    for m, folder in (('old', args.old), ('exact', args.exact)):
        have = {p.name for p in Path(folder).glob('*.npz')}
        tmp = sorted(x for x in have if x.endswith('.tmp.npz'))
        have -= set(tmp)
        missing = sorted(want - have)
        extra = sorted(have - want) if not args.only else []
        rep.result('files', 'PASS' if not missing and not extra and not tmp else 'FAIL',
                   '%s: %d .npz of %d expected; missing %s; extra %s; .tmp left %s'
                   % (m, len(have & want), len(want), missing[:10], extra[:10], tmp))


# ---------------------------------------------------------------- C2(ii) and NOMAP

def same_source(o_kind, o_a, e_kind, e_a):
    """One slot reads the same signal in both modes (or is zero in both)."""
    if o_kind == 'old' and e_kind == 'direct':
        return o_a == e_a
    return o_kind == 'zero' and e_kind == 'missing-allowed'


def check_nomap_file(rep, j, o, e):
    """Extra checks on one NOMAP file; returns (rebuilt slots over half NaN, P7-T7 cells differing)."""
    stem = j['stem']
    errs = []
    ok_, ek = s_list(o['chan_kind']), s_list(e['chan_kind'])
    oa, ea, eb = o['chan_a'].tolist(), e['chan_a'].tolist(), e['chan_b'].tolist()
    want = KINDS_CS2 if stem in CS2_FILES else KINDS_OTHER
    got = dict(Counter(ek))
    if got != want:
        errs.append('exact kinds %s, pre-check %s' % (got, want))
    same, moved_same, moved_diff = [], [], []
    for i in range(len(T)):
        eq = np.array_equal(o['feats'][:, i, :], e['feats'][:, i, :], equal_nan=True)
        if same_source(ok_[i], oa[i], ek[i], ea[i]):
            same.append(T[i])
            if not eq:
                errs.append('%s: same source (%s %d / %s %d) but feats differ' % (T[i], ok_[i], oa[i], ek[i], ea[i]))
        elif eq:
            moved_same.append(T[i])
        else:
            moved_diff.append(T[i])
    # the pre-check counts a slot as changed whenever its map entry differs, so zero in
    # old mode and missing-allowed in exact mode counts as changed here too
    moved = [T[i] for i in range(len(T)) if not (ok_[i] == 'old' and ek[i] == 'direct' and oa[i] == ea[i])]
    want_moved = list(T) if stem in CS2_FILES else list(MOVED_OTHER)
    if moved != want_moved:
        errs.append('source changed in %s, pre-check %s' % (moved, want_moved))
    # both T8-P8 slots are one signal
    if not (ek[I_T80] == ek[I_T81] and ea[I_T80] == ea[I_T81] and eb[I_T80] == eb[I_T81]
            and np.array_equal(e['feats'][:, I_T80, :], e['feats'][:, I_T81, :], equal_nan=True)):
        errs.append('T8-P8-0 and T8-P8-1 are not one signal')
    # P7-T7 is T7-P7 reversed
    if ek[I_P7T7] == 'minus':
        if not (ek[I_T7P7] == 'direct' and ea[I_P7T7] == ea[I_T7P7]):
            errs.append('P7-T7 minus source %d, T7-P7 %s %d' % (ea[I_P7T7], ek[I_T7P7], ea[I_T7P7]))
    elif ek[I_P7T7] == 'derived':
        if not (ek[I_T7P7] == 'derived' and ea[I_P7T7] == eb[I_T7P7] and eb[I_P7T7] == ea[I_T7P7]):
            errs.append('P7-T7 derived (%d, %d), T7-P7 %s (%d, %d)'
                        % (ea[I_P7T7], eb[I_P7T7], ek[I_T7P7], ea[I_T7P7], eb[I_T7P7]))
    else:
        errs.append('P7-T7 kind %s' % ek[I_P7T7])
    # the 3 FT inputs: allow-listed, zero, so every feature NaN
    for i in I_FT:
        if ek[i] != 'missing-allowed':
            errs.append('%s kind %s, expected missing-allowed' % (T[i], ek[i]))
        elif not np.isnan(e['feats'][:, i, :]).all():
            errs.append('%s missing-allowed but feats not all NaN' % T[i])
    # INFO only: rebuilt slots over half NaN (rule 6 reports them, it does not stop),
    # and P7-T7 feats against T7-P7's (every feature but skew equal, skew negated)
    flat = ['%s=%.2f' % (T[i], np.isnan(e['feats'][:, i, 0]).mean()) for i in range(len(T))
            if T[i] in moved and ek[i] in ('direct', 'minus', 'derived') and np.isnan(e['feats'][:, i, 0]).mean() > 0.5]
    p, q = e['feats'][:, I_P7T7, :].copy(), e['feats'][:, I_T7P7, :]
    p[:, I_SKEW] = -p[:, I_SKEW]
    p7_cells = int(np.sum(~((p == q) | (np.isnan(p) & np.isnan(q)))))
    rep.result('NOMAP', 'FAIL' if errs else 'PASS',
               '%s: exact kinds %s; %d slots same source or zero in both, identical feats; '
               'source changed: feats differ %d %s, '
               'feats identical %d %s; INFO rebuilt slots over half NaN %s, P7-T7 vs T7-P7 (skew negated) '
               '%d of %d cells differ%s'
               % (stem, dict(sorted(got.items())), len(same), len(moved_diff), moved_diff, len(moved_same),
                  moved_same, flat, p7_cells, p.size, ('; ' + '; '.join(errs)) if errs else ''))
    return flat, p7_cells


def matcher_errors(o, e, stem, allow):
    """Each stored map against its matcher re-run on the stored labels."""
    errs = []
    names = s_list(o['mne_names'])
    got = list(zip(s_list(o['chan_kind']), o['chan_a'].tolist(), o['chan_b'].tolist(), s_list(o['chan_src'])))
    for i, t in enumerate(T):
        idx = VF.find_matching_channel(t, names)
        want = ('old', idx, -1, names[idx]) if idx is not None else ('zero', -1, -1, 'ZERO')
        if got[i] != want:
            errs.append('old %s: stored %s, find_matching_channel gives %s' % (t, got[i], want))
    try:
        r = VF.exact_map(s_list(e['raw_labels']), stem + '.edf', allow)
    except VF.LoaderStop as x:
        return errs + ['exact matcher stops on the stored header labels: %s' % x]
    got = list(zip(s_list(e['chan_kind']), e['chan_a'].tolist(), e['chan_b'].tolist(), s_list(e['chan_src'])))
    for i, x in enumerate(r):
        want = (x['kind'], x['a'], x['b'], x['src'])
        if got[i] != want:
            errs.append('exact %s: stored %s, matcher gives %s' % (T[i], got[i], want))
    return errs


def check_c2ii(rep, args, jobs):
    rep.add('--- C2(ii): old vs exact; outside the 31 NOMAP files nothing may change')
    rep.add('--- NOMAP: per NOMAP file, the change is the one the header pre-check predicted')
    allow = VF.load_allowlist(args.allowlist)
    changed, n, n_win = [], 0, {'old': 0, 'exact': 0}
    tally, moved_path, flat_all, p7_all = Counter(), [], {}, {}
    for j in jobs:
        stem = j['stem']
        o, e = load(args.old, j), load(args.exact, j)
        if o is None or e is None:
            continue                                     # reported by check_files
        n += 1
        n_win['old'] += o['feats'].shape[0]
        n_win['exact'] += e['feats'].shape[0]
        errs = []
        if str(o['mode']) != 'old' or str(e['mode']) != 'exact':
            errs.append('mode fields %s / %s' % (o['mode'], e['mode']))
        if str(o['edf']) != str(e['edf']) or Path(str(o['edf'])).name != Path(j['edf']).name:
            errs.append('EDF %s / %s, job %s' % (o['edf'], e['edf'], j['edf']))
        elif str(o['edf']) != j['edf']:
            moved_path.append(stem)
        if s_list(o['raw_labels']) != s_list(e['raw_labels']) or s_list(o['mne_names']) != s_list(e['mne_names']):
            errs.append('raw_labels or mne_names differ')
        if s_list(o['names']) != list(VF.TF.NAMES) or s_list(e['names']) != list(VF.TF.NAMES):
            errs.append('feature names differ from trees_features.NAMES')
        shape_ok = o['feats'].shape == e['feats'].shape and o['feats'].shape[1:] == (len(T), len(VF.TF.NAMES))
        if not shape_ok:
            errs.append('feats shape %s / %s' % (o['feats'].shape, e['feats'].shape))
        if len(o['labels']) != o['feats'].shape[0] or len(e['labels']) != e['feats'].shape[0]:
            errs.append('labels %d / %d long, feats %d / %d windows'
                        % (len(o['labels']), len(e['labels']), o['feats'].shape[0], e['feats'].shape[0]))
        elif not np.array_equal(o['labels'], e['labels']):
            errs.append('labels differ between modes')
        has_o, has_e = 'labels_stored' in o, 'labels_stored' in e
        if has_o != bool(j['npz']) or has_e != bool(j['npz']):
            errs.append('labels_stored present old %s exact %s, local file %s' % (has_o, has_e, bool(j['npz'])))
        elif has_o and (len(o['labels_stored']) != o['feats'].shape[0]
                        or not np.array_equal(o['labels_stored'], e['labels_stored'])):
            errs.append('labels_stored differ between modes or from the window count')
        errs += matcher_errors(o, e, stem, allow)
        if errs:
            rep.result('C2(ii)', 'FAIL', '%s: %s' % (stem, '; '.join(errs)))
        same_map = (all(k == 'old' for k in s_list(o['chan_kind']))
                    and all(k == 'direct' for k in s_list(e['chan_kind']))
                    and np.array_equal(o['chan_a'], e['chan_a']))
        if not same_map:
            changed.append(stem)
        if not shape_ok:
            continue                                     # feats cannot be compared
        same_feats = np.array_equal(o['feats'], e['feats'], equal_nan=True)
        if stem in NOMAP:
            if same_map or same_feats:
                rep.result('C2(ii)', 'FAIL', '%s left the changed set (map same %s, feats same %s)'
                           % (stem, same_map, same_feats))
            else:
                tally['NOMAP changed'] += 1
                flat_all[stem], p7_all[stem] = check_nomap_file(rep, j, o, e)
        elif not same_map or not same_feats:
            rep.result('C2(ii)', 'FAIL', '%s entered the changed set (map same %s, feats same %s)'
                       % (stem, same_map, same_feats))
        else:
            tally['identical'] += 1
    want = sorted(s for s in NOMAP if any(j['stem'] == s for j in jobs))
    st = 'PASS' if sorted(changed) == want else 'FAIL'
    rep.result('C2(ii)', st, 'changed set %d files %s; expected %d (NOMAP); only changed %s; only expected %s'
               % (len(changed), sorted(changed), len(want), sorted(set(changed) - set(want)),
                  sorted(set(want) - set(changed))))
    rep.result('C2(ii)', 'PASS' if n == len(jobs) else 'FAIL',
               'compared %d files of %d; outside NOMAP identical %d; NOMAP changed %d'
               % (n, len(jobs), tally['identical'], tally['NOMAP changed']))
    win = 'windows old %d, exact %d; header pre-check over all %d files: %d' % (
        n_win['old'], n_win['exact'], N_FILES, PRECHECK_WINDOWS)
    if args.only:
        rep.result('C2(ii)', 'INFO', win + ' (partial run, not compared)')
    else:
        rep.result('C2(ii)', 'PASS' if n_win['old'] == n_win['exact'] == PRECHECK_WINDOWS else 'FAIL', win)
    rep.result('C2(ii)', 'INFO', 'EDF path differs from the job path today (same file name) in %d files %s'
               % (len(moved_path), moved_path[:10]))
    rep.result('NOMAP', 'INFO', 'rebuilt slots over half NaN: %d files %s'
               % (sum(1 for v in flat_all.values() if v), {k: v for k, v in flat_all.items() if v}))
    rep.result('NOMAP', 'INFO', 'P7-T7 vs T7-P7 (skew negated): cells differ in %d of %d files %s'
               % (sum(1 for v in p7_all.values() if v), len(p7_all), {k: v for k, v in p7_all.items() if v}))


# ---------------------------------------------------------------- chanmap.tsv

def check_chanmap(rep, args, jobs):
    rep.add('--- chanmap: chanmap.tsv agrees with the .npz channel fields')
    for m, folder in (('old', args.old), ('exact', args.exact)):
        p = Path(folder) / 'chanmap.tsv'
        if not p.exists():
            rep.result('chanmap', 'FAIL', '%s: no chanmap.tsv' % m)
            continue
        lines = p.read_text(encoding='ascii', errors='backslashreplace').splitlines()
        rows = {}
        for ln in lines[1:]:
            f, slot, target, kind, src, a, b = ln.split('\t')
            rows[(f, int(slot))] = (target, kind, src, int(a), int(b))
        errs, n = [], 0
        for j in jobs:
            d = load_chan(folder, j)
            if d is None:
                continue
            n += 1
            f = VF.out_name(j['case'], j['stem'])[:-4]
            for i in range(len(T)):
                want = (T[i], str(d['chan_kind'][i]), str(d['chan_src'][i]), int(d['chan_a'][i]),
                        int(d['chan_b'][i]))
                if rows.get((f, i)) != want:
                    errs.append('%s slot %d: tsv %s, npz %s' % (f, i, rows.get((f, i)), want))
        ok = (lines[0] == 'file\tslot\ttarget\tkind\tchan_src\ta\tb' and not errs
              and (args.only or len(rows) == len(lines) - 1 == N_FILES * len(T)))
        rep.result('chanmap', 'PASS' if ok else 'FAIL', '%s: %d rows, %d files checked, %d mismatches %s'
                   % (m, len(rows), n, len(errs), errs[:5]))


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--old', required=True, help='v3_features.py --mode old --records full folder')
    ap.add_argument('--exact', required=True, help='v3_features.py --mode exact --records full folder')
    ap.add_argument('--allowlist', default=str(VF.ALLOWLIST))
    ap.add_argument('--out', default=OUT_DEFAULT)
    ap.add_argument('--only', default='', help='comma list of stems, for small tests (exit 3 if no FAIL)')
    args = ap.parse_args()
    if Path.cwd().resolve() != VF.REPO.resolve():
        sys.exit('run with cwd C:\\EEG')
    out = Path(args.out)
    if out.exists():
        sys.exit('%s exists; give a new --out' % out)
    for f in (args.old, args.exact):
        if not Path(f).is_dir():
            sys.exit('no folder %s' % f)
    out.parent.mkdir(parents=True, exist_ok=True)

    try:
        jobs = VF.list_jobs('full')
    except AssertionError as x:
        sys.exit('the job list needs every EDF of the feature run in place: %s' % x)
    if args.only:
        want = {s.strip() for s in args.only.split(',') if s.strip()}
        unknown = want - {j['stem'] for j in jobs}
        if unknown:
            sys.exit('--only names not in the job list: %s' % ' '.join(sorted(unknown)))
        all_jobs, jobs = jobs, [j for j in jobs if j['stem'] in want]
    else:
        all_jobs = jobs

    rep = Report()
    rep.add('v3_check_full.py sha256 %s' % VF.sha256(__file__))
    rep.add('python %s, numpy %s, isolated (-I) %d' % (sys.executable, np.__version__, sys.flags.isolated))
    rep.add('args %s' % vars(args))
    rep.add('jobs %d%s' % (len(jobs), ' (PARTIAL, --only)' if args.only else ''))
    check_setup(rep, args, all_jobs)
    check_logs(rep, args, jobs)
    check_files(rep, args, jobs)
    check_c2ii(rep, args, jobs)
    check_chanmap(rep, args, jobs)
    verdict = 'FAIL' if rep.fail else ('PASS (PARTIAL)' if args.only else 'PASS')
    rep.add('OVERALL %s: %d FAIL' % (verdict, rep.fail))
    out.write_text('\n'.join(rep.lines) + '\n', encoding='ascii', errors='backslashreplace')
    sys.exit(1 if rep.fail else (3 if args.only else 0))


if __name__ == '__main__':
    main()
