"""
v3 helpers: EDF header reading, completeness check and the exact channel matcher.

Plain file reads only (no MNE). Used by v3_allowlist.py, v3_features.py and
v3_check_loader.py. Writes nothing.
"""
import os

# The 23 model inputs, in the order the repo uses (download_and_convert.py:29-36).
# v3_features.py and v3_allowlist.py assert this equals the repo list.
TARGETS = [
    'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1',
    'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
    'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
    'FP2-F8', 'F8-T8', 'T8-P8-0', 'P8-O2',
    'FZ-CZ', 'CZ-PZ',
    'P7-T7', 'T7-FT9', 'FT9-FT10', 'FT10-T8', 'T8-P8-1'
]
FT_TARGETS = ('T7-FT9', 'FT9-FT10', 'FT10-T8')
SYNONYMS = {'T3': 'T7', 'T4': 'T8', 'T5': 'P7', 'T6': 'P8'}
DUMMY_CHARS = set('-.0123456789')


# ---------------------------------------------------------------- header

def read_header(path):
    """Parse the fixed and per-signal EDF header fields with plain reads."""
    with open(path, 'rb') as f:
        fixed = f.read(256)
        if len(fixed) < 256:
            raise ValueError('%s: header shorter than 256 bytes' % path)
        ns = int(fixed[252:256].decode('ascii').strip())
        rest = f.read(256 * ns)
        if len(rest) < 256 * ns:
            raise ValueError('%s: signal header shorter than %d bytes' % (path, 256 * ns))

    def block(offset, width):
        out = []
        for i in range(ns):
            a = offset + i * width
            out.append(rest[a:a + width].strip().decode('latin-1'))
        return out

    hdr = {
        'path': str(path),
        'start_date': fixed[168:176].decode('latin-1').strip(),
        'start_time': fixed[176:184].decode('latin-1').strip(),
        'header_bytes': int(fixed[184:192].decode('ascii').strip()),
        'n_records': int(fixed[236:244].decode('ascii').strip()),
        'record_duration': float(fixed[244:252].decode('ascii').strip()),
        'ns': ns,
    }
    # offsets relative to byte 256, each field is ns * width bytes
    hdr['labels'] = block(0, 16)
    hdr['dimension'] = block(ns * 96, 8)
    hdr['physical_min'] = [float(x) for x in block(ns * 104, 8)]
    hdr['physical_max'] = [float(x) for x in block(ns * 112, 8)]
    hdr['digital_min'] = [int(float(x)) for x in block(ns * 120, 8)]
    hdr['digital_max'] = [int(float(x)) for x in block(ns * 128, 8)]
    hdr['samples_per_record'] = [int(x) for x in block(ns * 216, 8)]
    return hdr


def complete(path):
    """(ok, on_disk, expected) with expected = 256*(ns+1) + n_records*sum(spr)*2.

    A header that cannot be parsed (too short, bad fields) is not ok, expected -1.
    """
    on_disk = os.path.getsize(path)
    try:
        hdr = read_header(path)
    except ValueError:
        return False, on_disk, -1
    if hdr['n_records'] <= 0:
        return False, on_disk, -1
    expected = 256 * (hdr['ns'] + 1) + hdr['n_records'] * sum(hdr['samples_per_record']) * 2
    return on_disk == expected, on_disk, expected


# ---------------------------------------------------------------- matcher

def normalise(label):
    """Step 1 and 2: upper case, no spaces, no leading EEG, then synonyms."""
    s = label.upper().replace(' ', '')
    if s.startswith('EEG'):
        s = s[3:]
    if s == '01':
        s = 'O1'
    return '-'.join(SYNONYMS.get(t, t) for t in s.split('-'))


def is_dummy(norm):
    """Step 3: made only of '-', '.' and digits."""
    return set(norm) <= DUMMY_CHARS


def target_pair(target):
    """'T8-P8-0' -> ('T8', 'P8'); 'FP1-F7' -> ('FP1', 'F7')."""
    parts = target.split('-')
    return parts[0], parts[1]


def match_targets(labels):
    """Exact matcher on raw labels (header or summary block).

    Returns 23 dicts, one per target, with
      kind: 'direct' (a), 'minus' (a), 'derived' (a minus b), 'missing' or 'error'
      a, b: label indices (-1 when unused)
      src:  readable source, e.g. 'T8-P8#1', 'F7-CS2 minus T7-CS2', 'minus T7-P7'
      reason: for missing and error
    The allow-list is applied by the caller.
    """
    norm = [normalise(x) for x in labels]
    keep = [i for i, s in enumerate(norm) if not is_dummy(s)]
    by_name = {}
    for i in keep:
        by_name.setdefault(norm[i], []).append(i)
    bare, cs2 = {}, {}
    for i in keep:
        s = norm[i]
        if '-' not in s:
            bare.setdefault(s, []).append(i)
        elif s.endswith('-CS2') and s.count('-') == 1:
            cs2.setdefault(s[:-4], []).append(i)

    def dup_name(i):
        # '#k' marks the k-th occurrence of a raw label that occurs more than once,
        # counted over the raw labels as MNE does when it renames duplicates
        same = [j for j, x in enumerate(labels) if x == labels[i]]
        if len(same) > 1:
            return '%s#%d' % (labels[i], same.index(i))
        return labels[i]

    out = []
    for t in TARGETS:
        A, B = target_pair(t)
        ab = A + '-' + B
        hits = by_name.get(ab, [])
        r = {'target': t, 'kind': None, 'a': -1, 'b': -1, 'src': '', 'reason': ''}
        # step 4: direct bipolar label
        if ab == 'T8-P8' and len(hits) in (1, 2):
            k = 1 if (t == 'T8-P8-1' and len(hits) == 2) else 0
            r.update(kind='direct', a=hits[k], src=dup_name(hits[k]))
            out.append(r)
            continue
        if len(hits) > 1:
            r.update(kind='error', reason="label '%s' occurs %d times" % (ab, len(hits)))
            out.append(r)
            continue
        if len(hits) == 1:
            r.update(kind='direct', a=hits[0], src=dup_name(hits[0]))
            out.append(r)
            continue
        if t == 'P7-T7':
            rev = by_name.get('T7-P7', [])
            if len(rev) > 1:
                r.update(kind='error', reason="label 'T7-P7' occurs %d times" % len(rev))
                out.append(r)
                continue
            if len(rev) == 1:
                r.update(kind='minus', a=rev[0], src='minus ' + labels[rev[0]])
                out.append(r)
                continue
        # step 5: derive from bare or -CS2 referenced electrodes
        pick = None
        for pool, what in ((bare, 'bare'), (cs2, '-CS2')):
            if A in pool and B in pool:
                if len(pool[A]) > 1 or len(pool[B]) > 1:
                    r.update(kind='error', reason='electrode %s or %s occurs more than once as %s name'
                             % (A, B, what))
                    break
                if pick is not None:
                    r.update(kind='error', reason='electrodes %s and %s present both bare and -CS2' % (A, B))
                    break
                pick = (pool[A][0], pool[B][0])
        if r['kind'] == 'error':
            out.append(r)
            continue
        if pick is not None:
            a, b = pick
            r.update(kind='derived', a=a, b=b, src='%s minus %s' % (labels[a], labels[b]))
            out.append(r)
            continue
        # step 6 and 7 are the caller's (allow-list or stop)
        r.update(kind='missing',
                 reason="no '%s' label; electrodes %s, %s not both bare or both -CS2" % (ab, A, B))
        out.append(r)
    return out
