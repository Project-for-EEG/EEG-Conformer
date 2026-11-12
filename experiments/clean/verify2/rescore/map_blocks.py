"""rescore / step 3: rebuild the test-block -> file mapping independently and
find the artifact files.

Written without reading clean_score.py. Run with cwd C:\\EEG.
Uses only the repo helpers patient_files / find_split.

For each case:
  files = patient_files(case); split = find_split(files, 1) = (fi, hi)
  test pieces = labels[hi:] of files[fi], then every later file whole;
  keep pieces with >= 10 windows.  Assert count and labels equal the stored
  test blocks, for EVERY arm file (not only A0).
Flat-channel rule, computed two ways:
  (a) the rule's definition: rms NaN in OLD\\feats (rms < 1e-6 uV)
  (b) independent of feats: per-window std of the z-scored preprocessed
      segment < 0.1 (a real channel has std ~1 after z-scoring; an all-zero
      channel has std 0 because z = 0/(0+1e-8)).
  For each: fraction of the file's windows flat, per channel; also the
  fraction over the TEST part only (split file uses windows hi: only).
Writes block_map.json and flat_report.txt next to this script.
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, r"C:\EEG")
from finetune_per_patient import patient_files, find_split  # noqa: E402

HERE = Path(__file__).resolve().parent
NEW = Path(r"C:\Users\user\AppData\Local\Temp\claude\36c1f29744f93fe6\c7848ca6-bfef-414b-a321-d6686c1a43f8\scratchpad\clean")
OLD = Path(r"C:\Users\user\AppData\Local\Temp\claude\36c1f29744f93fe6\b3889703-629c-4c5a-b583-69f722743ff4\scratchpad\trees")

PRED_FILES = [OLD / "out" / "preds_a.npy", OLD / "out" / "preds_b.npy",
              OLD / "out" / "preds_c.npy", OLD / "out" / "preds_d.npy",
              NEW / "trees_out" / "preds_n1.npy", NEW / "trees_out" / "preds_rel0.npy",
              NEW / "trees_out" / "preds_rel1.npy", NEW / "trees_out" / "preds_rel2.npy",
              NEW / "trees_out" / "preds_relf0.npy"]


def seg_flat(args):
    """Per-window, per-channel flat mask from the z-scored segments (chunked)."""
    path, lo = args
    with np.load(path) as d:
        seg = d["segments"]
    n = seg.shape[0]
    out = np.zeros((n, seg.shape[1]), dtype=bool)
    for a in range(0, n, 256):
        out[a:a + 256] = seg[a:a + 256].std(axis=-1) < 0.1
    del seg
    return str(path), lo, out


def main():
    arms = {}
    for f in PRED_FILES:
        d = np.load(f, allow_pickle=True).item()
        for a, v in d.items():
            assert a not in arms, a
            arms[a] = v
    cases = sorted(arms["A0"])
    assert len(cases) == 24, cases
    for a in arms:
        assert sorted(arms[a]) == cases, a

    mapping = {}
    jobs = []
    for case in cases:
        files = patient_files(case)
        split = find_split(files, 1)
        assert split is not None, case
        fi, hi = split
        pieces = []
        for i in range(fi, len(files)):
            with np.load(files[i]) as d:
                lab = d["labels"]
            lo = hi if i == fi else 0
            part = lab[lo:]
            if len(part) >= 10:
                pieces.append((files[i], lo, part))
        for a in arms:
            blocks = arms[a][case]
            assert len(blocks) == len(pieces), (a, case, len(blocks), len(pieces))
            for (f, lo, part), (p, l) in zip(pieces, blocks):
                assert len(p) == len(l) == len(part), (a, case, f.name)
                assert np.array_equal(np.asarray(l).astype(int), part.astype(int)), (a, case, f.name)
        mapping[case] = [{"file": f.stem, "start": int(lo), "n": int(len(part)),
                          "label_windows": int(part.sum())} for f, lo, part in pieces]
        jobs += [(f, lo) for f, lo, _ in pieces]
    print("blocks per case:", {c: len(v) for c, v in mapping.items()})
    print("total test blocks:", sum(len(v) for v in mapping.values()))
    print("labels and lengths match block-by-block for", len(arms), "arms:", sorted(arms))

    # (a) feats NaN
    rep = []
    feat_flag, seg_flag = {}, {}
    for f, lo in jobs:
        case = f.parent.name
        fp = OLD / "feats" / ("%s__%s" % (case, f.name))
        with np.load(fp) as d:
            rms = d["feats"][:, :, 0]
            flab = d["labels"]
        with np.load(f) as d:
            plab = d["labels"]
        assert len(flab) == len(plab) and np.array_equal(flab.astype(int), plab.astype(int)), fp
        flat = np.isnan(rms)
        frac_all = flat.mean(0)
        frac_test = flat[lo:].mean(0)
        feat_flag[f.stem] = (int((frac_all > 0.5).sum()), int((frac_test > 0.5).sum()),
                             int((flat.any(0)).sum()))
    # (b) segments, 3 processes
    with Pool(3) as pool:
        for path, lo, flat in pool.imap_unordered(seg_flat, jobs):
            stem = Path(path).stem
            frac_all = flat.mean(0)
            frac_test = flat[lo:].mean(0)
            seg_flag[stem] = (int((frac_all > 0.5).sum()), int((frac_test > 0.5).sum()),
                              int((flat.any(0)).sum()))
    rep.append("file  [feats: ch>50%%flat(all), ch>50%%flat(test part), ch ever flat]  [segments: same]")
    hit_feat, hit_seg = [], []
    for case in cases:
        for m in mapping[case]:
            s = m["file"]
            a, b = feat_flag[s], seg_flag[s]
            if a[0] >= 8 or a[1] >= 8:
                hit_feat.append(s)
            if b[0] >= 8 or b[1] >= 8:
                hit_seg.append(s)
            m["feat_flat"] = a
            m["seg_flat"] = b
            if max(a + b) > 0:
                rep.append("%-22s feats %s   segs %s" % (s, a, b))
    rep.append("rule hits (feats): %s" % hit_feat)
    rep.append("rule hits (segments): %s" % hit_seg)
    print("\n".join(rep))
    (HERE / "flat_report.txt").write_text("\n".join(rep) + "\n")
    (HERE / "block_map.json").write_text(json.dumps(mapping, indent=1))


if __name__ == "__main__":
    main()
