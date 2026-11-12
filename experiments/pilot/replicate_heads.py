"""
Replication, step 2: checks, in-sample gate on all 23 folds, and E0/E1 heads for the
17 new folds. Head code, settings and seeds are pilot_analyse.py's, imported unchanged.
The 6 pilot folds keep their saved predictions (pilot_out/trial_predictions.npy).

Must run with cwd C:\\EEG (patient_files uses a relative path).
Writes <out>/all_predictions.npy, heads_log.json, heads_results.txt.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.dont_write_bytecode = True
REPO = Path(r"C:\EEG")
sys.path.insert(0, str(REPO))
from finetune_per_patient import patient_files, find_split   # noqa: E402
from score_official import score_all                          # noqa: E402
from pilot_analyse import (sig, post_split, as_data, sample_centres, train_head,  # noqa: E402
                           predict_blocks, n_params, TRIAL, SAME_SUBJECT, SEEDS, K, N)

HERE = Path(__file__).resolve().parent
CACHE1 = HERE / "pilot_cache"
CACHE2 = HERE / "pilot_cache2"
PILOT_PRED = HERE / "pilot_out" / "trial_predictions.npy"
CKPT = REPO / "checkpoints" / "lopo"


def load_case(c, new_folds):
    files = patient_files(c)
    recs = []
    for f in files:
        r = {}
        with np.load(CACHE1 / ("%s__%s.npz" % (c, f.stem))) as d:
            r["labels"] = d["labels"]
            r["own_lopo"] = d["own_lopo"]
            for p in TRIAL:
                r["logit_" + p] = d["logit_" + p]
        with np.load(CACHE2 / ("%s__%s.npz" % (c, f.stem))) as d:
            assert np.array_equal(d["labels"], r["labels"]), (c, f.stem)
            for name in new_folds:
                r["logit_" + name] = d["logit_" + name]
                r["emb_" + name] = d["emb_" + name]
        recs.append(r)
    return recs, find_split(files, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    lines = []

    def say(s=""):
        print(s, flush=True)
        lines.append(s)

    folds = {}
    for path in sorted(CKPT.glob("lopo_*.pt")):
        name = path.stem[len("lopo_"):]
        folds[name] = name.split("_")
    new_folds = [f for f in folds if f not in TRIAL]
    assert len(folds) == 23 and len(new_folds) == 17
    cases = sorted(d.name for d in (REPO / "preprocessed_data").iterdir() if d.is_dir())
    assert sorted(c for h in folds.values() for c in h) == cases
    t0 = time.time()
    recs, splits = {}, {}
    for c in cases:
        recs[c], splits[c] = load_case(c, new_folds)
        assert splits[c] is not None, c
    say("loaded %d cases, %d new folds, %.0f s" % (len(cases), len(new_folds), time.time() - t0))

    # ---- checks ------------------------------------------------------------
    ref = np.load(REPO / "endtoend_uncalibrated.npz", allow_pickle=True)["data"].item()
    stored = {c: float(ref[c]["t"]) for c in cases}
    worst = 0.0
    for name in new_folds:
        for c in folds[name]:
            for r in recs[c]:
                worst = max(worst, float(np.max(np.abs(sig(r["logit_" + name])
                                                       - sig(r["own_lopo"])))))
    say("CHECK new fold logits vs own_lopo on their held-out cases: max |dp| %.2e (%s)"
        % (worst, "PASS" if worst < 1e-4 else "FAIL"))
    a0 = {c: [(sig(lg), y) for lg, y in post_split(recs[c], splits[c], "own_lopo")]
          for c in cases}
    pl, _, _ = score_all(as_data(a0), stored, K, N)
    say("CHECK A0 rescored: %d/%d at %.1f FA/day (README: 131/168 at 131.2)"
        % (pl["caught"], pl["events"], pl["fa_day"]))
    pilot = np.load(PILOT_PRED, allow_pickle=True).item()
    wp = 0.0
    for c in TRIAL:
        for (p1, y1), (p2, y2) in zip(pilot["A0"][c], a0[c]):
            assert np.array_equal(y1, y2)
            wp = max(wp, float(np.max(np.abs(p1 - p2))))
    say("CHECK pilot A0 predictions vs this cache: max |dp| %.2e (%s)"
        % (wp, "PASS" if wp < 1e-9 else "FAIL"))
    for name in new_folds:
        t = {stored[c] for c in folds[name]}
        assert len(t) == 1, (name, t)
    say()

    # ---- in-sample gate, all 23 folds ---------------------------------------
    say("IN-SAMPLE GATE: each fold model at its own stored threshold")
    say("%-12s %18s %18s %10s %10s" % ("fold", "training patients", "held-out",
                                        "AUC train", "AUC held"))
    gate = {}
    tr_c = tr_e = ho_c = ho_e = 0
    for name, held in folds.items():
        key = "logit_" + name
        t = stored[held[0]]
        dtr = as_data({c: [(sig(a), y) for a, y in post_split(recs[c], splits[c], key)]
                       for c in cases if c not in held})
        dho = as_data({c: [(sig(a), y) for a, y in post_split(recs[c], splits[c], key)]
                       for c in held})
        ptr, _, _ = score_all(dtr, {c: t for c in dtr}, K, N)
        pho, _, _ = score_all(dho, {c: t for c in dho}, K, N)
        ytr = np.concatenate([y for c in dtr for _, y in dtr[c]["blocks"]])
        s_tr = np.concatenate([s for c in dtr for s, _ in dtr[c]["blocks"]])
        yho = np.concatenate([y for c in dho for _, y in dho[c]["blocks"]])
        s_ho = np.concatenate([s for c in dho for s, _ in dho[c]["blocks"]])
        auc_tr = roc_auc_score(ytr, s_tr)
        auc_ho = roc_auc_score(yho, s_ho) if 0 < yho.sum() < len(yho) else float("nan")
        gate[name] = {"train": [ptr["caught"], ptr["events"]],
                      "held": [pho["caught"], pho["events"]],
                      "auc_train": auc_tr, "auc_held": auc_ho,
                      "pilot": name in TRIAL}
        tr_c += ptr["caught"]
        tr_e += ptr["events"]
        ho_c += pho["caught"]
        ho_e += pho["events"]
        say("%-12s %8d/%-4d %.3f %8d/%-4d %.3f %10.3f %10.3f"
            % (name, ptr["caught"], ptr["events"], ptr["sens"], pho["caught"], pho["events"],
               pho["sens"], auc_tr, auc_ho))
    gap = tr_c / tr_e - ho_c / ho_e
    say("pooled, 23 folds: training %.3f vs held-out %.3f -> gap %+.3f (%s)"
        % (tr_c / tr_e, ho_c / ho_e, gap, "TRIPS (>0.15)" if gap > 0.15 else "ok"))
    say()

    # ---- heads for the 17 new folds -----------------------------------------
    groups = [list(g) for g in SAME_SUBJECT]
    seen = {c for g in groups for c in g}
    groups += [[c] for c in cases if c not in seen]
    new_pred = {"A0": {c: a0[c] for c in cases}}
    log = []
    for j, name in enumerate(new_folds):
        fi = len(TRIAL) + j
        held = folds[name]
        key = "emb_" + name
        train_groups = [g for g in groups if not set(g) & set(held)]
        vrng = np.random.default_rng(1000 + fi)
        val_groups = [train_groups[i] for i in vrng.choice(len(train_groups), 3, replace=False)]
        val_cases = [c for g in val_groups for c in g]
        tr_cases = [c for g in train_groups for c in g if c not in val_cases]
        blocks = {c: [(e.astype(np.float32), y) for e, y in post_split(recs[c], splits[c], key)]
                  for c in tr_cases + val_cases + held}
        allx = np.concatenate([e for c in tr_cases for e, _ in blocks[c]])
        mu, sd = allx.mean(0), allx.std(0) + 1e-6
        blocks = {c: [((e - mu) / sd, y) for e, y in bl] for c, bl in blocks.items()}
        srng = np.random.default_rng(fi)
        train_sets = [(blocks[c], sample_centres(blocks[c], srng)) for c in tr_cases]
        val_sets = [(blocks[c], sample_centres(blocks[c], srng)) for c in val_cases]
        for arm in ("E0", "E1"):
            for seed in SEEDS:
                model, ep, vl = train_head(arm, train_sets, val_sets, seed, device)
                for c in held:
                    new_pred.setdefault("%s_s%d" % (arm, seed), {})[c] = \
                        predict_blocks(model, blocks[c], device)
                log.append({"fold": name, "fi": fi, "arm": arm, "seed": seed, "epochs": ep,
                            "val_loss": vl, "params": n_params(model), "val": val_cases})
                print("  %s %s seed %d: %d epochs, val loss %.4f, %d params, %.0f s"
                      % (name, arm, seed, ep, vl, n_params(model), time.time() - t0), flush=True)
    capped = {a: sum(1 for r in log if r["arm"] == a and r["epochs"] == 10) for a in ("E0", "E1")}
    say("heads trained: %d runs; hit the 10-epoch cap: E0 %d/51, E1 %d/51"
        % (len(log), capped["E0"], capped["E1"]))

    # ---- merge with the pilot's saved predictions ---------------------------
    allp = {}
    for arm in new_pred:
        allp[arm] = {}
        for c in cases:
            if c in TRIAL:
                allp[arm][c] = pilot[arm][c]
            else:
                allp[arm][c] = new_pred[arm][c]
    for arm in allp:
        for c in cases:
            assert len(allp[arm][c]) == len(a0[c])
            for (p, y), (_, y0) in zip(allp[arm][c], a0[c]):
                assert len(p) == len(y0) and np.array_equal(y, y0), (arm, c)
    say("merged predictions: arms %s, %d cases each" % (", ".join(allp), len(cases)))
    say("done in %.0f min" % ((time.time() - t0) / 60))

    np.save(outdir / "all_predictions.npy", np.array(allp, dtype=object), allow_pickle=True)
    (outdir / "heads_log.json").write_text(json.dumps({"gate": gate, "gap": gap, "log": log},
                                                      indent=1, default=str))
    (outdir / "heads_results.txt").write_text("\n".join(lines) + "\n", encoding="ascii")


if __name__ == "__main__":
    main()
