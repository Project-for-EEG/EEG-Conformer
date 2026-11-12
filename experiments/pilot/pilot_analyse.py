"""
Pilot, step 2: ensemble check and the 6-fold context trial. Reads pilot_cache only.

Go/no-go rule, written before any trial result was seen:
  GO      E1 (context, +/-32 s) beats E0 (same summaries, no context) by >= 5 points
          of pooled sensitivity at the 25 FA/day budget, averaged over 3 seeds, and is
          not worse at 10 or 50 FA/day.
  STOP    no average gain at 25 FA/day.
  UNCLEAR anything in between.
  The in-sample gate is reported next to it: if the fold models catch far more seizures
  on their own training patients than on the held-out ones, a STOP is weaker evidence.

This is a screen for deciding what to run properly. Its numbers are not for reporting.
Budgets are oracle (one global threshold swept over the test predictions), the same
procedure for every arm, achieved FA printed.
"""
import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

REPO = Path(r"C:\EEG")
sys.path.insert(0, str(REPO))
from score_official import score_all                      # noqa: E402
from finetune_per_patient import patient_files, find_split  # noqa: E402

TRIAL = ["CHB12", "CHB15", "CHB24", "CHB13", "CHB06", "CHB16"]
FAMILIES = ["lopo", "lopo20", "lopoA"]
SAME_SUBJECT = [("CHB01", "CHB21")]
BUDGETS = (5, 10, 25, 50, 100)
R = 16                   # +/-16 windows = +/-32 s
K, N = 5, 3              # today's post-processing
BG_PER_PATIENT = 6000
SEEDS = (0, 1, 2)


def sig(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


# ---------------------------------------------------------------- data ------
def load_case(cache, c):
    """Per-file dicts in patient_files order, plus the find_split result."""
    files = patient_files(c)
    recs = []
    for f in files:
        with np.load(cache / ("%s__%s.npz" % (c, f.stem))) as d:
            recs.append({k: d[k] for k in d.files})
    return recs, find_split(files, 1)


def post_split(recs, split, key):
    """[(array, labels)] after the end of the first seizure, test_blocks' rule."""
    fi, wi = split
    out = []
    for i, r in enumerate(recs):
        if i < fi:
            continue
        a, y = r[key], r["labels"]
        if i == fi:
            a, y = a[wi:], y[wi:]
        if len(y) >= 10:
            out.append((a, y))
    return out


def as_data(blocks_by_case):
    return {c: {"blocks": [(np.asarray(p, dtype=np.float64), y) for p, y in b]}
            for c, b in blocks_by_case.items()}


def sweep(data, k=K, n=N):
    probs = np.concatenate([b[0] for c in data for b in data[c]["blocks"]])
    grid = np.unique(np.concatenate([np.linspace(0.01, 0.999, 100),
                                     np.quantile(probs, np.linspace(0.70, 0.99999, 500))]))
    return [score_all(data, {c: float(t) for c in data}, k, n)[0] for t in grid]


def budgets(sw):
    out = {}
    for b in BUDGETS:
        ok = [s for s in sw if s["fa_day"] <= b]
        best = max(ok, key=lambda s: s["sens"]) if ok else None
        out[b] = (best["sens"], best["fa_day"], best["caught"]) if best else (None, None, None)
    return out


def fmt(name, bd):
    row = "%-22s" % name
    for b in BUDGETS:
        s, f, _ = bd[b]
        row += "%15s" % ("-" if s is None else "%.3f (%5.1f)" % (s, f))
    return row


HEADER = "%-22s" % "" + "".join("%15s" % ("<=%d/d" % b) for b in BUDGETS)


# ---------------------------------------------------------------- heads -----
class MLP(nn.Module):
    def __init__(self, d=32, h=96):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Dropout(0.1),
                                 nn.Linear(h, h), nn.GELU(), nn.Dropout(0.1),
                                 nn.Linear(h, 1))

    def forward(self, x):            # x: (B, 2R+1, d) -> centre only
        return self.net(x[:, x.shape[1] // 2]).squeeze(-1)


class BiGRU(nn.Module):
    def __init__(self, d=32, h=32):
        super().__init__()
        self.drop = nn.Dropout(0.1)
        self.gru = nn.GRU(d, h, batch_first=True, bidirectional=True)
        self.out = nn.Linear(2 * h, 1)

    def forward(self, x):
        y, _ = self.gru(self.drop(x))
        return self.out(y[:, x.shape[1] // 2]).squeeze(-1)


def n_params(m):
    return sum(p.numel() for p in m.parameters())


class Seqs:
    """Fixed-length sequences around chosen centres, edge-padded per block."""

    def __init__(self, blocks, radius):
        self.r = radius
        self.pad = [np.pad(e, ((radius, radius), (0, 0)), mode="edge") for e, _ in blocks]
        self.lab = [y for _, y in blocks]

    def batch(self, idx):
        X = np.stack([self.pad[b][i:i + 2 * self.r + 1] for b, i in idx])
        y = np.array([self.lab[b][i] for b, i in idx], dtype=np.float32)
        return torch.from_numpy(X), torch.from_numpy(y)


def sample_centres(blocks, rng, bg_cap=BG_PER_PATIENT):
    pos, neg = [], []
    for b, (_, y) in enumerate(blocks):
        for i in np.flatnonzero(y == 1):
            pos.append((b, int(i)))
        for i in np.flatnonzero(y == 0):
            neg.append((b, int(i)))
    if len(neg) > bg_cap:
        neg = [neg[j] for j in rng.choice(len(neg), bg_cap, replace=False)]
    return pos + neg


def train_head(arm, train_sets, val_sets, seed, device, epochs=10, patience=3):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    radius = R if arm == "E1" else R      # same inputs for both; E0 reads only the centre
    model = (BiGRU() if arm == "E1" else MLP()).to(device)
    tr = [(Seqs(bl, radius), cen) for bl, cen in train_sets]
    va = [(Seqs(bl, radius), cen) for bl, cen in val_sets]
    flat = [(s, c) for s, cs in tr for c in cs]
    n_pos = sum(1 for s, c in flat if s.lab[c[0]][c[1]] == 1)
    pos_w = torch.tensor((len(flat) - n_pos) / max(n_pos, 1), device=device)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    def run_epoch(items, train):
        model.train(train)
        order = rng.permutation(len(items)) if train else np.arange(len(items))
        tot, cnt = 0.0, 0
        for j in range(0, len(order), 512):
            chunk = [items[k] for k in order[j:j + 512]]
            # group by owning Seqs object so batches can be stacked
            Xs, ys = [], []
            by = {}
            for s, c in chunk:
                by.setdefault(id(s), (s, []))[1].append(c)
            for s, cs in by.values():
                X, y = s.batch(cs)
                Xs.append(X)
                ys.append(y)
            X = torch.cat(Xs).float().to(device)
            y = torch.cat(ys).to(device)
            with torch.set_grad_enabled(train):
                loss = lossf(model(X), y)
                if train:
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
            tot += float(loss) * len(y)
            cnt += len(y)
        return tot / max(cnt, 1)

    vflat = [(s, c) for s, cs in va for c in cs]
    best, best_state, bad = float("inf"), None, 0
    for ep in range(epochs):
        run_epoch(flat, True)
        vl = run_epoch(vflat, False)
        if vl < best:
            best, best_state, bad = vl, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, ep + 1, best


def predict_blocks(model, blocks, device):
    out = []
    s = Seqs(blocks, R)
    with torch.no_grad():
        for b, (_, y) in enumerate(blocks):
            res = []
            for j in range(0, len(y), 2048):
                X, _ = s.batch([(b, i) for i in range(j, min(j + 2048, len(y)))])
                res.append(model(X.float().to(device)).double().cpu().numpy())
            out.append((sig(np.concatenate(res)), y))
    return out


# ---------------------------------------------------------------- main ------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cache, outdir = Path(args.cache), Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    lines = []

    def say(s=""):
        print(s, flush=True)
        lines.append(s)

    cases = sorted(d.name for d in (REPO / "preprocessed_data").iterdir() if d.is_dir())
    recs, splits = {}, {}
    for c in cases:
        recs[c], splits[c] = load_case(cache, c)
        assert splits[c] is not None, c

    # ---- reproduction check against the README cache ----------------------
    ref = np.load(REPO / "endtoend_uncalibrated.npz", allow_pickle=True)["data"].item()
    worst = 0.0
    for c in cases:
        mine = post_split(recs[c], splits[c], "own_lopo")
        theirs = ref[c]["blocks"]
        assert len(mine) == len(theirs), c
        for (lg, y1), (p2, y2) in zip(mine, theirs):
            assert np.array_equal(y1, y2), c
            worst = max(worst, float(np.max(np.abs(sig(lg) - p2))))
    say("CHECK lopo_ probabilities vs endtoend_uncalibrated.npz: max |diff| %.2e (%s)"
        % (worst, "PASS" if worst < 1e-4 else "FAIL"))
    stored = {c: float(ref[c]["t"]) for c in cases}
    a0 = as_data({c: [(sig(lg), y) for lg, y in post_split(recs[c], splits[c], "own_lopo")]
                  for c in cases})
    pl, _, _ = score_all(a0, stored, K, N)
    say("CHECK A0 rescored: %d/%d at %.1f FA/day (README: 131/168 at 131.2)"
        % (pl["caught"], pl["events"], pl["fa_day"]))
    say()

    # ---- free check 2: ensemble of the three CHB-only families ------------
    say("FREE CHECK 2: ensemble of lopo_, lopo20_, lopoA__ (all 24 cases, oracle budgets)")
    say(HEADER)
    fam_logits = {f: {c: post_split(recs[c], splits[c], "own_" + f) for c in cases}
                  for f in FAMILIES}
    results = {"ensemble": {}}
    for f in FAMILIES:
        bd = budgets(sweep(as_data({c: [(sig(lg), y) for lg, y in fam_logits[f][c]]
                                    for c in cases})))
        results["ensemble"][f] = bd
        say(fmt(f + "_ alone", bd))
    ens = {c: [(sig(np.mean([fam_logits[f][c][i][0] for f in FAMILIES], axis=0)), y)
               for i, (_, y) in enumerate(fam_logits["lopo"][c])] for c in cases}
    bd = budgets(sweep(as_data(ens)))
    results["ensemble"]["mean_logit"] = bd
    say(fmt("ensemble (mean logit)", bd))
    say()

    # ---- in-sample gate ---------------------------------------------------
    say("IN-SAMPLE GATE: each trial fold model at its own stored threshold")
    say("%-7s %22s %22s %10s %10s" % ("fold", "training patients", "held-out patient",
                                       "AUC train", "AUC held"))
    gate = {}
    tr_c = tr_e = ho_c = ho_e = 0
    for p in TRIAL:
        key = "logit_" + p
        train_cases = [c for c in cases if c != p]
        dtr = as_data({c: [(sig(a), y) for a, y in post_split(recs[c], splits[c], key)]
                       for c in train_cases})
        dho = as_data({p: [(sig(a), y) for a, y in post_split(recs[p], splits[p], key)]})
        t = stored[p]
        ptr, _, _ = score_all(dtr, {c: t for c in dtr}, K, N)
        pho, _, _ = score_all(dho, {p: t}, K, N)
        ytr = np.concatenate([y for c in dtr for _, y in dtr[c]["blocks"]])
        str_ = np.concatenate([s for c in dtr for s, _ in dtr[c]["blocks"]])
        yho = np.concatenate([y for _, y in dho[p]["blocks"]])
        sho = np.concatenate([s for s, _ in dho[p]["blocks"]])
        auc_tr, auc_ho = roc_auc_score(ytr, str_), roc_auc_score(yho, sho)
        gate[p] = {"train": (ptr["caught"], ptr["events"]), "held": (pho["caught"], pho["events"]),
                   "auc_train": auc_tr, "auc_held": auc_ho}
        tr_c += ptr["caught"]
        tr_e += ptr["events"]
        ho_c += pho["caught"]
        ho_e += pho["events"]
        say("%-7s %10d/%-4d %.3f %10d/%-4d %.3f %10.3f %10.3f"
            % (p, ptr["caught"], ptr["events"], ptr["sens"], pho["caught"], pho["events"],
               pho["sens"], auc_tr, auc_ho))
    gap = tr_c / tr_e - ho_c / ho_e
    say("pooled: training %.3f vs held-out %.3f -> gap %+.3f (%s)"
        % (tr_c / tr_e, ho_c / ho_e, gap, "TRIPS (>0.15)" if gap > 0.15 else "ok"))
    results["gate"] = {"folds": gate, "gap": gap}
    say()

    # ---- the trial --------------------------------------------------------
    say("TRIAL: 6 folds, heads trained on training patients' post-first-seizure windows")
    arms_pred = {"A0": {p: [(sig(lg), y) for lg, y in post_split(recs[p], splits[p], "own_lopo")]
                        for p in TRIAL}}
    groups = []
    seen = set()
    for a, b in SAME_SUBJECT:
        groups.append([a, b])
        seen.update((a, b))
    groups += [[c] for c in cases if c not in seen]
    log = []
    for fi, p in enumerate(TRIAL):
        key = "emb_" + p
        train_groups = [g for g in groups if p not in g]
        vrng = np.random.default_rng(1000 + fi)
        val_groups = [train_groups[j] for j in vrng.choice(len(train_groups), 3, replace=False)]
        val_cases = [c for g in val_groups for c in g]
        tr_cases = [c for g in train_groups for c in g if c not in val_cases]
        blocks = {c: [(e.astype(np.float32), y) for e, y in post_split(recs[c], splits[c], key)]
                  for c in tr_cases + val_cases + [p]}
        allx = np.concatenate([e for c in tr_cases for e, _ in blocks[c]])
        mu, sd = allx.mean(0), allx.std(0) + 1e-6
        blocks = {c: [((e - mu) / sd, y) for e, y in bl] for c, bl in blocks.items()}
        srng = np.random.default_rng(fi)
        train_sets = [(blocks[c], sample_centres(blocks[c], srng)) for c in tr_cases]
        val_sets = [(blocks[c], sample_centres(blocks[c], srng)) for c in val_cases]
        for arm in ("E0", "E1"):
            for seed in SEEDS:
                model, ep, vl = train_head(arm, train_sets, val_sets, seed, device)
                arms_pred.setdefault("%s_s%d" % (arm, seed), {})[p] = \
                    predict_blocks(model, blocks[p], device)
                log.append({"fold": p, "arm": arm, "seed": seed, "epochs": ep,
                            "val_loss": vl, "params": n_params(model), "val": val_cases})
                print("  %s %s seed %d: %d epochs, val loss %.4f, %d params"
                      % (p, arm, seed, ep, vl, n_params(model)), flush=True)
    results["train_log"] = log
    say()
    say("Oracle budgets on the 6 trial patients, today's post-processing (smooth 5, 3 in a row)")
    say(HEADER)
    trial_bd = {}
    for name, pred in arms_pred.items():
        bd = budgets(sweep(as_data(pred)))
        trial_bd[name] = bd
        say(fmt(name, bd))
    results["trial"] = trial_bd
    say()
    say("Same, without smoothing (k=1, 3 in a row), heads only")
    say(HEADER)
    for name, pred in arms_pred.items():
        if name == "A0":
            continue
        say(fmt(name + " k=1", budgets(sweep(as_data(pred), k=1))))
    say()

    def mean_at(arm, b):
        return float(np.mean([trial_bd["%s_s%d" % (arm, s)][b][0] for s in SEEDS]))
    d = {b: mean_at("E1", b) - mean_at("E0", b) for b in (10, 25, 50)}
    say("E1 - E0, mean over seeds: 10/d %+.3f | 25/d %+.3f | 50/d %+.3f" % (d[10], d[25], d[50]))
    if d[25] >= 0.05 and d[10] >= 0 and d[50] >= 0:
        verdict = "GO"
    elif d[25] <= 0:
        verdict = "STOP"
    else:
        verdict = "UNCLEAR"
    say("VERDICT (rule fixed in advance): %s" % verdict)
    results["verdict"] = {"diff": d, "verdict": verdict}

    np.save(outdir / "trial_predictions.npy", np.array(arms_pred, dtype=object),
            allow_pickle=True)
    (outdir / "pilot_results.json").write_text(json.dumps(results, indent=1, default=str))
    (outdir / "pilot_results.txt").write_text("\n".join(lines) + "\n", encoding="ascii")


if __name__ == "__main__":
    main()
