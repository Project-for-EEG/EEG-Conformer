"""critic step D (Q3): amplitude levels per case, missing channels, one-feature amplitude AUC,
and how per-case amplitude relates to each arm's per-case FA and sensitivity at fixed targets.
Reads feats/*.npz (read-only), verify/critic/deploy_tables.npz, auc_per_case.npz. Run with cwd C:\\EEG."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
sys.path.insert(0, r"C:\EEG")
from finetune_per_patient import patient_files, find_split  # noqa: E402
from evaluate_temporal import smooth                        # noqa: E402

TREES = Path(r"C:\Users\user\AppData\Local\Temp\claude\36c1f29744f93fe6"
             r"\b3889703-629c-4c5a-b583-69f722743ff4\scratchpad\trees")
ME = TREES / "verify" / "critic"
D = np.load(ME / "deploy_tables.npz")
AU = np.load(ME / "auc_per_case.npz")
TG = [float(x) for x in D["targets"]]
cases = sorted(d.name for d in Path(r"C:\EEG\preprocessed_data").iterdir() if d.is_dir())
L = []
rms_bg, ll_bg, nan_ch, auc_ll, auc_rms, auc_nll, ratio_sz = [], [], [], [], [], [], []
for c in cases:
    files = patient_files(c)
    fi, wi = find_split(files, 1)
    bgr, bgl, nanc = [], [], []
    sc_ll, sc_rms, sc_nll, yy = [], [], [], []
    szl = []
    for i, f in enumerate(files):
        with np.load(TREES / "feats" / ("%s__%s.npz" % (c, f.stem))) as d:
            X, y = d["feats"], d["labels"]
        nanc.append(np.isnan(X[:, :, 0]).mean(0))
        lr = np.log(X[:, :, 0])
        ll = np.log(X[:, :, 1])
        bgr.append(np.nanmedian(lr[y == 0], axis=1))
        bgl.append(np.nanmedian(ll[y == 0], axis=1))
        szl.append(np.nanmedian(ll[y == 1], axis=1) if (y == 1).any() else np.zeros(0))
        if i < fi:
            continue
        lo = wi if i == fi else 0
        if len(y) - lo < 10:
            continue
        def mm(a):
            a = np.where(np.isfinite(a), a, np.nan)
            v = np.nanmean(a, 1).astype(np.float64)
            ok = np.isfinite(v)
            if not ok.any():
                return np.zeros(len(v))
            return smooth(np.where(ok, v, v[ok].min()), 5)
        sc_ll.append(mm(ll[lo:]))
        sc_rms.append(mm(lr[lo:]))
        sc_nll.append(mm(np.log(X[lo:, :, 8])))
        yy.append(y[lo:])
    yy = np.concatenate(yy)
    rms_bg.append(np.nanmedian(np.concatenate(bgr)))
    ll_bg.append(np.nanmedian(np.concatenate(bgl)))
    szl = np.concatenate(szl)
    ratio_sz.append(np.exp(np.nanmedian(szl) - ll_bg[-1]))
    nan_ch.append(int(np.round(np.mean(np.concatenate([n[None] for n in nanc]), 0).sum())))
    auc_ll.append(roc_auc_score(yy, np.concatenate(sc_ll)))
    auc_rms.append(roc_auc_score(yy, np.concatenate(sc_rms)))
    auc_nll.append(roc_auc_score(yy, np.concatenate(sc_nll)))
rms_bg, ll_bg, ratio_sz = np.exp(rms_bg), np.exp(ll_bg), np.array(ratio_sz)
L.append("PER CASE: background median RMS (uV) | seizure/background line-length ratio | missing "
         "channels | one-feature test AUC (mean log LL, mean log RMS, mean log LL/RMS) | AUC A0/Tamp/Tfree")
for i, c in enumerate(cases):
    L.append("  %s  RMS %5.1f  LLratio %4.2f  nan_ch %2d  AUC LL %.2f RMS %.2f nLL %.2f | %.2f %.2f %.2f"
             % (c, rms_bg[i], ratio_sz[i], nan_ch[i], auc_ll[i], auc_rms[i], auc_nll[i],
                AU["A0"][i], AU["Tamp"][i], AU["Tfree"][i]))
L.append("  RMS range %.1f-%.1f uV (max/min %.1fx), IQR %.1f-%.1f"
         % (rms_bg.min(), rms_bg.max(), rms_bg.max() / rms_bg.min(), *np.percentile(rms_bg, [25, 75])))
L.append("  median one-feature AUC: LL %.3f, RMS %.3f, LL/RMS %.3f (A0 %.3f, Tamp %.3f, Tfree %.3f)"
         % (np.median(auc_ll), np.median(auc_rms), np.median(auc_nll), np.median(AU["A0"]),
            np.median(AU["Tamp"]), np.median(AU["Tfree"])))
lrms = np.log(rms_bg)
for tg in (0.05, 2.0):
    j = TG.index(tg)
    for g, arms in (("A0", ["A0"]), ("Tamp", ["Tamp_s%d" % s for s in range(3)]),
                    ("Tfree", ["Tfree_s%d" % s for s in range(3)])):
        fpd = np.mean([D[a + "__fp"][:, j] / (D[a + "__sec"] / 86400) for a in arms], 0)
        sen = np.mean([D[a + "__tp"][:, j] / D[a + "__ref"] for a in arms], 0)
        r1, p1 = spearmanr(lrms, np.log1p(fpd))
        r2, p2 = spearmanr(lrms, sen)
        L.append("  target %.2g %-5s: Spearman(bg RMS, FA/day) %+.2f (p %.3f); (bg RMS, sens) %+.2f (p %.3f)"
                 % (tg, g, r1, p1, r2, p2))
d1 = AU["Tamp"] - AU["Tfree"]
r, p = spearmanr(lrms, d1)
L.append("  Spearman(bg RMS, AUC Tamp-Tfree) %+.2f (p %.3f); (nan_ch, AUC Tamp-Tfree) %+.2f"
         % (r, p, spearmanr(nan_ch, d1)[0]))
r, p = spearmanr(lrms, AU["Tamp"] - AU["A0"])
L.append("  Spearman(bg RMS, AUC Tamp-A0) %+.2f (p %.3f)" % (r, p))
(ME / "a4_amp.txt").write_text("\n".join(L) + "\n", encoding="ascii")
np.savez(ME / "a4_amp.npz", cases=np.array(cases), rms_bg=rms_bg, nan_ch=np.array(nan_ch))
print("\n".join(L))
