import numpy as np, os
F = r'C:\Users\user\AppData\Local\Temp\claude\36c1f29744f93fe6\b3889703-629c-4c5a-b583-69f722743ff4\scratchpad\trees\feats'
for n in ['CHB12__chb12_27_labeled','CHB12__chb12_28_labeled','CHB12__chb12_29_labeled','CHB13__chb13_40_labeled','CHB16__chb16_18_labeled','CHB01__chb01_03_labeled']:
    with np.load(os.path.join(F, n + '.npz')) as d:
        x = d['feats']; y = d['labels']
    s = x[:, :, :8]
    nanch = np.isnan(x[:, :, 0]).mean(0)
    fin = s[np.isfinite(s)]
    print(n, x.shape, 'seizure windows', int(y.sum()), '| chans >50% NaN:', int((nanch > 0.5).sum()),
          '| partly NaN chans:', int(((nanch > 0) & (nanch <= 0.5)).sum()),
          '| size<=0 (non-NaN):', int((fin <= 0).sum()), '| min size', float(fin.min()) if fin.size else None)
