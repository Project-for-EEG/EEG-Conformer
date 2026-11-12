import numpy as np
OLD = r"C:\Users\user\AppData\Local\Temp\claude\36c1f29744f93fe6\b3889703-629c-4c5a-b583-69f722743ff4\scratchpad\trees"
d = np.load(OLD + r"\feats\CHB12__chb12_27_labeled.npz", allow_pickle=True)
for k in d.files:
    a = d[k]; print(k, a.shape, a.dtype, a.ravel()[:5] if a.dtype != object else a)
d = np.load(r"preprocessed_data\CHB12\chb12_27_labeled.npz", allow_pickle=True)
print(d.files)
for k in d.files:
    print(k, d[k].shape if hasattr(d[k],'shape') else None, d[k].dtype)
