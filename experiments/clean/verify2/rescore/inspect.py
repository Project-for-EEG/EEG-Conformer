import numpy as np, json, sys
NEW = r"C:\Users\user\AppData\Local\Temp\claude\36c1f29744f93fe6\c7848ca6-bfef-414b-a321-d6686c1a43f8\scratchpad\clean"
OLD = r"C:\Users\user\AppData\Local\Temp\claude\36c1f29744f93fe6\b3889703-629c-4c5a-b583-69f722743ff4\scratchpad\trees"
files = [OLD+r"\out\preds_a.npy", OLD+r"\out\preds_b.npy", OLD+r"\out\preds_c.npy", OLD+r"\out\preds_d.npy", OLD+r"\out\preds.npy",
         NEW+r"\trees_out\preds_n1.npy", NEW+r"\trees_out\preds_rel0.npy", NEW+r"\trees_out\preds_rel1.npy", NEW+r"\trees_out\preds_rel2.npy", NEW+r"\trees_out\preds_relf0.npy"]
for f in files:
    d = np.load(f, allow_pickle=True)
    print(f.split("\\")[-1], type(d), getattr(d,'shape',None), d.dtype)
    if d.dtype == object and d.shape == ():
        d = d.item()
    print("  keys:", list(d.keys()) if isinstance(d, dict) else None)
    for arm in d:
        cases = d[arm]
        c0 = sorted(cases)[0]
        b = cases[c0]
        print("   ", arm, len(cases), c0, type(b), len(b), type(b[0]), [type(x) for x in b[0]], [np.asarray(x).shape for x in b[0]], np.asarray(b[0][0]).dtype)
        break
for f in [OLD+r"\out\thresholds_a.json", OLD+r"\out\thresholds_b.json", OLD+r"\out\thresholds_c.json", OLD+r"\out\thresholds_d.json", NEW+r"\trees_out\thresholds_n1.json", NEW+r"\trees_out\thresholds_rel0.json"]:
    t = json.load(open(f))
    print(f.split("\\")[-1], list(t.keys()), {a: len(t[a]) for a in t}, list(t[list(t)[0]]['CHB02'].keys()))
