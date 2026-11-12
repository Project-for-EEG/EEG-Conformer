import sys, time
from multiprocessing import Pool
import numpy as np
sys.dont_write_bytecode = True
sys.path.insert(0, sys.argv[1])
from replicate_score import build, best, per_patient, E0, E1
if __name__ == "__main__":
    path = sys.argv[2]
    t = time.time()
    with Pool(6) as pool:
        out = pool.map(build, [(path, a, 5) for a in ["A0"] + E0 + E1])
    tabs = {a: T for a, k, T in out}
    print("tables in %.0f s, grid sizes %s" % (time.time() - t, [len(T["grid"]) for T in tabs.values()]))
    for b in (5, 10, 25, 50, 100):
        r = {a: best(tabs[a], b) for a in tabs}
        e1 = np.mean([r[a][0] for a in E1]); e0 = np.mean([r[a][0] for a in E0])
        print("%3d/d " % b + " ".join("%s %.3f(%4.1f)" % (a, r[a][0], r[a][1]) for a in tabs) + "  E1-E0 %+.3f" % (e1 - e0))
    for b in (10, 25, 50):
        e1 = np.mean([per_patient(tabs[a], b) for a in E1]); e0 = np.mean([per_patient(tabs[a], b) for a in E0])
        print("per-patient %d/d: E0 %.3f E1 %.3f d %+.3f" % (b, e0, e1, e1 - e0))
