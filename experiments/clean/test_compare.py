import sys, warnings
sys.path.insert(0, r"C:\Users\user\AppData\Local\Temp\claude\36c1f29744f93fe6\c7848ca6-bfef-414b-a321-d6686c1a43f8\scratchpad\clean")
import clean_score as cs
warnings.simplefilter("ignore")
P, T, src = cs.load_arms()
cases = sorted(P["A0"])
keep = {c: [True] * len(P["A0"][c]) for c in cases}
D = {a: cs.deploy_table(P[a], T[a], cases, keep) for a in ["A0", "Tamp_s0", "Tamp_s1", "Tamp_s2"]}
b = cs.Bench(D, cases, cases, "168-test")
r = b.compare("Tamp", "A0")
cs.show(r)
