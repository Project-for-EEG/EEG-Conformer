import numpy as np
def find_runs(m):
    if not m.any(): return []
    p=np.concatenate(([False],m,[False])); e=np.flatnonzero(p[1:]!=p[:-1]); return list(zip(e[::2],e[1::2]))
def smooth(x,k=5):
    if len(x)<k: return x
    ker=np.ones(k); return np.convolve(x,ker,'same')/np.convolve(np.ones_like(x,dtype=float),ker,'same')
u=np.load('endtoend_uncalibrated.npz',allow_pickle=True)['data'].item()
a=np.load('endtoend_lopoA.npz',allow_pickle=True)['data'].item()
pats=[p for p in sorted(u) if a[p].get('model')=='base']
def prep(d): return {p:[(smooth(np.asarray(b[0],float)),np.asarray(b[1]).astype(bool)) for b in d[p]['blocks']] for p in pats}
def curve(pre):
    allp=np.concatenate([s for p in pats for s,_ in pre[p]])
    out=[]
    for t in np.unique(np.quantile(allp,np.linspace(0.95,0.99999,200))):
        H=E=F=W=0
        for p in pats:
            for s,tr in pre[p]:
                pred=s>=t; c=np.zeros_like(pred)
                for lo,hi in find_runs(pred):
                    if hi-lo>=3: c[lo:hi]=True
                W+=len(c); ev=find_runs(tr); E+=len(ev); H+=sum(1 for lo,hi in ev if c[lo:hi].any())
                F+=sum(1 for lo,hi in find_runs(c) if not tr[lo:hi].any())
        out.append((F/(W*2/86400),H,E))
    return out
cu,ca=curve(prep(u)),curve(prep(a))
print('patients (base in both):',len(pats))
for B in (10,25,50,100):
    bu=max([c for c in cu if c[0]<=B],key=lambda c:c[1]); ba=max([c for c in ca if c[0]<=B],key=lambda c:c[1])
    print('<=%3d/day strict  lopo %d/%d (%.3f)   lopoA %d/%d (%.3f)   diff %+.3f'%(B,bu[1],bu[2],bu[1]/bu[2],ba[1],ba[2],ba[1]/ba[2],ba[1]/ba[2]-bu[1]/bu[2]))
