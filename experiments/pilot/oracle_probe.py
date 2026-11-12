import numpy as np, sys
def find_runs(m):
    if not m.any(): return []
    p=np.concatenate(([False],m,[False])); e=np.flatnonzero(p[1:]!=p[:-1]); return list(zip(e[::2],e[1::2]))
def smooth(x,k=5):
    if len(x)<k: return x
    ker=np.ones(k); return np.convolve(x,ker,'same')/np.convolve(np.ones_like(x,dtype=float),ker,'same')
def det(pr,t,k=5,m=3):
    pred=smooth(np.asarray(pr,float),k)>=t; out=np.zeros_like(pred)
    for lo,hi in find_runs(pred):
        if hi-lo>=m: out[lo:hi]=True
    return out
d=np.load(sys.argv[1],allow_pickle=True)['data'].item()
pats=sorted(d)
pre={p:[(smooth(np.asarray(b[0],float)),np.asarray(b[1]).astype(bool)) for b in d[p]['blocks']] for p in pats}
def score(t):
    per={}
    for p in pats:
        ev=hit=fa=0; w=0
        for s,tr in pre[p]:
            pred=s>=t; c=np.zeros_like(pred)
            for lo,hi in find_runs(pred):
                if hi-lo>=3: c[lo:hi]=True
            w+=len(c); evs=find_runs(tr); ev+=len(evs); hit+=sum(1 for lo,hi in evs if c[lo:hi].any())
            fa+=sum(1 for lo,hi in find_runs(c) if not tr[lo:hi].any())
        per[p]=(ev,hit,fa,w)
    E=sum(v[0] for v in per.values()); H=sum(v[1] for v in per.values()); F=sum(v[2] for v in per.values()); W=sum(v[3] for v in per.values())
    return H,E,F/(W*2/86400),per
allp=np.concatenate([np.asarray(b[0],float) for p in pats for b in d[p]['blocks']])
grid=np.unique(np.quantile(allp,np.linspace(0.97,0.99999,250)))
res=[(t,)+score(t) for t in grid]
for B in (10,25,50):
    ok=[r for r in res if r[3]<=B]
    t,H,E,fd,per=max(ok,key=lambda r:r[1])
    contrib={p:(v[1],v[2]) for p,v in per.items() if v[1] or v[2]}
    print('budget %d/day: thr %.6f caught %d/%d (%.3f) at %.1f/day'%(B,t,H,E,H/E,fd))
    print('   caught by patient:',{p:v[0] for p,v in contrib.items() if v[0]})
    print('   FA by patient    :',{p:v[1] for p,v in contrib.items() if v[1]})
# per-fold test-prob scale
print('per-patient test prob quantiles (50,99,99.9) and own deployable t:')
for p in pats:
    x=np.concatenate([np.asarray(b[0],float) for b in d[p]['blocks']])
    print('  %s q50 %.3f q99 %.4f q999 %.5f t %.4f'%(p,*np.quantile(x,[.5,.99,.999]),d[p]['t']))
