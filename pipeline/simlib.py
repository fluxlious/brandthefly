"""Leaky integrate-and-fire model of the whole fly brain (Shiu et al. 2024 parameters), numpy/scipy version."""
import numpy as np, scipy.sparse as sp
N=138639
pre=np.fromfile('build/pre.i32',dtype=np.int32); post=np.fromfile('build/post.i32',dtype=np.int32); w=np.fromfile('build/w.f32',dtype=np.float32)
def build(T=1, gain=1.0):
    k=np.abs(w)>=T
    return sp.csc_matrix((w[k]*0.275*gain,(post[k],pre[k])),shape=(N,N))
def run(W, stim, T_ms=500, on=20, off=320, rate=150., seed=0):
    rng=np.random.default_rng(seed)
    dt=0.1; v0=-52.; vth=-45.; tm=20.; tau=5.; trfc=2.2; dly=18
    v=np.full(N,v0,np.float32); g=np.zeros(N,np.float32); ref=np.zeros(N,np.float32)
    buf=np.zeros((dly,N),np.float32); stim=np.asarray(stim); p=rate*dt/1000
    isst=np.zeros(N,bool); isst[stim]=True
    ts=[]; is_=[]
    for s in range(int(T_ms/dt)):
        g+=buf[s%dly]; buf[s%dly]=0
        a=ref<=0
        v[a]+=(dt/tm)*(v0-v[a]+g[a]); g[a]-=(dt/tau)*g[a]
        if on/dt<=s<off/dt and len(stim):
            v[stim[rng.random(len(stim))<p]]+=68.75
        ref-=dt
        f=np.where(v>vth)[0]
        if len(f):
            v[f]=v0; g[f]=0; ref[f]=np.where(isst[f],0.,trfc)
            ts.append(np.full(len(f),s*dt,np.float32)); is_.append(f)
            buf[s%dly]+=np.asarray(W[:,f].sum(axis=1)).ravel()
    return np.concatenate(ts) if ts else np.zeros(0), np.concatenate(is_) if is_ else np.zeros(0,int)
