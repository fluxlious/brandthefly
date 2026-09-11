"""Fly mood test: runs brands/shapes like the website clip maker and prints appetite, curiosity, dopamine, neurons lit.
Usage: python 08_fly_mood_test.py "CEM,PIZZA,shape:ring,sugar" """
import numpy as np, json, time, sys
from simlib import build, N
FONT=json.load(open('font57.json'))
P=np.load('build/pos.npz',allow_pickle=True); X,Y,OK=P['x'],P['y'],P['ok']
xmin,xmax,ymin,ymax=np.nanmin(X),np.nanmax(X),np.nanmin(Y),np.nanmax(Y)
G=json.load(open('build/groups.json')); pam=np.array(G['PAM']); ppl1=np.array(G['PPL1'])
W=build(3); yc=267.86; ON0=200; DUR=1600; RATE=300
rng=np.random.default_rng(5); order=rng.permutation(N)
def plan_text(text):
    PITCH=20; RAD=8.5; cols=[]
    for ch in text:
        g=FONT.get(ch,FONT[' '])
        for c in range(5): cols.append([g[r*5+c]=='1' for r in range(7)])
        cols.append([False]*7)
    x0=xmax+17; dist=(xmax-xmin)+len(cols)*PITCH+34; vel=dist/DUR
    taken=set(); out=[]
    for i in order:
        if not OK[i]: continue
        r=int(round((Y[i]-yc)/PITCH))+3
        if r<0 or r>6: continue
        dy=Y[i]-(yc+(r-3)*PITCH)
        if abs(dy)>RAD: continue
        cell=(int((Y[i]-ymin)//4),int((X[i]-xmin)//4))
        if cell in taken: continue
        taken.add(cell); h=np.sqrt(RAD*RAD-dy*dy)
        for c in range(len(cols)):
            if cols[c][r]:
                tS=(x0+c*PITCH-X[i]-h)/vel; tE=(x0+c*PITCH-X[i]+h)/vel
                if tE<0 or tS>DUR: continue
                out.append((i,ON0+int(round(max(0,tS)*10)),ON0+int(round(min(DUR,tE)*10))))
    return out
def plan_shape(kind):
    # simple logo silhouettes sliding: 'block', 'ring', 'thin'
    LH=250; mw,mh=100,100
    yy,xx=np.mgrid[0:mh,0:mw]
    if kind=='block': m=np.ones((mh,mw),bool)
    elif kind=='ring': rr=np.hypot(xx-50,yy-50); m=(rr<48)&(rr>34)
    elif kind=='thin': m=np.abs(yy-50)<5
    elif kind=='top': m=yy<35
    elif kind=='bottom': m=yy>65
    Wd=min((xmax-xmin)*0.95,LH*mw/mh); lh=Wd*mh/mw; top=yc-lh/2; x0=xmax+17; dist=(xmax-xmin)+Wd+34; vel=dist/DUR
    taken=set(); out=[]
    for i in order:
        if not OK[i] or abs(Y[i]-yc)>LH/2+2: continue
        cell=(int((Y[i]-ymin)//4),int((X[i]-xmin)//4))
        if cell in taken: continue
        taken.add(cell)
        q=(Y[i]-top)/lh
        if q<0 or q>=1: continue
        row=m[int(q*mh)]; a=-1
        for cu in range(mw+1):
            on=cu<mw and row[cu]
            if on and a<0: a=cu
            if (not on) and a>=0:
                tS=(a/mw*Wd+x0-X[i])/vel; tE=(cu/mw*Wd+x0-X[i])/vel; a=-1
                if tE<0 or tS>DUR: continue
                out.append((i,ON0+int(round(max(0,tS)*10)),ON0+int(round(min(DUR,tE)*10))))
    return out
def sim(intervals):
    STEPS=ON0+DUR*10+900; dt=0.1; v0=-52.; vth=-45.; dly=18
    v=np.full(N,v0,np.float32); g=np.zeros(N,np.float32); ref=np.zeros(N,np.float32); buf=np.zeros((dly,N),np.float32)
    drive=np.zeros(N,np.int32); sB=[[] for _ in range(STEPS+1)]; eB=[[] for _ in range(STEPS+1)]
    for n,a,b in intervals: sB[a].append(n); eB[b].append(n)
    p=RATE*dt/1000; cp=np.zeros(N); lit=np.zeros(N,bool); stimd=np.zeros(N,bool)
    for s in range(STEPS):
        if sB[s]: np.add.at(drive,sB[s],1)
        if eB[s]: np.add.at(drive,eB[s],-1)
        g+=buf[s%dly]; buf[s%dly]=0; a=ref<=0
        v[a]+=(dt/20)*(v0-v[a]+g[a]); g[a]-=(dt/5)*g[a]
        act=np.where(drive>0)[0]
        if len(act): v[act[np.random.random(len(act))<p]]+=68.75
        ref-=dt; f=np.where(v>vth)[0]
        if len(f):
            v[f]=v0; g[f]=0; ref[f]=np.where(drive[f]>0,0.,2.2)
            if ON0<=s<ON0+DUR*10:
                nf=f[drive[f]==0]; cp[nf]+=1; lit[f]=True; stimd[f[drive[f]>0]]=True
            buf[s%dly]+=np.asarray(W[:,f].sum(axis=1)).ravel()
    dur=DUR/1000
    allc=cp  # recruited-only counts
    grp=lambda k: allc[G[k]].mean()/dur
    return cp[pam].mean()/dur, cp[ppl1].mean()/dur, int(lit.sum()), int(stimd.sum()), {k:round(grp(k),1) for k in ['MN9','antenna_MN_L','antenna_MN_R','DNa02_L','DNa02_R','GF_L','GF_R','ingest_MN']}
tests=sys.argv[1].split(',')
for t in tests:
    t0=time.time()
    if t=='sugar': iv=[(int(i),ON0,ON0+DUR*10) for i in G['sugar_GRN']]
    else: iv=plan_shape(t[6:]) if t.startswith('shape:') else plan_text(t)
    pa,pp,nl,ns,ex=sim(iv)
    print(t, ex, 'lit', nl, 'stim', ns, 'PAM', round(pa,1), 'PPL1', round(pp,1), 'score', round(100*pa/(pa+pp+1e-9),1), round(time.time()-t0),'s', flush=True)
