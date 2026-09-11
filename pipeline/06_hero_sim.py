"""Runs the hero replay: a 5x7 LED text slides through the brain (2.4 s of brain time). Usage: python 06_hero_sim.py "BRAND THE FLY" """
import numpy as np, json, time, sys
from simlib import build, N
FONT=json.load(open('font57.json'))
P=np.load('build/pos.npz',allow_pickle=True); X,Y,OK=P['x'],P['y'],P['ok']
xmin,xmax,ymin,ymax=np.nanmin(X),np.nanmax(X),np.nanmin(Y),np.nanmax(Y)
G=json.load(open('build/groups.json'))
TEXT=(sys.argv[1] if len(sys.argv)>1 else 'BRAND THE FLY').upper(); PITCH=20; RAD=8.5; ON0=200; RATE=300
cols=[]
for ch in TEXT:
    g=FONT.get(ch,FONT[' '])
    for c in range(5): cols.append([g[r*5+c]=='1' for r in range(7)])
    cols.append([False]*7)
# band: maximize neuron count in rows
okx=np.where(OK)[0]
best=None
for yc in np.arange(ymin+3*PITCH+RAD, ymax-3*PITCH-RAD, 4):
    sc=0
    for r in range(7):
        yr=yc+(r-3)*PITCH; sc+=np.sum(np.abs(Y[okx]-yr)<=RAD)
    if best is None or sc>best[0]: best=(sc,yc)
yc=best[1]
textW=len(cols)*PITCH; x0=xmax+RAD+4; dist=(xmax-xmin)+textW+2*RAD+8
dur=2400.0; vel=dist/dur
rng=np.random.default_rng(3); order=rng.permutation(N)
taken=set(); iN=[];iS=[];iE=[]
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
        if not cols[c][r]: continue
        tS=(x0+c*PITCH-X[i]-h)/vel; tE=(x0+c*PITCH-X[i]+h)/vel
        if tE<0 or tS>dur: continue
        iN.append(i); iS.append(ON0+int(round(max(0,tS)*10))); iE.append(ON0+int(round(min(dur,tE)*10)))
iN=np.array(iN); iS=np.array(iS); iE=np.array(iE)
off=ON0+int(round(dur*10)); STEPS=off+1500
print('yc',yc,'dur',dur,'steps',STEPS,'intervals',len(iN),'neurons',len(np.unique(iN)))
W=build(3)
dt=0.1; v0=-52.; vth=-45.; tm=20.; tau=5.; dly=18
v=np.full(N,v0,np.float32); g=np.zeros(N,np.float32); ref=np.zeros(N,np.float32); buf=np.zeros((dly,N),np.float32)
drive=np.zeros(N,np.int32)
startB=[[] for _ in range(STEPS+1)]; endB=[[] for _ in range(STEPS+1)]
for n,s0,e0 in zip(iN,iS,iE): startB[s0].append(n); endB[e0].append(n)
p=RATE*dt/1000; spk_s=[]; spk_n=[]; spk_f=[]
t0=time.time()
for s in range(STEPS):
    if startB[s]: np.add.at(drive,startB[s],1)
    if endB[s]: np.add.at(drive,endB[s],-1)
    g+=buf[s%dly]; buf[s%dly]=0
    a=ref<=0
    v[a]+=(dt/tm)*(v0-v[a]+g[a]); g[a]-=(dt/tau)*g[a]
    act=np.where(drive>0)[0]
    if len(act): v[act[np.random.random(len(act))<p]]+=68.75
    ref-=dt
    f=np.where(v>vth)[0]
    if len(f):
        v[f]=v0; g[f]=0; ref[f]=np.where(drive[f]>0,0.,2.2)
        spk_s.append(np.full(len(f),s,np.int32)); spk_n.append(f.astype(np.int32)); spk_f.append((drive[f]>0).astype(np.int8))
        buf[s%dly]+=np.asarray(W[:,f].sum(axis=1)).ravel()
    if s%3000==0: print(s, round(time.time()-t0,1), flush=True)
S=np.concatenate(spk_s); Nn=np.concatenate(spk_n); F=np.concatenate(spk_f)
np.savez('build/hero_spikes.npz',s=S,n=Nn,f=F,steps=STEPS,on=ON0,off=off,yc=yc,dur=dur)
print('spikes',len(S),'stim',F.sum())
# reaction report over stimulus window
sel=(S>=ON0)&(S<off); durs=(off-ON0)*dt/1000
c=np.bincount(Nn[sel],minlength=N)/durs
rep={k:round(float(c[G[k]].mean()),1) for k in ['DNa01_L','DNa01_R','DNa02_L','DNa02_R','MDN_L','MDN_R','GF_L','GF_R','MN9','proboscis_MN','ingest_MN','antenna_MN_L','antenna_MN_R','neck_MN_L','neck_MN_R']}
rep['DN_active']=int((c[G['DN_all']]>0).sum()); rep['recruited']=int(len(np.setdiff1d(np.unique(Nn[sel]),iN))); rep['stim_neurons']=int(len(np.unique(iN)))
rep['gf_spikes']=int(np.isin(Nn[sel],G['GF_L']+G['GF_R']).sum())
print(rep); json.dump(rep,open('build/hero_report.json','w'))
