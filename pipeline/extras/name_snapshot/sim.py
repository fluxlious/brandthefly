import numpy as np, scipy.sparse as sp, time, sys
from PIL import Image, ImageDraw, ImageFont
N=138639
P=np.load('../../build/pos.npz', allow_pickle=True); x,y,ok=P['x'],P['y'],P['ok']
# display coords: dorsal up
X=x; Y=-y
# ---- letter mask in brain coordinates (y down, like the image) ----
xmin,xmax=np.nanmin(x),np.nanmax(x); ymin,ymax=np.nanmin(y),np.nanmax(y)
Wpx=int(xmax-xmin)+1; Hpx=int(ymax-ymin)+1
img=Image.new('L',(Wpx,Hpx),0); d=ImageDraw.Draw(img)
fs=int(sys.argv[1]) if len(sys.argv)>1 else 300
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', fs)
for ch,(cx,cy) in zip('CEM',[(190,222),(415,205),(688,222)]):
    d.text((cx,cy),ch,fill=255,font=font,anchor='mm')
mask=np.array(img)>0
col=np.where(ok,np.clip(np.nan_to_num(x-xmin).astype(int),0,Wpx-1),0)
row=np.where(ok,np.clip(np.nan_to_num(y-ymin).astype(int),0,Hpx-1),0)
inmask=ok & mask[row,col]
cell=float(sys.argv[3]) if len(sys.argv)>3 else 4.0
rng=np.random.default_rng(1)
cand=np.where(inmask)[0]; cand=cand[rng.permutation(len(cand))]
key=(col[cand]//cell).astype(np.int64)*100000+(row[cand]//cell).astype(np.int64)
_,first=np.unique(key,return_index=True)
stim=np.sort(cand[first])
print('in letters:',inmask.sum(),'stimulated:',len(stim))
np.save('mask.npy',mask); np.save('stim.npy',stim)
# ---- network (Shiu et al. 2024 LIF params) ----
pre=np.fromfile('../../build/pre.i32',dtype=np.int32); post=np.fromfile('../../build/post.i32',dtype=np.int32); w=np.fromfile('../../build/w.f32',dtype=np.float32)
W=sp.csc_matrix((w*0.275,(post,pre)),shape=(N,N))  # mV per spike
dt=0.1; v0=-52.; vth=-45.; tm=20.; tau=5.; trfc=2.2; dly=18  # steps
T=int(float(sys.argv[4]) if len(sys.argv)>4 else 400)/dt
T=int(T); stim_on=int(20/dt); stim_off=int(220/dt)
v=np.full(N,v0,np.float32); g=np.zeros(N,np.float32); ref=np.zeros(N,np.float32)
buf=[np.zeros(N,np.float32) for _ in range(dly)]
rate=150.; pspk=rate*dt/1000.
isstim=np.zeros(N,bool); isstim[stim]=True
spk_t=[]; spk_i=[]
t0=time.time()
for s in range(T):
    g+=buf[s%dly]; buf[s%dly][:]=0
    active=ref<=0
    v[active]+=(dt/tm)*(v0-v[active]+g[active])
    g[active]-=(dt/tau)*g[active]
    if stim_on<=s<stim_off:
        hits=stim[rng.random(len(stim))<pspk]
        v[hits]+=0.275*250
    ref-=dt
    fired=np.where(v>vth)[0]
    if len(fired):
        v[fired]=v0; g[fired]=0; ref[fired]=np.where(isstim[fired],0.,trfc)
        spk_t.append(np.full(len(fired),s,np.int32)); spk_i.append(fired.astype(np.int32))
        inp=np.asarray(W[:,fired].sum(axis=1)).ravel().astype(np.float32)
        buf[(s+dly)%dly]+=inp
    if s%1000==0: print(s*dt,'ms',len(fired),round(time.time()-t0,1),flush=True)
spk_t=np.concatenate(spk_t)*dt; spk_i=np.concatenate(spk_i)
np.savez('spikes.npz',t=spk_t,i=spk_i,stim=stim)
print('total spikes',len(spk_t),'unique neurons',len(np.unique(spk_i)),'non-stim active',len(np.setdiff1d(np.unique(spk_i),stim)))
