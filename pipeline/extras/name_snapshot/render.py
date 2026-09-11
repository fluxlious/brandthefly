import numpy as np
from scipy.ndimage import gaussian_filter
from PIL import Image, ImageDraw, ImageFont
import imageio.v2 as imageio
P=np.load('../../build/pos.npz',allow_pickle=True); x,y,ok=P['x'],P['y'],P['ok']
S=np.load('spikes.npz'); t,i,stim=S['t'],S['i'],S['stim']
N=len(x); isstim=np.zeros(N,bool); isstim[stim]=True
sc=2.0; pad=60
xmin,xmax=np.nanmin(x),np.nanmax(x); ymin,ymax=np.nanmin(y),np.nanmax(y)
W=int((xmax-xmin)*sc)+2*pad; H=int((ymax-ymin)*sc)+2*pad
px=np.where(ok,((np.nan_to_num(x)-xmin)*sc+pad).astype(int),0); py=np.where(ok,((np.nan_to_num(y)-ymin)*sc+pad).astype(int),0)
def dens(idx, weights=None):
    img=np.zeros((H,W),np.float32)
    np.add.at(img,(py[idx],px[idx]),1 if weights is None else weights)
    return img
allidx=np.where(ok)[0]
base=gaussian_filter(dens(allidx),1.2); base=np.log1p(base)/np.log1p(base).max()
fB=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',30)
fR=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',22)
def frame(t0,t1,label=True,norm=None):
    sel=(t>=t0)&(t<t1)&ok[i]
    cnt=np.bincount(i[sel],minlength=N).astype(np.float32)
    dur=(t1-t0)/1000.
    act=np.where(cnt>0)[0]
    sA=act[isstim[act]]; sB=act[~isstim[act]]
    a=gaussian_filter(dens(sA,np.minimum(cnt[sA]/(60*dur),1.0)),1.3)
    b=gaussian_filter(dens(sB,np.minimum(cnt[sB]/(40*dur),1.0)),1.3)
    a=1-np.exp(-a*16); b=(1-np.exp(-b*9))*0.9
    rgb=np.zeros((H,W,3),np.float32)+np.array([0.02,0.025,0.04])
    rgb+=(base**1.2)[...,None]*np.array([0.16,0.16,0.17])
    rgb=rgb*(1-b[...,None])+b[...,None]*np.array([1.0,0.45,0.15])
    rgb=rgb*(1-a[...,None])+a[...,None]*np.array([0.30,0.88,1.0])
    glow=gaussian_filter(rgb,(7,7,0))*0.5
    rgb=np.clip(rgb+glow,0,1)
    im=Image.fromarray((rgb*255).astype(np.uint8))
    top=Image.new('RGB',(W,H+90),(5,6,10)); top.paste(im,(0,70)); d=ImageDraw.Draw(top)
    d.text((pad,18),'Simulated fruit fly brain · 138,639 real neurons (FlyWire)',font=fB,fill=(220,225,235))
    d.text((W-pad-260,24),f't = {t0:.0f}–{t1:.0f} ms',font=fR,fill=(160,170,190))
    y0=H+70-10
    d.ellipse((pad,y0-2,pad+16,y0+14),fill=(64,216,255)); d.text((pad+26,y0-6),'neurons I stimulated (letter-shaped)',font=fR,fill=(200,205,215))
    d.ellipse((pad+470,y0-2,pad+486,y0+14),fill=(255,115,40)); d.text((pad+496,y0-6),'neurons the real wiring activated on its own',font=fR,fill=(200,205,215))
    stim_state='stimulus ON' if t0<220 and t1>20 else 'stimulus OFF'
    d.text((W-pad-260,H+70-16),stim_state,font=fR,fill=(120,230,140) if 'ON' in stim_state else (150,150,160))
    return top
frame(120,200).save('snapshot.png')
frame(20,40).save('f_early.png'); frame(300,320).save('f_late.png')
frames=[np.array(frame(s,s+20).resize((W//2,(H+90)//2),Image.LANCZOS)) for s in range(0,400,10)]
imageio.mimsave('animation.gif',frames,duration=120,loop=0)
# stats
for a_,b_ in [(20,220),(120,200),(250,400)]:
    sel=(t>=a_)&(t<b_); u=np.unique(i[sel]); print(a_,b_,'stim active',isstim[u].sum(),'recruited',(~isstim[u]).sum())
