"""Packs the hero replay spikes + output-neuron rates for the website -> ../docs/hero.json"""
import numpy as np, json, base64
d = np.load('build/hero_spikes.npz'); s, n, f = d['s'], d['n'], d['f']
BIN = 20; steps = int(d['steps']); nf = (steps + BIN - 1) // BIN     # 2 ms frames
G = json.load(open('build/groups.json'))
fr = (s // BIN).astype(np.int64)
key = np.unique(fr * (1 << 20) + n.astype(np.int64) * 2 + f.astype(np.int64))
fr2 = key >> 20; val = key & ((1 << 20) - 1)
counts = np.bincount(fr2, minlength=nf).astype(np.uint16)
nid = val >> 1; flag = val & 1
starts = np.concatenate([[0], np.cumsum(counts.astype(np.int64))[:-1]]).astype(np.int64)
prev = np.empty_like(nid); prev[0] = 0; prev[1:] = nid[:-1]
first = np.zeros(len(nid), bool); first[starts[counts > 0]] = True
code = (np.where(first, nid, nid - prev) * 2 + flag).astype(np.int64)
nb = np.where(code < 128, 1, np.where(code < 16384, 2, 3)).astype(np.int64)
buf = np.zeros(int(nb.sum()), np.int64); pos = np.concatenate([[0], np.cumsum(nb)[:-1]]).astype(np.int64)
buf[pos] = (code & 127) | np.where(nb > 1, 128, 0)
m2 = nb > 1; buf[pos[m2] + 1] = ((code[m2] >> 7) & 127) | np.where(nb[m2] > 2, 128, 0)
m3 = nb > 2; buf[pos[m3] + 2] = (code[m3] >> 14) & 127
keys = ['DNa01_L', 'DNa01_R', 'DNa02_L', 'DNa02_R', 'MDN_L', 'MDN_R', 'GF_L', 'GF_R', 'MN9', 'proboscis_MN', 'ingest_MN', 'antenna_MN_L', 'antenna_MN_R', 'neck_MN_L', 'neck_MN_R']
rates = {}
for k in keys:
    mm = np.isin(n, G[k]); cc = np.bincount(s[mm], minlength=steps + 1)[:steps]; cum = np.concatenate([[0], np.cumsum(cc)])
    e = np.minimum(steps, (np.arange(nf) + 1) * BIN); a = np.maximum(0, e - 400)                  # 40 ms rate window
    rates[k] = base64.b64encode(np.clip(np.round((cum[e] - cum[a]) / (len(G[k]) * (e - a) * 0.1 / 1000)), 0, 65535).astype(np.uint16).tobytes()).decode()
gf = np.isin(n, G['GF_L'] + G['GF_R']); gfc = np.bincount(s[gf] // BIN, minlength=nf)[:nf].astype(np.uint16)
meta = dict(bin_ms=BIN / 10, frames=int(nf), on_ms=int(d['on']) / 10, off_ms=int(d['off']) / 10, steps=steps, yc=float(d['yc']), dur=float(d['dur']))
json.dump({'meta': meta, 'counts': base64.b64encode(counts.tobytes()).decode(), 'b64': base64.b64encode(buf.astype(np.uint8).tobytes()).decode(),
           'rates': rates, 'gf': base64.b64encode(gfc.tobytes()).decode()}, open('../docs/hero.json', 'w'))
print(meta)
