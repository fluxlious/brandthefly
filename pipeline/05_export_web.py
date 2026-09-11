"""Packs the pruned connectome + positions + groups into compact JSON files the website loads -> ../docs/"""
import numpy as np, json, base64, os
N = 138639
OUT = '../docs'
pre = np.fromfile('build/pre.i32', dtype=np.int32); post = np.fromfile('build/post.i32', dtype=np.int32); w = np.fromfile('build/w.f32', dtype=np.float32)
k = np.abs(w) >= 3; pre, post, w = pre[k], post[k], w[k]
o = np.lexsort((post, pre)); pre, post, w = pre[o], post[o], w[o]
counts = np.bincount(pre, minlength=N).astype(np.uint32)
starts = np.concatenate([[0], np.cumsum(counts.astype(np.int64))[:-1]]).astype(np.int64)
prev = np.empty_like(post); prev[0] = 0; prev[1:] = post[:-1]
first = np.zeros(len(post), bool); first[starts[counts > 0]] = True
delta = np.where(first, post, post - prev).astype(np.int64)            # per-row delta encoding
nb = np.where(delta < 128, 1, np.where(delta < 16384, 2, 3)).astype(np.int64)
assert (delta < (1 << 21)).all()
buf = np.zeros(int(nb.sum()), np.int64); pos = np.concatenate([[0], np.cumsum(nb)[:-1]]).astype(np.int64)
buf[pos] = (delta & 127) | np.where(nb > 1, 128, 0)                    # LEB128 varints
m2 = nb > 1; buf[pos[m2] + 1] = ((delta[m2] >> 7) & 127) | np.where(nb[m2] > 2, 128, 0)
m3 = nb > 2; buf[pos[m3] + 2] = (delta[m3] >> 14) & 127
wi = np.round(w).astype(np.int32); big = np.abs(wi) > 127
w8 = np.where(big, -128, wi).astype(np.int8); wover = wi[big].astype(np.int16)   # int8 weights, rare big ones in an overflow list
P = np.load('build/pos.npz', allow_pickle=True); x, y, ok = P['x'], P['y'], P['ok']
xy = np.stack([np.where(ok, np.round(np.nan_to_num(x) * 10), 0), np.where(ok, np.round(np.nan_to_num(y) * 10), 0)], 1).astype(np.uint16).ravel()
def dump(name, arr):
    b = arr.tobytes(); json.dump({'bytes': len(b), 'b64': base64.b64encode(b).decode()}, open(os.path.join(OUT, name), 'w'))
os.makedirs(OUT, exist_ok=True)
dump('counts.json', counts); dump('idx.json', buf.astype(np.uint8)); dump('w8.json', w8); dump('wover.json', wover); dump('pos.json', xy)
json.dump(json.load(open('build/groups.json')), open(os.path.join(OUT, 'groups.json'), 'w'))
print('edges kept:', len(post))
