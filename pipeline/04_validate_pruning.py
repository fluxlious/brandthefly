"""Checks that keeping only connections with >=3 synapses (small enough for the web) gives the same outputs as the full network."""
import numpy as np, json, time
from simlib import build, run, N
G = json.load(open('build/groups.json'))
def summarize(t, i, dur=0.3):
    sel = (t >= 20) & (t < 320); c = np.bincount(i[sel], minlength=N) / dur
    out = {k: round(float(c[v].mean()), 1) for k, v in G.items() if k not in ('DN_all', 'sugar_GRN', 'loom', 'PAM', 'PPL1')}
    out['DN_active'] = int((c[G['DN_all']] > 0).sum()); out['total_active'] = int((c > 0).sum())
    return out
for label, T in [('full', 1), ('pruned >=3 synapses', 3)]:
    W = build(T)
    for sname, st in [('sugar', np.array(G['sugar_GRN'])), ('looming', np.array(G['loom']))]:
        t0 = time.time(); t, i = run(W, st)
        print(label, sname, f'{time.time() - t0:.0f}s', summarize(t, i))
