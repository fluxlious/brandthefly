// Re-simulates the hero replay (a 5x7 LED text sliding through the brain) from the website's own data files
// in ../docs and writes ../docs/hero.json. Same model and parameters as 06_hero_sim.py + 07_pack_hero.py, but
// needs no raw download: it reads counts/idx/w8/wover/pos/groups.json exactly like the site does.
//
//   node 09_hero_sim_web.mjs "BRAND THE FLY"        # band placed automatically (densest 7 rows)
//   node 09_hero_sim_web.mjs "BRAND THE FLY" 243    # band centred at y = 243 um
//   node 09_hero_sim_web.mjs "BRAND THE FLY" 243 out.json   # write somewhere else (default ../docs/hero.json)
//
// Takes about a minute. Prints the fly-mood numbers for the run (paste into CONFIG.verdicts in docs/index.html).
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const DOCS = path.join(here, '..', 'docs');
const N = 138639;
const TEXT = (process.argv[2] || 'BRAND THE FLY').toUpperCase();
const YC_ARG = process.argv[3] ? +process.argv[3] : null;
const OUT = process.argv[4] || path.join(DOCS, 'hero.json');
const PITCH = 20, RAD = 8.5, ON0 = 200, RATE = 300, DUR = 2400;      // as in 06_hero_sim.py
const BIN = 20;                                                        // 2 ms frames, as in 07_pack_hero.py

/* ---------- load the website's data files ---------- */
const t0 = performance.now();
const say = m => console.log(`[${((performance.now() - t0) / 1000).toFixed(1)}s] ${m}`);
const b64 = f => { const o = JSON.parse(fs.readFileSync(path.join(DOCS, f), 'utf8')); const b = Buffer.from(o.b64, 'base64'); return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength); };
const counts = new Uint32Array(b64('counts.json')), rowptr = new Uint32Array(N + 1);
for (let i = 0; i < N; i++) rowptr[i + 1] = rowptr[i] + counts[i];
const E = rowptr[N], bytes = new Uint8Array(b64('idx.json')), idx = new Uint32Array(E);
for (let r = 0, p = 0; r < N; r++) {
  let prev = 0; const a = rowptr[r], b = rowptr[r + 1];
  for (let j = a; j < b; j++) {
    let c = bytes[p++], val = c & 127;
    if (c & 128) { c = bytes[p++]; val |= (c & 127) << 7; if (c & 128) { c = bytes[p++]; val |= (c & 127) << 14; } }
    prev = (j === a) ? val : prev + val; idx[j] = prev;
  }
}
const w8 = new Int8Array(b64('w8.json')), wo = new Int16Array(b64('wover.json')), W = new Float32Array(E);
for (let j = 0, k = 0; j < E; j++) { let v = w8[j]; if (v === -128) v = wo[k++]; W[j] = v * 0.275; }
const pos = new Uint16Array(b64('pos.json')), X = new Float32Array(N), Y = new Float32Array(N), OK = new Uint8Array(N);
let xmin = 1e9, xmax = -1e9, ymin = 1e9, ymax = -1e9;
for (let i = 0; i < N; i++) { const x = pos[2 * i] / 10, y = pos[2 * i + 1] / 10; if (x > 0) { OK[i] = 1; X[i] = x; Y[i] = y; xmin = Math.min(xmin, x); xmax = Math.max(xmax, x); ymin = Math.min(ymin, y); ymax = Math.max(ymax, y); } }
const G = JSON.parse(fs.readFileSync(path.join(DOCS, 'groups.json'), 'utf8'));
const FONT = JSON.parse(fs.readFileSync(path.join(here, 'font57.json'), 'utf8'));
say(`wiring: ${E.toLocaleString()} connections; brain x ${xmin.toFixed(0)}-${xmax.toFixed(0)}, y ${ymin.toFixed(0)}-${ymax.toFixed(0)} um`);

/* ---------- the LED panel: 7 rows of neurons, one per 4 um cell ---------- */
const cols = [];
for (const ch of TEXT) { const g = FONT[ch] || FONT[' ']; for (let c = 0; c < 5; c++) cols.push([0, 1, 2, 3, 4, 5, 6].map(r => g[r * 5 + c] === '1')); cols.push([false, false, false, false, false, false, false]); }
let yc;
if (YC_ARG !== null) yc = YC_ARG;
else {  // densest 7-row band, as in 06_hero_sim.py
  let best = -1;
  for (let y = ymin + 3 * PITCH + RAD; y <= ymax - 3 * PITCH - RAD; y += 4) {
    let sc = 0;
    for (let r = 0; r < 7; r++) { const yr = y + (r - 3) * PITCH; for (let i = 0; i < N; i++) if (OK[i] && Math.abs(Y[i] - yr) <= RAD) sc++; }
    if (sc > best) { best = sc; yc = y; }
  }
}
const textW = cols.length * PITCH, x0 = xmax + RAD + 4, dist = (xmax - xmin) + textW + 2 * RAD + 8, vel = dist / DUR;
let seed = 3; const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
const order = new Int32Array(N); for (let i = 0; i < N; i++) order[i] = i;
for (let i = N - 1; i > 0; i--) { const j = Math.floor(rnd() * (i + 1)); const t = order[i]; order[i] = order[j]; order[j] = t; }
// The panel never drives the fly's output neurons directly (descending + motor neurons); they only fire through the wiring.
const SKIP = new Uint8Array(N);
for (const k of ['DN_all', 'DNa01_L', 'DNa01_R', 'DNa02_L', 'DNa02_R', 'MDN_L', 'MDN_R', 'GF_L', 'GF_R', 'MN9', 'proboscis_MN', 'ingest_MN', 'antenna_MN_L', 'antenna_MN_R', 'neck_MN_L', 'neck_MN_R']) for (const i of G[k]) SKIP[i] = 1;
const taken = new Set(), iN = [], iS = [], iE = [], stimSet = new Set();
for (const i of order) {
  if (!OK[i] || SKIP[i]) continue;
  const r = Math.round((Y[i] - yc) / PITCH) + 3; if (r < 0 || r > 6) continue;
  const dy = Y[i] - (yc + (r - 3) * PITCH); if (Math.abs(dy) > RAD) continue;
  const key = Math.floor((Y[i] - ymin) / 4) * 4096 + Math.floor((X[i] - xmin) / 4); if (taken.has(key)) continue; taken.add(key);
  const h = Math.sqrt(RAD * RAD - dy * dy);
  for (let c = 0; c < cols.length; c++) {
    if (!cols[c][r]) continue;
    const tS = (x0 + c * PITCH - X[i] - h) / vel, tE = (x0 + c * PITCH - X[i] + h) / vel;
    if (tE < 0 || tS > DUR) continue;
    iN.push(i); iS.push(ON0 + Math.round(Math.max(0, tS) * 10)); iE.push(ON0 + Math.round(Math.min(DUR, tE) * 10)); stimSet.add(i);
  }
}
const OFF = ON0 + Math.round(DUR * 10), STEPS = OFF + 1500;
say(`text "${TEXT}" band yc=${yc.toFixed(2)} um, ${DUR} ms scroll, ${STEPS} steps, ${iN.length} intervals over ${stimSet.size} neurons`);

/* ---------- leaky integrate-and-fire, identical to the site's clip maker ---------- */
const DLY = 18, DT = 0.1, V0 = -52, VTH = -45, K1 = DT / 20, K2 = DT / 5, KICK = 68.75, STIMBIT = 1 << 30, IDXMASK = STIMBIT - 1;
const v = new Float32Array(N).fill(V0), g = new Float32Array(N), ref = new Float32Array(N), buf = new Float32Array(DLY * N);
const bucket = steps => { const ptr = new Uint32Array(STEPS + 2); for (const t of steps) ptr[t + 1]++; for (let t = 0; t <= STEPS; t++) ptr[t + 1] += ptr[t]; const out = new Int32Array(steps.length), fill = ptr.slice(); for (let k = 0; k < steps.length; k++) out[fill[steps[k]]++] = iN[k]; return { ptr, out }; };
const starts = bucket(iS), ends = bucket(iE);
const drive = new Uint16Array(N), act = new Int32Array(N), actPos = new Int32Array(N); let nAct = 0;
const pIn = RATE * DT / 1000, fired = new Int32Array(N);
let spk = new Int32Array(1 << 22), nSpk = 0; const stepStart = new Uint32Array(STEPS + 1);
for (let s = 0; s < STEPS; s++) {
  const o = (s % DLY) * N;
  for (let q = starts.ptr[s]; q < starts.ptr[s + 1]; q++) { const n = starts.out[q]; if (drive[n]++ === 0) { actPos[n] = nAct; act[nAct++] = n; } }
  for (let q = ends.ptr[s]; q < ends.ptr[s + 1]; q++) { const n = ends.out[q]; if (--drive[n] === 0) { const pp = actPos[n], last = act[--nAct]; act[pp] = last; actPos[last] = pp; } }
  for (let k = 0; k < nAct; k++) if (Math.random() < pIn) v[act[k]] += KICK;
  stepStart[s] = nSpk;
  let nf = 0;
  for (let n = 0; n < N; n++) {
    let gn = g[n] + buf[o + n]; buf[o + n] = 0;
    const rn = ref[n];
    if (rn <= 0) {
      let vn = v[n]; vn += K1 * (V0 - vn + gn); gn -= K2 * gn;
      if (vn > VTH) { vn = V0; gn = 0; ref[n] = drive[n] ? 0 : 2.2; fired[nf++] = n; }
      v[n] = vn;
    } else ref[n] = rn - DT;
    g[n] = gn;
  }
  if (nSpk + nf > spk.length) { const a = new Int32Array(spk.length * 2); a.set(spk); spk = a; }
  for (let f = 0; f < nf; f++) {
    const n = fired[f]; spk[nSpk++] = drive[n] ? (n | STIMBIT) : n;
    for (let j = rowptr[n], e = rowptr[n + 1]; j < e; j++) buf[o + idx[j]] += W[j];
  }
  if (s % 5000 === 0) say(`step ${s}/${STEPS}, ${nSpk.toLocaleString()} spikes`);
}
stepStart[STEPS] = nSpk;
say(`done: ${nSpk.toLocaleString()} spikes`);

/* ---------- pack like 07_pack_hero.py ---------- */
const nf = Math.ceil(STEPS / BIN), frameCounts = new Uint16Array(nf), code = [];
for (let f = 0; f < nf; f++) {
  const keys = new Set();
  for (let s = f * BIN; s < Math.min(STEPS, (f + 1) * BIN); s++) for (let j = stepStart[s]; j < stepStart[s + 1]; j++) { const c = spk[j]; keys.add((c & IDXMASK) * 2 + (c & STIMBIT ? 1 : 0)); }
  const sorted = Int32Array.from(keys).sort();
  frameCounts[f] = sorted.length;
  let prev = 0;
  for (let k = 0; k < sorted.length; k++) {
    const nid = sorted[k] >> 1, flag = sorted[k] & 1, val = (k === 0 ? nid : nid - prev) * 2 + flag; prev = nid;
    if (val < 128) code.push(val);
    else if (val < 16384) code.push((val & 127) | 128, (val >> 7) & 127);
    else code.push((val & 127) | 128, ((val >> 7) & 127) | 128, (val >> 14) & 127);
  }
}
const KEYS = ['DNa01_L', 'DNa01_R', 'DNa02_L', 'DNa02_R', 'MDN_L', 'MDN_R', 'GF_L', 'GF_R', 'MN9', 'proboscis_MN', 'ingest_MN', 'antenna_MN_L', 'antenna_MN_R', 'neck_MN_L', 'neck_MN_R'];
const member = new Int8Array(N).fill(-1); KEYS.forEach((k, b) => { for (const i of G[k]) member[i] = b; });
const cc = KEYS.map(() => new Uint32Array(STEPS));
for (let s = 0; s < STEPS; s++) for (let j = stepStart[s]; j < stepStart[s + 1]; j++) { const b = member[spk[j] & IDXMASK]; if (b >= 0) cc[b][s]++; }
const cum = cc.map(a => { const c = new Float64Array(STEPS + 1); for (let s = 0; s < STEPS; s++) c[s + 1] = c[s] + a[s]; return c; });
const rates = {};
KEYS.forEach((k, b) => {
  const r = new Uint16Array(nf);
  for (let f = 0; f < nf; f++) { const e = Math.min(STEPS, (f + 1) * BIN), a = Math.max(0, e - 400); r[f] = Math.max(0, Math.min(65535, Math.round((cum[b][e] - cum[b][a]) / (G[k].length * (e - a) * 0.1 / 1000)))); }
  rates[k] = Buffer.from(r.buffer).toString('base64');
});
const gfc = new Uint16Array(nf);
for (let f = 0; f < nf; f++) { const e = Math.min(STEPS, (f + 1) * BIN); for (let s = f * BIN; s < e; s++) gfc[f] += cc[KEYS.indexOf('GF_L')][s] + cc[KEYS.indexOf('GF_R')][s]; }
const seenAll = new Uint8Array(N); let recruited = 0;
for (let s = ON0; s < STEPS; s++) for (let j = stepStart[s]; j < stepStart[s + 1]; j++) { const n = spk[j] & IDXMASK; if (!seenAll[n] && !stimSet.has(n)) { seenAll[n] = 1; recruited++; } }
const meta = { bin_ms: BIN / 10, frames: nf, on_ms: ON0 / 10, off_ms: OFF / 10, steps: STEPS, yc: +yc.toFixed(2), dur: DUR, text: TEXT, stim_neurons: stimSet.size, recruited };
fs.writeFileSync(OUT, JSON.stringify({ meta, counts: Buffer.from(frameCounts.buffer).toString('base64'), b64: Buffer.from(Uint8Array.from(code)).toString('base64'), rates, gf: Buffer.from(gfc.buffer).toString('base64') }));
say(`wrote ${OUT}: ${JSON.stringify(meta)}`);

/* ---------- fly mood for CONFIG.verdicts, computed exactly like the site's clip maker ---------- */
const endS = Math.min(STEPS - 1, OFF), seen = new Uint8Array(N); let lit = 0;
for (let s = ON0; s <= endS; s++) for (let j = stepStart[s]; j < stepStart[s + 1]; j++) { const n = spk[j] & IDXMASK; if (!seen[n]) { seen[n] = 1; lit++; } }
const avgRate = keys => { let tot = 0, cnt = 0; for (const k of keys) { const b = KEYS.indexOf(k); tot += cum[b][endS + 1] - cum[b][ON0]; cnt += G[k].length; } return tot / (cnt * (endS - ON0) * 0.1 / 1000); };
const daBit = new Uint8Array(N); for (const i of G.PAM) daBit[i] = 1; for (const i of G.PPL1) daBit[i] = 2;
let pa = 0, pp = 0;
for (let s = ON0; s <= endS; s++) for (let j = stepStart[s]; j < stepStart[s + 1]; j++) { const c = spk[j]; if (c & STIMBIT) continue; const d = daBit[c & IDXMASK]; if (d === 1) pa++; else if (d === 2) pp++; }
const daRate = (n, size) => n / (size * (endS - ON0) * 0.1 / 1000);
console.log(`\nCONFIG.verdicts entry for "${TEXT}":`);
console.log(`{ brand: '${TEXT}', neurons: ${lit}, appetite: ${avgRate(['MN9']).toFixed(1)}, curiosity: ${avgRate(['antenna_MN_L', 'antenna_MN_R']).toFixed(1)}, reward: ${daRate(pa, G.PAM.length).toFixed(1)}, punishment: ${daRate(pp, G.PPL1.length).toFixed(1)}, note: 'example run' }`);
