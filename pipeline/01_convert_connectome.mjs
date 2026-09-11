// Converts the Shiu et al. connectivity parquet into raw arrays (build/pre.i32, post.i32, w.f32).
// Needs: npm install hyparquet   (run from the pipeline/ folder)
import { parquetMetadataAsync, parquetReadObjects, asyncBufferFromFile } from 'hyparquet'
import fs from 'fs'
import zlib from 'zlib'
const compressors = { BROTLI: (input) => new Uint8Array(zlib.brotliDecompressSync(input)) }
fs.mkdirSync('build', { recursive: true })
const file = await asyncBufferFromFile('raw/Drosophila_brain_model/Connectivity_783.parquet')
const md = await parquetMetadataAsync(file)
const cols = ['Presynaptic_Index', 'Postsynaptic_Index', 'Excitatory x Connectivity']
const n = Number(md.num_rows)
const pre = new Int32Array(n), post = new Int32Array(n), w = new Float32Array(n)
let off = 0
for (const rg of md.row_groups) {
  const rgRows = Number(rg.num_rows)
  const rows = await parquetReadObjects({ file, metadata: md, compressors, columns: cols, rowStart: off, rowEnd: off + rgRows })
  for (let i = 0; i < rows.length; i++) { const r = rows[i]; pre[off + i] = Number(r[cols[0]]); post[off + i] = Number(r[cols[1]]); w[off + i] = Number(r[cols[2]]) }
  off += rgRows
  console.log('read', off, 'of', n)
}
fs.writeFileSync('build/pre.i32', Buffer.from(pre.buffer))
fs.writeFileSync('build/post.i32', Buffer.from(post.buffer))
fs.writeFileSync('build/w.f32', Buffer.from(w.buffer))
