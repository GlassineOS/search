// SPDX-License-Identifier: GPL-2.0-or-later
// Scoring, a port of glassine_docs/{store,quant,staticmodel,bm25,fusion}.py:
// the file container, int8/sign-bit quantization, the static embedder, BM25
// and reciprocal rank fusion. Needs text.js first. Attaches to globalThis.GDS.
(function (G) {
  "use strict";
  const GDS = (G.GDS = G.GDS || {});

  // --- container (store.py) -------------------------------------------------
  const MAGIC = "GLDSRCH1";
  const TYPES = { i8: Int8Array, u8: Uint8Array, f32: Float32Array };

  function unpack(buf) {
    const u8 = new Uint8Array(buf);
    const magic = String.fromCharCode(...u8.subarray(0, 8));
    if (magic !== MAGIC) throw new Error("not a glassine search file");
    const hlen = new DataView(buf).getUint32(8, true);
    const header = JSON.parse(new TextDecoder().decode(u8.subarray(12, 12 + hlen)));
    const arrays = {};
    for (const [name, a] of Object.entries(header.arrays)) {
      const count = a.shape.reduce((x, y) => x * y, 1);
      arrays[name] = { data: new TYPES[a.dtype](buf, a.offset, count), shape: a.shape };
    }
    return { meta: header.meta, arrays };
  }

  // --- quantization (quant.py) ----------------------------------------------
  const roundHalfAway = (x) => Math.sign(x) * Math.floor(Math.abs(x) + 0.5);

  function quantizeVec(v) {
    let m = 0;
    for (let i = 0; i < v.length; i++) m = Math.max(m, Math.abs(v[i]));
    const scale = Math.fround(m / 127);
    const q = new Int8Array(v.length);
    if (scale > 0) {
      for (let i = 0; i < v.length; i++) {
        q[i] = Math.max(-127, Math.min(127, roundHalfAway(v[i] / scale)));
      }
    }
    return { q, scale };
  }

  function signBits(v) {
    const out = new Uint8Array(Math.ceil(v.length / 8));
    for (let i = 0; i < v.length; i++) if (v[i] > 0) out[i >> 3] |= 0x80 >> (i & 7);
    return out;
  }

  const POP = new Uint8Array(256);
  for (let i = 0; i < 256; i++) POP[i] = (i & 1) + POP[i >> 1];

  const byScoreThenIndex = (a, b) => (b[1] - a[1]) || (a[0] - b[0]);

  // Binary pre-selection by Hamming distance, then int8 rescoring of the best
  // `candidates`. bits: Uint8Array n*nb, q8: Int8Array n*dims, scale: Float32Array n.
  function staticScores(qvec, bits, q8, scale, candidates) {
    const n = scale.length, dims = qvec.length, nb = dims >> 3;
    const qb = signBits(qvec);
    const ham = new Array(n);
    for (let d = 0; d < n; d++) {
      let h = 0;
      const o = d * nb;
      for (let j = 0; j < nb; j++) h += POP[bits[o + j] ^ qb[j]];
      ham[d] = [d, h];
    }
    ham.sort((a, b) => (a[1] - b[1]) || (a[0] - b[0]));
    const { q, scale: qs } = quantizeVec(qvec);
    const out = [];
    for (const [d] of ham.slice(0, candidates)) {
      let dot = 0;
      const o = d * dims;
      for (let j = 0; j < dims; j++) dot += q8[o + j] * q[j];
      out.push([d, (dot * qs) * scale[d]]);
    }
    return out.sort(byScoreThenIndex);
  }

  // --- static embedder (staticmodel.py) -------------------------------------
  class StaticModel {
    constructor(buf) {
      const { meta, arrays } = unpack(buf);
      if (meta.kind !== "static-model") throw new Error("not a static model file");
      this.meta = meta;
      this.tokenizer = new GDS.WordPiece(meta.tokenizer);
      this.table = arrays.table.data;
      this.scale = arrays.scale.data;
      this.dims = meta.dims;
      this.normalize = meta.normalize;
      this.maxLength = meta.max_length;
      this.median = meta.median_token_length;
    }

    ids(text) {
      if (this.maxLength !== null) {
        const cs = Array.from(text);
        const m = this.maxLength * this.median;
        if (cs.length > m) text = cs.slice(0, m).join("");
      }
      const unk = this.tokenizer.unkId;
      return this.tokenizer.encode(text, false, this.maxLength).filter((i) => i !== unk);
    }

    embed(text) {
      const ids = this.ids(text);
      const d = this.dims;
      const v = new Float64Array(d);
      if (!ids.length) return v;
      for (const id of ids) {
        const s = this.scale[id], o = id * d;
        for (let j = 0; j < d; j++) v[j] += this.table[o + j] * s;
      }
      for (let j = 0; j < d; j++) v[j] /= ids.length;
      if (this.normalize) {
        let ss = 0;
        for (let j = 0; j < d; j++) ss += v[j] * v[j];
        const norm = Math.sqrt(ss) + 1e-32;
        for (let j = 0; j < d; j++) v[j] /= norm;
      }
      return v;
    }
  }

  // --- BM25 (bm25.py) -------------------------------------------------------
  function bm25Scores(index, query) {
    if (!index._stop) index._stop = new Set(index.stopwords);
    const { n, k1, b, avgdl, doclen, postings } = index;
    const acc = new Map();
    const terms = [...new Set(GDS.bm25Terms(query, index._stop))].sort((x, y) => (x < y ? -1 : x > y ? 1 : 0));
    for (const term of terms) {
      const post = Object.prototype.hasOwnProperty.call(postings, term) ? postings[term] : null;
      if (!post || !post.length) continue;
      const df = post.length / 2;
      const idf = Math.log(1 + (n - df + 0.5) / (df + 0.5));
      for (let i = 0; i < post.length; i += 2) {
        const d = post[i], tf = post[i + 1];
        const norm = tf * (k1 + 1) / (tf + k1 * (1 - b + b * doclen[d] / avgdl));
        acc.set(d, (acc.get(d) || 0) + idf * norm);
      }
    }
    return [...acc.entries()].sort(byScoreThenIndex);
  }

  // --- fusion (fusion.py) ---------------------------------------------------
  function rrf(lists, k = 60, depth = 100) {
    const acc = new Map();
    for (const lst of lists) {
      const m = Math.min(depth, lst.length);
      for (let r = 0; r < m; r++) acc.set(lst[r], (acc.get(lst[r]) || 0) + 1 / (k + r + 1));
    }
    return [...acc.entries()].sort(byScoreThenIndex);
  }

  Object.assign(GDS, { unpack, roundHalfAway, quantizeVec, signBits, staticScores, StaticModel,
    bm25Scores, rrf });
})(globalThis);
