// SPDX-License-Identifier: GPL-2.0-or-later
// Search worker: loads the static profile (model table, vectors), the chunks
// and the BM25 index, then answers queries with fused keyword + meaning ranks.
//   in:  {type: "init", base, manifest}       base = URL of the search/ directory
//        {type: "query", seq, q, k}
//   out: {type: "ready", ms} | {type: "error", message}
//        {type: "results", seq, q, ms, results: [{chunk fields, score, bm25, vector}]}
"use strict";

let state = null;
const pending = [];

async function load(base, manifest) {
  const t0 = performance.now();
  importScripts(new URL(manifest.web["text.js"], base).href, new URL(manifest.web["score.js"], base).href);
  const prof = manifest.profiles.static;
  const get = async (name, kind) => {
    const r = await fetch(new URL(name, base));
    if (!r.ok) throw new Error(`${name}: HTTP ${r.status}`);
    return kind === "json" ? r.json() : r.arrayBuffer();
  };
  const [model, vecs, chunks, bm25] = await Promise.all([
    get(prof.model.file), get(prof.vectors.file), get(manifest.chunks.file, "json"), get(manifest.bm25.file, "json"),
  ]);
  const v = GDS.unpack(vecs);
  state = {
    model: new GDS.StaticModel(model),
    bits: v.arrays.bits.data, q8: v.arrays.q8.data, scale: v.arrays.scale.data,
    candidates: prof.quantization.candidates || 100,
    chunks: chunks.chunks, bm25, fusion: manifest.fusion,
  };
  return performance.now() - t0;
}

function search(q, k) {
  const t0 = performance.now();
  const s = state;
  const kw = GDS.bm25Scores(s.bm25, q);
  const vec = GDS.staticScores(s.model.embed(q), s.bits, s.q8, s.scale, s.candidates);
  // A query with no known tokens has no meaning to search by: keywords only.
  const lists = [kw.map((p) => p[0])];
  const hasMeaning = s.model.ids(q).length > 0;
  if (hasMeaning) lists.push(vec.map((p) => p[0]));
  const fused = GDS.rrf(lists, s.fusion.k, s.fusion.depth);
  const kwRank = new Map(kw.map((p, r) => [p[0], r + 1]));
  const vecRank = new Map(vec.map((p, r) => [p[0], r + 1]));
  const results = fused.slice(0, k).map(([i, score]) => {
    const c = s.chunks[i];
    return { id: c.id, title: c.title, heading: c.heading, path: c.path, url: c.url, text: c.text,
      score, bm25: kwRank.get(i) || null, vector: hasMeaning ? (vecRank.get(i) || null) : null };
  });
  return { results, ms: performance.now() - t0 };
}

function answer(msg) {
  const { results, ms } = search(msg.q, msg.k || 20);
  postMessage({ type: "results", seq: msg.seq, q: msg.q, ms, results });
}

onmessage = async (e) => {
  const msg = e.data;
  if (msg.type === "init") {
    try {
      const ms = await load(msg.base, msg.manifest);
      postMessage({ type: "ready", ms });
      // answer only the newest query that came in while loading
      if (pending.length) answer(pending[pending.length - 1]);
      pending.length = 0;
    } catch (err) {
      postMessage({ type: "error", message: String(err && err.message || err) });
    }
  } else if (msg.type === "query") {
    if (state) answer(msg); else pending.push(msg);
  }
};
