// SPDX-License-Identifier: GPL-2.0-or-later
// Material for MkDocs search worker with semantic ranking.
//
// The plugin points Material's config ("search" in #__config) at this file:
//   glassine-search.<hash>.js?material=<Material's worker file>&search=<URL of search/>
// It lives next to Material's own worker, runs it unchanged for the keyword side
// (lunr, highlighting, suggestions) and speaks Material's protocol to the page:
//   in   {type: 0 SETUP, data: {config, docs, options}}   {type: 2 QUERY, data: string}
//   out  {type: 1 READY}                                    {type: 3 RESULT, data: {items: SearchItem[][], suggest?}}
// Material's page code renders every RESULT it receives (it does not match results to
// queries), so a RESULT is only ever sent for the newest query.
//
// The first non-empty QUERY starts loading the static model, the vectors and the chunk list.
// Until they are there, Material's keyword results go out as they are. From then on,
// each query's keyword list and vector list are fused (reciprocal rank fusion) into
// one list in Material's result format.
"use strict";

const params = new URL(self.location.href).searchParams;
const SEARCH_BASE = new URL(params.get("search") || "../../../search/", self.location.href);
const VECTOR_DEPTH = 10;   // vector hits fused into the keyword list
const SCORE_SCALE = 1000;  // RRF scores are ~0.03; Material prints scores with 2 decimals

// --- run Material's worker, intercepting its message listener and its replies ---
const fromPage = [];       // Material's "message" listeners
const realAdd = self.addEventListener.bind(self);
const realPost = self.postMessage.bind(self);
self.addEventListener = (type, fn, opts) => (type === "message" ? fromPage.push(fn) : realAdd(type, fn, opts));
self.postMessage = (msg) => onMaterial(msg);
importScripts(new URL(params.get("material"), self.location.href).href);
self.addEventListener = realAdd;

const SETUP = 0, READY = 1, QUERY = 2, RESULT = 3;
let docs = null;            // Material's documents (Material links sections to their page)
let byLocation = null;
const pending = [];         // queries handed to Material, in order (it answers in order)
let latest = null;          // newest query from the page
let keyword = null;         // {q, data}: Material's answer to the newest query
let sem = null;             // loaded semantic state
let loading = null;

function toMaterial(msg) {
  for (const fn of fromPage) fn({ data: msg });
}

realAdd("message", (ev) => {
  const msg = ev.data;
  if (msg && msg.type === SETUP) docs = msg.data.docs;
  if (msg && msg.type === QUERY) {
    latest = msg.data;
    pending.push(msg.data);
    // Material sends the (empty) box value once at start-up: wait for real text
    if (!loading && String(msg.data).trim()) loading = load().catch((err) => { console.warn("semantic search:", err); });
  }
  toMaterial(msg);
});

function onMaterial(msg) {
  if (!msg || msg.type !== RESULT) { realPost(msg); return; }
  const q = pending.shift();
  if (q !== latest) return;           // an answer to an outdated query
  keyword = { q, data: msg.data };
  realPost({ type: RESULT, data: sem ? fuse(q, msg.data) : msg.data });
}

async function load() {
  const get = async (name, kind) => {
    const r = await fetch(new URL(name, SEARCH_BASE), kind === "manifest" ? { cache: "no-cache" } : {});
    if (!r.ok) throw new Error(`${name}: HTTP ${r.status}`);
    return kind === "buf" ? r.arrayBuffer() : r.json();
  };
  const manifest = await get("manifest.json", "manifest");
  importScripts(new URL(manifest.web["text.js"], SEARCH_BASE).href, new URL(manifest.web["score.js"], SEARCH_BASE).href);
  const prof = manifest.profiles.static;
  const [model, vecs, chunks] = await Promise.all([
    get(prof.model.file, "buf"), get(prof.vectors.file, "buf"), get(manifest.chunks.file),
  ]);
  const v = GDS.unpack(vecs);
  sem = {
    model: new GDS.StaticModel(model),
    bits: v.arrays.bits.data, q8: v.arrays.q8.data, scale: v.arrays.scale.data,
    candidates: prof.quantization.candidates || 100,
    urls: chunks.chunks.map((c) => c.url),
    k: manifest.fusion.k,
  };
  // the newest query already has keyword results on screen: send the fused list
  if (keyword && keyword.q === latest) realPost({ type: RESULT, data: fuse(keyword.q, keyword.data) });
}

function docIndex() {
  if (!byLocation) {
    byLocation = new Map();
    (docs || []).forEach((d, i) => byLocation.set(d.location, i));
  }
  return byLocation;
}

// Material's result + vector ranking -> one result in Material's format.
function fuse(q, data) {
  if (!docs || !sem.model.ids(q).length) return data;  // no known word: keywords only
  const index = docIndex();
  // keyword ranking: Material's items by score (skip the score-0 page fillers)
  const items = new Map();
  for (const group of data.items) for (const it of group) {
    if (it.score > 0 || Object.keys(it.terms).length) items.set(it.location, it);
  }
  const kw = [...items.values()].sort((a, b) => b.score - a.score)
    .map((it) => index.get(it.location)).filter((i) => i !== undefined);
  // vector ranking: chunk -> its section's document, first hit counts
  const vec = [];
  const seen = new Set();
  for (const [c] of GDS.staticScores(sem.model.embed(q), sem.bits, sem.q8, sem.scale, sem.candidates)) {
    const i = index.get(sem.urls[c]);
    if (i === undefined || seen.has(i)) continue;
    seen.add(i);
    vec.push(i);
    if (vec.length >= VECTOR_DEPTH) break;
  }
  const fused = GDS.rrf([kw, vec], sem.k, 100);
  // group by page, in fused order; every group carries its page, as Material's do
  const groups = new Map();
  for (const [i, score] of fused) {
    const doc = docs[i];
    const it = { ...(items.get(doc.location) || { ...doc, terms: {} }), score: score * SCORE_SCALE };
    const page = doc.parent ? doc.parent.location : doc.location;
    if (!groups.has(page)) groups.set(page, []);
    groups.get(page).push(it);
  }
  for (const [page, list] of groups) {
    if (!list.some((it) => it.location === page)) {
      const i = index.get(page);
      if (i !== undefined) list.push({ ...docs[i], score: 0, terms: {} });
    }
  }
  const out = { items: [...groups.values()] };
  if (data.suggest !== undefined) out.suggest = data.suggest;
  return out;
}
