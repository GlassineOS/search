// SPDX-License-Identifier: GPL-2.0-or-later
// The browser's scoring against the Python side's outputs (tests/make_parity.py).
// GDS_PARITY_DIR names the directory make_parity.py wrote.
"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");

require("../../glassine_docs/web/text.js");
require("../../glassine_docs/web/score.js");
const GDS = globalThis.GDS;
const dir = process.env.GDS_PARITY_DIR;

test("parity with the Python implementation", { skip: !dir && "GDS_PARITY_DIR not set" }, async (t) => {
  const exp = JSON.parse(fs.readFileSync(path.join(dir, "expected.json"), "utf8"));
  const site = exp.site;
  const m = JSON.parse(fs.readFileSync(path.join(site, "manifest.json"), "utf8"));
  const buf = (f) => { const b = fs.readFileSync(path.join(site, f)); return b.buffer.slice(b.byteOffset, b.byteOffset + b.length); };
  const prof = m.profiles.static;
  const model = new GDS.StaticModel(buf(prof.model.file));
  const v = GDS.unpack(buf(prof.vectors.file));
  const index = JSON.parse(fs.readFileSync(path.join(site, m.bm25.file), "utf8"));
  const bits = v.arrays.bits.data, q8 = v.arrays.q8.data, scale = v.arrays.scale.data;
  const pairs = (a) => a.map(([d, s]) => [d, s]);

  for (const row of exp.rows) {
    await t.test(JSON.stringify(row.q).slice(0, 50), () => {
      assert.deepEqual(GDS.bm25Terms(row.q, index.stopwords), row.terms, "bm25 terms");
      assert.deepEqual(model.ids(row.q), row.ids, "token ids");
      const vec = model.embed(row.q);
      assert.equal(Buffer.from(new Float64Array(vec).buffer).toString("hex"), row.vec_hex, "query vector bits");
      const kw = GDS.bm25Scores(index, row.q);
      assert.deepEqual(kw.map((p) => p[0]), row.bm25.map((p) => p[0]), "bm25 order");
      kw.forEach((p, i) => assert.ok(Math.abs(p[1] - row.bm25[i][1]) <= 1e-12 * Math.max(1, p[1]), "bm25 score"));
      const st = GDS.staticScores(vec, bits, q8, scale, 100);
      assert.deepEqual(pairs(st), row.static, "static scores");
      const lists = [kw.map((p) => p[0])];
      if (model.ids(row.q).length) lists.push(st.map((p) => p[0]));
      assert.deepEqual(GDS.rrf(lists), row.fused, "fused");
    });
  }
});

test("quantization helpers", () => {
  assert.deepEqual([0.5, -0.5, 1.5, -2.5, 0.49].map(GDS.roundHalfAway), [1, -1, 2, -3, 0]);
  assert.deepEqual(Array.from(GDS.signBits([1, -1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])), [0b10010000, 0b10000000]);
  assert.deepEqual(GDS.rrf([[2, 0, 1], [0, 3]]).map((p) => p[0]), [0, 2, 3, 1]);
});
