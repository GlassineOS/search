// SPDX-License-Identifier: GPL-2.0-or-later
// The JS tokenizer against Hugging Face tokenizers' ids (tests/fixtures/golden-tokens.json).
"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");

require("../../glassine_docs/web/text.js");
const { WordPiece } = globalThis.GDS;
const FIX = path.join(__dirname, "..", "fixtures");
const read = (f) => JSON.parse(fs.readFileSync(path.join(FIX, f), "utf8"));
const TEXTS = read("token-texts.json");
const GOLDEN = read("golden-tokens.json");

for (const profile of ["static", "transformer"]) {
  test(`WordPiece ${profile} matches tokenizers`, () => {
    const wp = new WordPiece(read(`${profile}-tokenizer.json`));
    TEXTS.forEach((t, i) => assert.deepEqual(wp.encode(t), GOLDEN[profile].plain[i], JSON.stringify(t)));
  });
}

test("WordPiece transformer with special tokens", () => {
  const wp = new WordPiece(read("transformer-tokenizer.json"));
  TEXTS.forEach((t, i) => assert.deepEqual(wp.encode(t, true), GOLDEN.transformer.special[i], JSON.stringify(t)));
});
