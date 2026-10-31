// SPDX-License-Identifier: GPL-2.0-or-later
// Text processing, a line-by-line port of glassine_docs/text.py: the BERT
// WordPiece tokenizer and the BM25 term splitter. Plain script; attaches to
// globalThis.GDS so a web worker (importScripts) and node (require) load it alike.
(function (G) {
  "use strict";
  const GDS = (G.GDS = G.GDS || {});

  const CJK = [
    [0x4e00, 0x9fff], [0x3400, 0x4dbf], [0x20000, 0x2a6df], [0x2a700, 0x2b73f],
    [0x2b740, 0x2b81f], [0x2b820, 0x2ceaf], [0xf900, 0xfaff], [0x2f800, 0x2fa1f],
  ];
  const WS = new Set([0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x20, 0x85, 0xa0, 0x1680, 0x2028, 0x2029,
    0x202f, 0x205f, 0x3000]);
  for (let c = 0x2000; c < 0x200b; c++) WS.add(c);
  const ASCII_PUNCT = new Set('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~');
  const RE_C = /^\p{C}$/u, RE_MN = /^\p{Mn}$/u, RE_P = /^\p{P}$/u, RE_LN = /^[\p{L}\p{N}]$/u;

  const isWs = (c) => WS.has(c.codePointAt(0));
  const isControl = (c) => !(c === "\t" || c === "\n" || c === "\r") && RE_C.test(c);
  const isCjk = (cp) => CJK.some(([lo, hi]) => lo <= cp && cp <= hi);
  const isPunct = (c) => ASCII_PUNCT.has(c) || RE_P.test(c);

  // char by char, as tokenizers does: no context rules (a final sigma stays σ)
  function lower(text) {
    let out = "";
    for (const c of text) out += c.toLowerCase();
    return out;
  }

  function stripMn(text) {
    let out = "";
    for (const c of text.normalize("NFD")) if (!RE_MN.test(c)) out += c;
    return out;
  }

  class WordPiece {
    constructor(config) {
      this.vocabList = config.vocab;
      this.vocab = new Map();
      config.vocab.forEach((t, i) => this.vocab.set(t, i));
      this.unk = config.unk_token;
      this.unkId = this.vocab.get(this.unk);
      this.prefix = config.continuing_subword_prefix ?? "##";
      this.maxChars = config.max_input_chars_per_word ?? 100;
      this.lowercase = config.lowercase ?? true;
      const sa = config.strip_accents;
      this.stripAccents = sa === null || sa === undefined ? this.lowercase : !!sa;
      this.cleanText = config.clean_text ?? true;
      this.handleCjk = config.handle_chinese_chars ?? true;
      this.added = [...(config.added_tokens || [])].sort((a, b) => [...b].length - [...a].length);
      this.clsId = this.vocab.get(config.cls_token ?? "[CLS]");
      this.sepId = this.vocab.get(config.sep_token ?? "[SEP]");
    }

    normalize(text) {
      if (this.cleanText) {
        let out = "";
        for (const c of text) {
          const cp = c.codePointAt(0);
          if (cp === 0 || cp === 0xfffd || isControl(c)) continue;
          out += isWs(c) ? " " : c;
        }
        text = out;
      }
      if (this.handleCjk) {
        let out = "";
        for (const c of text) out += isCjk(c.codePointAt(0)) ? " " + c + " " : c;
        text = out;
      }
      if (this.stripAccents) text = stripMn(text);
      if (this.lowercase) text = lower(text);
      return text;
    }

    static preTokenize(text) {
      const words = [];
      let cur = "";
      for (const c of text) {
        if (isWs(c)) {
          if (cur) { words.push(cur); cur = ""; }
        } else if (isPunct(c)) {
          if (cur) { words.push(cur); cur = ""; }
          words.push(c);
        } else {
          cur += c;
        }
      }
      if (cur) words.push(cur);
      return words;
    }

    wordIds(word) {
      const cs = Array.from(word);
      if (cs.length > this.maxChars) return [this.unkId];
      const ids = [];
      let start = 0;
      const n = cs.length;
      while (start < n) {
        let end = n, found;
        while (start < end) {
          let piece = cs.slice(start, end).join("");
          if (start > 0) piece = this.prefix + piece;
          const tid = this.vocab.get(piece);
          if (tid !== undefined) { found = tid; break; }
          end -= 1;
        }
        if (found === undefined) return [this.unkId];
        ids.push(found);
        start = end;
      }
      return ids;
    }

    *splitAdded(text) {
      if (!this.added.length) { yield [text, false]; return; }
      let i = 0, start = 0;
      const n = text.length;  // UTF-16 offsets are fine here: tokens start with "["
      while (i < n) {
        let hit = null;
        if (text[i] === "[") {
          for (const a of this.added) if (text.startsWith(a, i)) { hit = a; break; }
        }
        if (hit === null) { i += 1; continue; }
        if (start < i) yield [text.slice(start, i), false];
        yield [hit, true];
        i += hit.length;
        start = i;
      }
      if (start < n) yield [text.slice(start), false];
    }

    encode(text, addSpecialTokens = false, maxLength = null) {
      let ids = [];
      for (const [part, isAdded] of this.splitAdded(text)) {
        if (isAdded) { ids.push(this.vocab.get(part)); continue; }
        for (const w of WordPiece.preTokenize(this.normalize(part))) {
          for (const id of this.wordIds(w)) ids.push(id);
        }
      }
      if (addSpecialTokens) {
        if (maxLength !== null) ids = ids.slice(0, maxLength - 2);
        return [this.clsId, ...ids, this.sepId];
      }
      if (maxLength !== null) ids = ids.slice(0, maxLength);
      return ids;
    }
  }

  // --- BM25 terms -----------------------------------------------------------
  const JOINERS = new Set("_-.:/");
  const fold = (text) => lower(stripMn(text));
  const isWordChar = (c) => RE_LN.test(c);

  function bm25Terms(text, stopwords) {
    const stop = stopwords instanceof Set ? stopwords : new Set(stopwords || []);
    const cs = Array.from(fold(text));
    const out = [];
    const n = cs.length;
    let i = 0;
    while (i < n) {
      if (!isWordChar(cs[i])) { i += 1; continue; }
      let j = i, cur = i;
      const parts = [];
      while (j < n) {
        const c = cs[j];
        if (isWordChar(c)) j += 1;
        else if (JOINERS.has(c) && j + 1 < n && isWordChar(cs[j + 1]) && j > cur) {
          parts.push(cs.slice(cur, j).join(""));
          j += 1;
          cur = j;
        } else break;
      }
      parts.push(cs.slice(cur, j).join(""));
      if (parts.length > 1) out.push(cs.slice(i, j).join(""));
      for (const p of parts) if (!stop.has(p)) out.push(p);
      i = j;
    }
    return out;
  }

  GDS.WordPiece = WordPiece;
  GDS.bm25Terms = bm25Terms;
})(globalThis);
