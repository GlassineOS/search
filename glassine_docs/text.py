# SPDX-License-Identifier: GPL-2.0-or-later
"""Text processing shared by the builder and the CLI.

Two tokenizers, each mirrored line by line in ``web/text.js``:

- ``WordPiece``: the BERT tokenizer (BertNormalizer, BertPreTokenizer, greedy
  longest-match WordPiece) as Hugging Face ``tokenizers`` runs it for the
  ``tokenizer.json`` of both models. The golden tests compare its ids with
  ``tokenizers``.
- ``bm25_terms``: the keyword tokenizer of the BM25 index.
"""
from __future__ import annotations

import json
import unicodedata

# CJK Unified Ideographs blocks, as in BERT's _is_chinese_char.
_CJK = (
    (0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0x20000, 0x2A6DF), (0x2A700, 0x2B73F),
    (0x2B740, 0x2B81F), (0x2B820, 0x2CEAF), (0xF900, 0xFAFF), (0x2F800, 0x2FA1F),
)
# Unicode White_Space (what Rust's char::is_whitespace tests).
_WS = frozenset(
    [0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x20, 0x85, 0xA0, 0x1680, 0x2028, 0x2029, 0x202F, 0x205F, 0x3000]
    + list(range(0x2000, 0x200B))
)
_ASCII_PUNCT = frozenset('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')


def _is_ws(c: str) -> bool:
    return ord(c) in _WS


def _is_control(c: str) -> bool:
    if c in "\t\n\r":
        return False
    return unicodedata.category(c)[0] == "C"


def _is_cjk(cp: int) -> bool:
    return any(lo <= cp <= hi for lo, hi in _CJK)


def _lower(text: str) -> str:
    # char by char, as tokenizers does: no context rules (a final sigma stays σ)
    return "".join(c.lower() for c in text)


def _is_punct(c: str) -> bool:
    return c in _ASCII_PUNCT or unicodedata.category(c)[0] == "P"


class WordPiece:
    """BERT tokenizer. ``config`` is the dict that ``to_config`` returns."""

    def __init__(self, config: dict):
        self.vocab_list = list(config["vocab"])
        self.vocab = {t: i for i, t in enumerate(self.vocab_list)}
        self.unk = config["unk_token"]
        self.unk_id = self.vocab[self.unk]
        self.prefix = config.get("continuing_subword_prefix", "##")
        self.max_chars = int(config.get("max_input_chars_per_word", 100))
        self.lowercase = bool(config.get("lowercase", True))
        sa = config.get("strip_accents")
        self.strip_accents = self.lowercase if sa is None else bool(sa)
        self.clean_text = bool(config.get("clean_text", True))
        self.handle_cjk = bool(config.get("handle_chinese_chars", True))
        # Added tokens are matched on the raw text, longest first.
        self.added = sorted(config.get("added_tokens", []), key=lambda a: -len(a))
        self.cls_id = self.vocab.get(config.get("cls_token", "[CLS]"))
        self.sep_id = self.vocab.get(config.get("sep_token", "[SEP]"))

    @classmethod
    def from_hf_json(cls, path) -> "WordPiece":
        return cls(hf_tokenizer_config(path))

    # --- normalizer -------------------------------------------------------
    def normalize(self, text: str) -> str:
        if self.clean_text:
            out = []
            for c in text:
                cp = ord(c)
                if cp == 0 or cp == 0xFFFD or _is_control(c):
                    continue
                out.append(" " if _is_ws(c) else c)
            text = "".join(out)
        if self.handle_cjk:
            text = "".join(f" {c} " if _is_cjk(ord(c)) else c for c in text)
        if self.strip_accents:
            text = "".join(c for c in unicodedata.normalize("NFD", text)
                           if unicodedata.category(c) != "Mn")
        if self.lowercase:
            text = _lower(text)
        return text

    # --- pre-tokenizer ----------------------------------------------------
    @staticmethod
    def pre_tokenize(text: str) -> list[str]:
        words, cur = [], []
        for c in text:
            if _is_ws(c):
                if cur:
                    words.append("".join(cur))
                    cur = []
            elif _is_punct(c):
                if cur:
                    words.append("".join(cur))
                    cur = []
                words.append(c)
            else:
                cur.append(c)
        if cur:
            words.append("".join(cur))
        return words

    # --- model ------------------------------------------------------------
    def word_ids(self, word: str) -> list[int]:
        if len(word) > self.max_chars:
            return [self.unk_id]
        ids, start, n = [], 0, len(word)
        while start < n:
            end, found = n, None
            while start < end:
                piece = word[start:end]
                if start > 0:
                    piece = self.prefix + piece
                tid = self.vocab.get(piece)
                if tid is not None:
                    found = tid
                    break
                end -= 1
            if found is None:
                return [self.unk_id]
            ids.append(found)
            start = end
        return ids

    def encode(self, text: str, add_special_tokens: bool = False, max_length: int | None = None) -> list[int]:
        ids: list[int] = []
        for part, is_added in self._split_added(text):
            if is_added:
                ids.append(self.vocab[part])
                continue
            for w in self.pre_tokenize(self.normalize(part)):
                ids.extend(self.word_ids(w))
        if add_special_tokens:
            if max_length is not None:
                ids = ids[: max_length - 2]
            return [self.cls_id] + ids + [self.sep_id]
        if max_length is not None:
            ids = ids[:max_length]
        return ids

    def _split_added(self, text: str):
        if not self.added:
            yield text, False
            return
        i, start, n = 0, 0, len(text)
        while i < n:
            hit = None
            if text[i] == "[":
                for a in self.added:
                    if text.startswith(a, i):
                        hit = a
                        break
            if hit is None:
                i += 1
                continue
            if start < i:
                yield text[start:i], False
            yield hit, True
            i += len(hit)
            start = i
        if start < n:
            yield text[start:], False

    def to_config(self) -> dict:
        return {
            "vocab": self.vocab_list, "unk_token": self.unk,
            "continuing_subword_prefix": self.prefix, "max_input_chars_per_word": self.max_chars,
            "lowercase": self.lowercase, "strip_accents": self.strip_accents,
            "clean_text": self.clean_text, "handle_chinese_chars": self.handle_cjk,
            "added_tokens": [a for a in self.added], "cls_token": self.vocab_list[self.cls_id],
            "sep_token": self.vocab_list[self.sep_id],
        }


def hf_tokenizer_config(path) -> dict:
    """Read a Hugging Face ``tokenizer.json`` of the BERT kind into our config."""
    with open(path, encoding="utf-8") as f:
        t = json.load(f)
    norm, pre, model = t["normalizer"], t["pre_tokenizer"], t["model"]
    if norm.get("type") != "BertNormalizer" or pre.get("type") != "BertPreTokenizer" or model.get("type") != "WordPiece":
        raise ValueError(f"{path}: not a BERT WordPiece tokenizer")
    for a in t.get("added_tokens", []):
        if a.get("normalized") or a.get("lstrip") or a.get("rstrip") or a.get("single_word"):
            raise ValueError(f"{path}: added token {a['content']!r} has flags this tokenizer does not implement")
        if not a["content"].startswith("["):
            raise ValueError(f"{path}: added token {a['content']!r} does not start with '['")
    vocab = [None] * len(model["vocab"])
    for tok, i in model["vocab"].items():
        vocab[i] = tok
    if any(v is None for v in vocab):
        raise ValueError(f"{path}: vocabulary ids are not contiguous")
    return {
        "vocab": vocab, "unk_token": model["unk_token"],
        "continuing_subword_prefix": model.get("continuing_subword_prefix", "##"),
        "max_input_chars_per_word": model.get("max_input_chars_per_word", 100),
        "lowercase": norm.get("lowercase", True), "strip_accents": norm.get("strip_accents"),
        "clean_text": norm.get("clean_text", True), "handle_chinese_chars": norm.get("handle_chinese_chars", True),
        "added_tokens": [a["content"] for a in t.get("added_tokens", [])],
        "cls_token": "[CLS]", "sep_token": "[SEP]",
    }


# --- BM25 terms -------------------------------------------------------------
STOPWORDS = sorted(set(
    "a an and are as at be but by can do does for from has have how i if in into is it its "
    "me my no not of on or our so than that the their then there these this to too was we "
    "what when where which who why will with you your".split()))
_JOINERS = "_-.:/"


def _fold(text: str) -> str:
    text = unicodedata.normalize("NFD", text)
    return _lower("".join(c for c in text if unicodedata.category(c) != "Mn"))


def _is_word_char(c: str) -> bool:
    return unicodedata.category(c)[0] in "LN"


def bm25_terms(text: str, stopwords=STOPWORDS) -> list[str]:
    """Keyword terms: runs of letters/digits joined by ``_ - . : /``.

    A joined run is emitted whole and as its parts: ``boot_b`` gives
    ``boot_b``, ``boot``, ``b``. Stop words are dropped only as single parts.
    """
    stop = stopwords if isinstance(stopwords, (set, frozenset)) else set(stopwords)
    text = _fold(text)
    out: list[str] = []
    n, i = len(text), 0
    while i < n:
        if not _is_word_char(text[i]):
            i += 1
            continue
        j = i
        parts, cur = [], i
        while j < n:
            c = text[j]
            if _is_word_char(c):
                j += 1
            elif c in _JOINERS and j + 1 < n and _is_word_char(text[j + 1]) and j > cur:
                parts.append(text[cur:j])
                j += 1
                cur = j
            else:
                break
        parts.append(text[cur:j])
        if len(parts) > 1:
            out.append(text[i:j])
        for p in parts:
            if p not in stop:
                out.append(p)
        i = j
    return out
