# SPDX-License-Identifier: GPL-2.0-or-later
"""The static profile's model: a model2vec embedding table, converted once to int8.

``convert`` turns the pinned model2vec files into ``static-model.bin`` (the
container of ``store.py``). That file is what the browser, the CLI and the
builder all embed with, so query and document vectors come from the same table.
Embedding follows model2vec's ``StaticModel.encode``: no [CLS]/[SEP], the text
cut to ``max_length * median_token_length`` characters, at most ``max_length``
tokens, [UNK] ids dropped, the mean of the token rows, then L2 normalisation
when the model's config says ``normalize``. Mirrored in ``web/embed.js``.
"""
from __future__ import annotations

import json
import math
import struct
from pathlib import Path

import numpy as np

from . import quant, store
from .text import WordPiece, hf_tokenizer_config

DEFAULT_MAX_LENGTH = 512  # model2vec's default when the config names none


def read_safetensors(path: Path) -> dict[str, np.ndarray]:
    data = Path(path).read_bytes()
    (n,) = struct.unpack_from("<Q", data, 0)
    header = json.loads(data[8:8 + n])
    out = {}
    for name, t in header.items():
        if name == "__metadata__":
            continue
        if t["dtype"] != "F32":
            raise ValueError(f"{path}: {name} is {t['dtype']}, expected F32")
        a, b = t["data_offsets"]
        out[name] = np.frombuffer(data, dtype="<f4", count=(b - a) // 4, offset=8 + n + a).reshape(t["shape"])
    return out


def convert(model_dir: Path, entry: dict) -> bytes:
    """Build static-model.bin from a model2vec model directory."""
    model_dir = Path(model_dir)
    cfg = json.loads((model_dir / "config.json").read_text())
    tensors = read_safetensors(model_dir / "model.safetensors")
    if set(tensors) != {"embeddings"}:
        # weights/mappings (vocabulary quantization) are not implemented
        raise ValueError(f"unsupported model2vec tensors: {sorted(tensors)}")
    emb = tensors["embeddings"]
    tok = hf_tokenizer_config(model_dir / "tokenizer.json")
    if emb.shape[0] != len(tok["vocab"]):
        raise ValueError("embedding rows != vocabulary size")
    q, scale = quant.quantize_rows(emb)
    meta = {
        "kind": "static-model",
        "model": entry["id"], "revision": entry["revision"],
        "source_sha256": {f["path"]: f["sha256"] for f in entry["files"]},
        "dims": int(emb.shape[1]),
        "normalize": bool(cfg.get("normalize", False)),
        "max_length": cfg.get("max_length", DEFAULT_MAX_LENGTH),
        "median_token_length": int(np.median([len(t) for t in tok["vocab"]])),
        "tokenizer": tok,
    }
    return store.pack(meta, {"scale": scale, "table": q})


class StaticModel:
    def __init__(self, data: bytes):
        meta, arrays = store.unpack(data)
        if meta.get("kind") != "static-model":
            raise ValueError("not a static model file")
        self.meta = meta
        self.tokenizer = WordPiece(meta["tokenizer"])
        self.table = arrays["table"]
        self.scale = arrays["scale"].astype(np.float64)
        self.dims = meta["dims"]
        self.normalize = meta["normalize"]
        self.max_length = meta["max_length"]
        self.median = meta["median_token_length"]

    @classmethod
    def load(cls, path: Path) -> "StaticModel":
        return cls(Path(path).read_bytes())

    def ids(self, text: str) -> list[int]:
        if self.max_length is not None:
            text = text[: self.max_length * self.median]
        ids = self.tokenizer.encode(text, add_special_tokens=False, max_length=self.max_length)
        return [i for i in ids if i != self.tokenizer.unk_id]

    def embed(self, text: str) -> np.ndarray:
        ids = self.ids(text)
        if not ids:
            return np.zeros(self.dims)
        # Sequential sums (row by row, then element by element), the order web/score.js
        # adds in, so both give the same bits.
        v = np.zeros(self.dims)
        for i in ids:
            v += self.table[i].astype(np.float64) * self.scale[i]
        v /= len(ids)
        if self.normalize:
            ss = 0.0
            for x in v.tolist():  # not sum(): it compensates since Python 3.12
                ss += x * x
            norm = math.sqrt(ss)
            v = v / (norm + 1e-32)
        return v

    def embed_many(self, texts: list[str]) -> np.ndarray:
        return np.stack([self.embed(t) for t in texts]) if texts else np.zeros((0, self.dims))
