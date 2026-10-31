# SPDX-License-Identifier: GPL-2.0-or-later
"""The transformer profile's model (bge-small-en-v1.5 in ONNX), on CPU.

[CLS] pooling, then L2 normalisation, as the model's sentence-transformers
config says. Queries get the model's query prefix; passages get none.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from . import quant
from .text import WordPiece


class TransformerModel:
    def __init__(self, onnx_path: Path, tokenizer_json: Path, *, query_prefix: str = "",
                 max_length: int = 512, threads: int | None = None):
        try:
            import onnxruntime as ort
        except ImportError as e:  # pragma: no cover
            raise SystemExit("the transformer profile needs onnxruntime: pip install onnxruntime "
                             "(or use --profile static)") from e
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads or min(4, os.cpu_count() or 1)
        opts.log_severity_level = 3
        self.session = ort.InferenceSession(str(onnx_path), opts, providers=["CPUExecutionProvider"])
        self.inputs = {i.name for i in self.session.get_inputs()}
        self.tokenizer = WordPiece.from_hf_json(tokenizer_json)
        self.query_prefix = query_prefix
        self.max_length = max_length

    def _run(self, batch: list[list[int]]) -> np.ndarray:
        n = max(len(x) for x in batch)
        ids = np.zeros((len(batch), n), dtype=np.int64)
        mask = np.zeros((len(batch), n), dtype=np.int64)
        for i, x in enumerate(batch):
            ids[i, :len(x)] = x
            mask[i, :len(x)] = 1
        feed = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self.inputs:
            feed["token_type_ids"] = np.zeros_like(ids)
        out = self.session.run(None, feed)[0]
        return quant.normalize(out[:, 0, :].astype(np.float64))

    def embed_many(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        enc = [self.tokenizer.encode(t, add_special_tokens=True, max_length=self.max_length) for t in texts]
        # sort by length so batches pad little; restore the order afterwards
        order = sorted(range(len(enc)), key=lambda i: len(enc[i]))
        out = np.zeros((len(enc), 0))
        res = {}
        for s in range(0, len(order), batch_size):
            idx = order[s:s + batch_size]
            vecs = self._run([enc[i] for i in idx])
            for i, v in zip(idx, vecs):
                res[i] = v
        if res:
            out = np.stack([res[i] for i in range(len(enc))])
        return out

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_many([self.query_prefix + text])[0]
