# SPDX-License-Identifier: GPL-2.0-or-later
"""Checks against the real models. Opt-in: GDS_MODELS=<model cache dir> (make check-models)."""
import json

import numpy as np
import pytest

from conftest import real_models_dir
from glassine_docs import models
from glassine_docs.staticmodel import StaticModel, convert
from helpers import FIX

pytestmark = pytest.mark.skipif(real_models_dir() is None, reason="GDS_MODELS not set")
TEXTS = json.loads((FIX / "token-texts.json").read_text(encoding="utf-8"))


def test_lock_hashes():
    lock = models.load_lock()
    for prof, entry in lock["models"].items():
        for f in entry["files"] + entry.get("derived", []):
            p = models.cached_path(real_models_dir(), entry, f["path"])
            assert models.sha256_file(p) == f["sha256"], f["path"]


def test_static_matches_model2vec():
    model2vec = pytest.importorskip("model2vec")
    entry = models.load_lock()["models"]["static"]
    d = models.model_dir(real_models_dir(), entry)
    ours = StaticModel(convert(d, entry))
    ref = model2vec.StaticModel.from_pretrained(str(d))
    texts = TEXTS + ["recovery and boot slots " * 300]
    for t, r in zip(texts, ref.encode(texts)):
        v = ours.embed(t)
        if not np.any(r):
            assert not np.any(v), t
            continue
        assert float(v @ r / np.linalg.norm(r)) > 0.999, t  # int8 table vs float32 table
    assert ours.dims == 256 and entry["license"] == "MIT"


def test_transformer_embeds():
    from glassine_docs.transformer import TransformerModel
    entry = models.load_lock()["models"]["transformer"]
    d = models.model_dir(real_models_dir(), entry)
    m = TransformerModel(d / entry["onnx"], d / "tokenizer.json", query_prefix=entry["query_prefix"])
    v = m.embed_many(["The recovery partition boots when the system does not.", "e-ink refresh"])
    q = m.embed_query("how do I get into recovery")
    assert v.shape == (2, 384) and np.allclose(np.linalg.norm(v, axis=1), 1)
    assert q @ v[0] > q @ v[1]
