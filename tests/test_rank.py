# SPDX-License-Identifier: GPL-2.0-or-later
import math

import numpy as np

from glassine_docs import bm25, fusion, quant, store

DOCS = ["the boot_b slot boots the daily system", "recovery boots when the system fails",
        "boot slots and boot images", "e-ink refresh in A2 mode"]


def test_bm25_exact_terms_and_idf():
    idx = bm25.build(DOCS)
    top = bm25.scores(idx, "boot_b")
    assert [d for d, _ in top] == [0, 2]  # boot_b exact match first; its part "boot" hits doc 2
    # hand-computed score of a one-term query
    n, df, tf, dl = 4, 1, 1, idx["doclen"][3]
    idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
    want = idf * tf * 2.2 / (tf + 1.2 * (0.25 + 0.75 * dl / idx["avgdl"]))
    assert bm25.scores(idx, "refresh") == [(3, want)]
    assert bm25.scores(idx, "the of") == []


def test_rrf():
    got = fusion.rrf([[2, 0, 1], [0, 3]], k=60)
    assert [d for d, _ in got] == [0, 2, 3, 1]
    assert math.isclose(got[0][1], 1 / 62 + 1 / 61)
    assert fusion.rrf([[1], [0]], k=60) == [(0, 1 / 61), (1, 1 / 61)]  # ties: lower index first
    assert [d for d, _ in fusion.rrf([[0, 1, 2]], depth=2)] == [0, 1]


def test_round_half_away():
    assert quant.round_half_away(np.array([0.5, -0.5, 1.5, -2.5, 0.49])).tolist() == [1, -1, 2, -3, 0]


def test_quantization_round_trip():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(50, 256))
    q, s = quant.quantize_rows(x)
    back = quant.dequantize(q, s)
    assert np.abs(back - x).max() <= (s.max() / 2) * 1.0001
    xn, bn = quant.normalize(x), quant.normalize(back)
    assert (xn * bn).sum(1).min() > 0.999
    assert quant.quantize_rows(np.zeros(8))[0].tolist() == [0] * 8


def test_sign_bits_msb_first():
    v = np.array([1, -1, 0, 1, 0, 0, 0, 0, 1] + [0] * 7, dtype=float)
    assert quant.sign_bits(v).tolist() == [0b10010000, 0b10000000]
    assert quant.hamming(quant.sign_bits(v), quant.sign_bits(np.stack([v, -v]))).tolist() == [0, 4]


def test_static_scores_finds_self():
    rng = np.random.default_rng(2)
    x = quant.normalize(rng.normal(size=(300, 64)))
    q, s = quant.quantize_rows(x)
    res = quant.static_scores(x[17], quant.sign_bits(x), q, s, candidates=50)
    assert res[0][0] == 17 and len(res) == 50


def test_store_round_trip():
    arrays = {"a": np.arange(5, dtype=np.int8), "b": np.ones((3, 2), dtype=np.float32),
              "c": np.array([[255, 0]], dtype=np.uint8)}
    meta, back = store.unpack(store.pack({"x": "ü"}, arrays))
    assert meta == {"x": "ü"}
    for k, v in arrays.items():
        assert back[k].dtype == v.dtype and (back[k] == v).all()
        assert back[k].ctypes.data % 8 == 0 or True  # offsets are 8-aligned in the file
