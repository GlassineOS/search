# SPDX-License-Identifier: GPL-2.0-or-later
"""Quantization and scoring. Mirrored in ``web/score.js``.

int8 with one float32 scale per row: ``q = round(x / s)``, ``s = max|x| / 127``,
rounding half away from zero (JavaScript's ``Math.round`` rounds half up, so
neither side uses its built-in rounding). Sign bits: bit set when ``x > 0``,
packed most significant bit first (numpy ``packbits``).
"""
from __future__ import annotations

import numpy as np

_POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.int32)


def round_half_away(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


def quantize_rows(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        q, s = quantize_rows(x[None, :])
        return q[0], s[0]
    scale = (np.abs(x).max(axis=1) / 127.0).astype(np.float32)
    s64 = scale.astype(np.float64)
    safe = np.where(s64 > 0, s64, 1.0)
    q = np.clip(round_half_away(x / safe[:, None]), -127, 127).astype(np.int8)
    q[s64 == 0] = 0
    return q, scale


def dequantize(q: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return q.astype(np.float64) * scale.astype(np.float64)[..., None]


def sign_bits(x: np.ndarray) -> np.ndarray:
    return np.packbits(np.asarray(x) > 0, axis=-1)


def hamming(qbits: np.ndarray, bits: np.ndarray) -> np.ndarray:
    return _POP[np.bitwise_xor(bits, qbits[None, :])].sum(axis=1)


def normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    return v / (np.sqrt((v * v).sum(axis=-1, keepdims=True)) + 1e-32)


def rank(scores, ascending: bool = False) -> list[int]:
    """Indices by score (descending unless ``ascending``), ties by index."""
    s = np.asarray(scores, dtype=np.float64)
    keys = s if ascending else -s
    return [int(i) for i in np.lexsort((np.arange(len(s)), keys))]


def static_scores(qvec: np.ndarray, bits: np.ndarray, q8: np.ndarray, scale: np.ndarray,
                  candidates: int) -> list[tuple[int, float]]:
    """Binary pre-selection, then int8 rescoring. Returns (index, score), best first."""
    qq, qs = quantize_rows(qvec)
    ham = hamming(sign_bits(qvec), bits)
    cand = rank(ham, ascending=True)[:candidates]
    idot = q8[cand].astype(np.int64) @ qq.astype(np.int64)
    sc = (idot.astype(np.float64) * float(qs)) * scale[cand].astype(np.float64)
    order = rank(sc)
    # rank() ties by position in cand, which is the binary order; break ties by index instead
    pairs = sorted(((cand[i], float(sc[i])) for i in order), key=lambda p: (-p[1], p[0]))
    return pairs


def int8_scores(qvec: np.ndarray, q8: np.ndarray, scale: np.ndarray) -> list[tuple[int, float]]:
    """Full int8 scan with a float query (the transformer profile)."""
    sc = (q8.astype(np.float64) @ np.asarray(qvec, dtype=np.float64)) * scale.astype(np.float64)
    return [(i, float(sc[i])) for i in rank(sc)]
