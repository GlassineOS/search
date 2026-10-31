# SPDX-License-Identifier: GPL-2.0-or-later
"""Reciprocal rank fusion. Mirrored in ``web/score.js``.

score(d) = sum over lists of 1 / (k + rank), rank counted from 1, each list cut
to ``depth`` entries. Lists are added in the order given; ties go to the lower
chunk index.
"""
from __future__ import annotations

K = 60
DEPTH = 100


def rrf(lists: list[list[int]], k: int = K, depth: int = DEPTH) -> list[tuple[int, float]]:
    acc: dict[int, float] = {}
    for lst in lists:
        for r, d in enumerate(lst[:depth]):
            acc[d] = acc.get(d, 0.0) + 1.0 / (k + r + 1)
    return sorted(acc.items(), key=lambda p: (-p[1], p[0]))
