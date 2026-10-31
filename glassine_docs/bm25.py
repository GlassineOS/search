# SPDX-License-Identifier: GPL-2.0-or-later
"""BM25 keyword index over the chunks. Mirrored in ``web/score.js``.

Okapi BM25 with Lucene's idf, ``log(1 + (N - df + 0.5) / (df + 0.5))``.
Scores are summed over the query's distinct terms in sorted order, so both
implementations add the same floats in the same order.
"""
from __future__ import annotations

import math

from .text import STOPWORDS, bm25_terms

K1 = 1.2
B = 0.75


def build(texts: list[str], k1: float = K1, b: float = B) -> dict:
    postings: dict[str, dict[int, int]] = {}
    lengths = []
    for d, t in enumerate(texts):
        terms = bm25_terms(t)
        lengths.append(len(terms))
        for term in terms:
            postings.setdefault(term, {}).setdefault(d, 0)
            postings[term][d] += 1
    n = len(texts)
    return {
        "k1": k1, "b": b, "n": n,
        "avgdl": (sum(lengths) / n) if n else 0.0,
        "doclen": lengths,
        "stopwords": STOPWORDS,
        # term -> flat [doc, tf, doc, tf, ...], docs ascending
        "postings": {t: [x for d in sorted(p) for x in (d, p[d])] for t, p in sorted(postings.items())},
    }


def scores(index: dict, query: str) -> list[tuple[int, float]]:
    """(doc, score) for docs with a positive score, best first, ties by doc."""
    n, k1, b, avgdl = index["n"], index["k1"], index["b"], index["avgdl"]
    doclen = index["doclen"]
    acc: dict[int, float] = {}
    for term in sorted(set(bm25_terms(query, set(index["stopwords"])))):
        post = index["postings"].get(term)
        if not post:
            continue
        df = len(post) // 2
        idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
        for i in range(0, len(post), 2):
            d, tf = post[i], post[i + 1]
            norm = tf * (k1 + 1) / (tf + k1 * (1 - b + b * doclen[d] / avgdl))
            acc[d] = acc.get(d, 0.0) + idf * norm
    return sorted(acc.items(), key=lambda p: (-p[1], p[0]))
