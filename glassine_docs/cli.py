# SPDX-License-Identifier: GPL-2.0-or-later
"""glassine-docs: search the GlassineOS documentation from a terminal or an agent.

    glassine-docs search "how do I get back into recovery" [--profile transformer|static]
                         [-k N] [--json] [--base URL|DIR] [--offline]
"""
from __future__ import annotations

import argparse
import json
import sys
import textwrap

from . import __version__
from .client import DEFAULT_BASE, FetchError, Searcher, Site
from .models import ModelError


def _trim(text: str, width: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= width else text[: width - 1].rsplit(" ", 1)[0] + " …"


def cmd_search(a) -> int:
    profile = a.profile or "transformer"
    if profile == "transformer" and a.profile is None:
        try:
            import onnxruntime  # noqa: F401
        except ImportError:
            print("glassine-docs: onnxruntime is not installed, using --profile static", file=sys.stderr)
            profile = "static"
    try:
        site = Site(a.base, offline=a.offline)
        s = Searcher(site, profile, models_mirror=a.models_mirror, hf=a.hf)
        hits, secs = s.search(a.query, a.k)
    except (FetchError, ModelError) as e:
        print(f"glassine-docs: {e}", file=sys.stderr)
        return 2
    m = site.manifest
    if a.json:
        print(json.dumps({
            "query": a.query, "profile": profile, "commit": m["commit"], "sources": m["sources"],
            "model": s.prof["model"]["id"], "ms": round(secs * 1000, 2),
            "results": [{"rank": h.rank, "score": round(h.score, 6), "id": h.chunk["id"],
                         "title": h.chunk["title"], "heading": h.chunk["heading"], "url": h.url,
                         "text": h.chunk["text"], "bm25_rank": h.bm25_rank, "vector_rank": h.vector_rank,
                         "vector_score": None if h.vector_score is None else round(h.vector_score, 6)}
                        for h in hits]}, ensure_ascii=False, indent=1))
        return 0
    for h in hits:
        c = h.chunk
        where = c["title"] if c["heading"] == c["title"] else f"{c['title']} › {c['heading']}"
        print(f"{h.rank}. {h.score:.4f}  {where}")
        print(f"   {h.url}")
        print(textwrap.indent(textwrap.fill(_trim(c["text"], a.width * 3), a.width - 3), "   "))
    print(f"docs commit {m['commit']} · profile {profile} · {secs * 1000:.0f} ms")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="glassine-docs", description="Search the GlassineOS documentation.")
    p.add_argument("--version", action="version", version=f"glassine-docs {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("search", help="search by keywords and meaning")
    s.add_argument("query")
    s.add_argument("--profile", choices=["transformer", "static"],
                   help="embedding profile (default transformer; static needs no onnxruntime)")
    s.add_argument("-k", type=int, default=5, help="number of results (default 5)")
    s.add_argument("--json", action="store_true", help="machine-readable output")
    s.add_argument("--base", default=DEFAULT_BASE, help=f"search files: URL or local directory (default {DEFAULT_BASE})")
    s.add_argument("--offline", action="store_true", help="use the cache only")
    s.add_argument("--models-mirror", help="where model files come from (default: the GlassineOS/search release)")
    s.add_argument("--hf", action="store_true", help="fall back to Hugging Face for model files (development)")
    s.add_argument("--width", type=int, default=88, help=argparse.SUPPRESS)
    a = p.parse_args(argv)
    if a.cmd == "search":
        return cmd_search(a)
    return 1


if __name__ == "__main__":
    sys.exit(main())
