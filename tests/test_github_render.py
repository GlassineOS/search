# SPDX-License-Identifier: GPL-2.0-or-later
"""Our rendering of other repositories' pages against GitHub's, on pinned public pages.

Network: run with ``make check-github`` (GDS_GITHUB=1). It reads GitHub's rendering of
each page at a pinned commit and checks that our headings get the same anchors, in the
same order, and that the page's title is its first heading. These pages once differed:
GlassineOS/boot docs/blocklist.md has lines starting "#55" that GitHub shows as text.
"""
import os
import re
import urllib.request

import pytest

if not os.environ.get("GDS_GITHUB"):
    pytest.skip("network test: make check-github", allow_module_level=True)

from glassine_docs.chunk import chunk_page  # noqa: E402
from glassine_docs.sources import render  # noqa: E402

BOOT = "bb5adca9cf88dd789d9b7bc662276fe74644cee2"
DOCS = "9d8f0b9806059564a94ce195aef2df9498c31b49"
SEARCH = "3d4e3e2dbff86573ae1d88ed5fe4c9b8b1b710e4"
PAGES = [
    ("GlassineOS/boot", BOOT, "docs/blocklist.md", "Block-list hand-off: a cold boot of a kernel file"),
    ("GlassineOS/boot", BOOT, "docs/deployments.md", None),
    ("GlassineOS/boot", BOOT, "docs/kernel-artifacts.md", None),
    ("GlassineOS/search", SEARCH, "docs/format.md", "Search file format, version 1"),
    ("GlassineOS/docs", DOCS, "docs/decisions/0012-where-documentation-lives.md", None),
    ("GlassineOS/docs", DOCS, "docs/devices/onyx-noteair/index.md", None),
    ("GlassineOS/docs", DOCS, "docs/guides/ab-on-single-slot.md", None),
]


def get(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "glassine-docs tests"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode("utf-8")


@pytest.mark.parametrize("repo, commit, path, title", PAGES, ids=[f"{r}/{p}" for r, _, p, _ in PAGES])
def test_anchors_match_github(repo, commit, path, title):
    page = get(f"https://github.com/{repo}/blob/{commit}/{path}")
    theirs = list(dict.fromkeys(re.findall(r'user-content-([^"\\&]*)', page)))
    html = render(get(f"https://raw.githubusercontent.com/{repo}/{commit}/{path}"))
    ours = re.findall(r'<h[1-6] id="([^"]*)"', html)
    assert theirs, "no anchors found in GitHub's page: has its markup changed?"
    assert ours == theirs
    if title:
        assert chunk_page(html, source="x", commit=commit, page=path, title="t", url="u/")[0].title == title
