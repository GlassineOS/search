# SPDX-License-Identifier: GPL-2.0-or-later
from glassine_docs.chunk import chunk_page
from helpers import PAGES


def chunks_of(i, **kw):
    src, title, url, html = PAGES[i]
    return chunk_page(html, source="docs", commit="abc", page=src, title=title, url=url, **kw)


def test_ids_use_real_anchors():
    cs = chunks_of(1)
    assert [c.id for c in cs] == [
        "docs@abc:guides/recovery.md",
        "docs@abc:guides/recovery.md#enter-recovery",
        "docs@abc:guides/recovery.md#edl",
    ]
    assert cs[1].url == "guides/recovery/#enter-recovery"
    assert cs[0].url == "guides/recovery/"


def test_headerlinks_and_mermaid_are_dropped():
    cs = chunks_of(1)
    assert all("¶" not in c.text and "¶" not in c.heading for c in cs)
    assert cs[1].heading == "Enter recovery"
    assert "graph" not in cs[2].text
    assert "fastboot oem reboot-recovery" in cs[1].text


def test_heading_trail_and_embed_text():
    cs = chunks_of(3)
    tries = [c for c in cs if c.id.endswith("#tries")][0]
    assert tries.path == ["When an update fails", "Tries"]
    assert tries.embed_text().startswith("Updates › When an update fails › Tries\n")


def test_empty_sections_skipped_and_tables_kept():
    cs = chunks_of(2)
    assert [c.id.split("#")[-1] for c in cs] == ["context", "decision"]
    assert "recovery key printed" in cs[1].text


def test_long_sections_split_with_suffix():
    html = '<h2 id="long">Long</h2>' + "".join(f"<p>{'word ' * 120}</p>" for _ in range(5))
    cs = chunk_page(html, source="docs", commit="c", page="p.md", title="P", url="p/", max_words=300)
    assert [c.id for c in cs] == ["docs@c:p.md#long", "docs@c:p.md#long~2", "docs@c:p.md#long~3"]
    assert all(c.url == "p/#long" for c in cs)
    assert all(len(c.text.split()) <= 300 for c in cs)


def test_search_exclude_is_skipped():
    html = '<p>keep</p><nav data-search-exclude><p>drop</p></nav>'
    cs = chunk_page(html, source="docs", commit="c", page="p.md", title="P", url="p/")
    assert cs[0].text == "keep"
