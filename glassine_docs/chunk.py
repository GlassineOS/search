# SPDX-License-Identifier: GPL-2.0-or-later
"""Split a rendered page into chunks at its headings.

Input is the page's HTML body as MkDocs renders it (headings carry the ids
the table of contents links to). Each heading with an id from h2 down starts a
section; everything before the first one belongs to the page itself. Long
sections are cut at block boundaries into parts of at most ``max_words``
words; the parts after the first get the id suffix ``~2``, ``~3``, ... but
link to the same anchor.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from html.parser import HTMLParser

MAX_WORDS = 300
BLOCK = {"p", "li", "pre", "div", "table", "tr", "blockquote", "dt", "dd", "ul", "ol",
         "details", "summary", "section", "article", "figure", "figcaption", "hr", "br",
         "h1", "h2", "h3", "h4", "h5", "h6", "dl"}
CELL = {"td", "th"}
SKIP_TAGS = {"script", "style", "svg", "template", "noscript"}
VOID = {"br", "hr", "img", "input", "meta", "link", "area", "base", "col", "embed",
        "param", "source", "track", "wbr"}
HEADINGS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}


@dataclass
class Section:
    anchor: str | None
    heading: str
    level: int
    path: list[str]
    blocks: list[str] = field(default_factory=list)


@dataclass
class Chunk:
    id: str
    page: str
    title: str
    heading: str
    path: list[str]
    url: str
    text: str

    def embed_text(self) -> str:
        """What gets embedded and indexed: the heading trail, then the text."""
        trail = " › ".join([self.title] + [p for p in self.path if p != self.title])
        return trail + "\n" + self.text


def _classes(attrs) -> set[str]:
    for k, v in attrs:
        if k == "class" and v:
            return set(v.split())
    return set()


class _Parser(HTMLParser):
    def __init__(self, title: str):
        super().__init__(convert_charrefs=True)
        self.sections = [Section(None, title, 1, [])]
        self.stack: list[tuple[str, bool]] = []  # (tag, skipping)
        self.skip_depth = 0
        self.in_pre = 0
        self.cur: list[str] = []
        self.heading: tuple[int, str | None] | None = None
        self.heading_text: list[str] = []
        self.trail: list[tuple[int, str]] = []

    # text accumulation -----------------------------------------------------
    def _flush(self):
        text = "".join(self.cur)
        self.cur = []
        if self.in_pre:
            lines = [ln.rstrip() for ln in text.split("\n")]
            text = "\n".join(ln for ln in lines if ln.strip())
        else:
            text = re.sub(r"\s+", " ", text).strip()
        if text:
            self.sections[-1].blocks.append(text)

    def handle_starttag(self, tag, attrs):
        cls = _classes(attrs)
        skip = (tag in SKIP_TAGS or "headerlink" in cls or "mermaid" in cls
                or any(k == "data-search-exclude" for k, _ in attrs))
        if tag in VOID:
            if tag in BLOCK and not self.skip_depth and self.heading is None:
                self._flush()
            return
        self.stack.append((tag, skip))
        if skip:
            self.skip_depth += 1
            return
        if self.skip_depth:
            return
        if tag in HEADINGS:
            self._flush()
            ident = dict(attrs).get("id")
            self.heading = (HEADINGS[tag], ident)
            self.heading_text = []
        elif tag in BLOCK:
            if self.heading is None:
                self._flush()
            if tag == "pre":
                self.in_pre += 1
        elif tag in CELL:
            self.cur.append(" ")

    def handle_endtag(self, tag):
        if tag in VOID:
            return
        # pop to the matching tag (tolerates unclosed children)
        while self.stack:
            t, skip = self.stack.pop()
            if skip:
                self.skip_depth -= 1
            elif not self.skip_depth:
                self._close(t)
            if t == tag:
                break

    def _close(self, tag):
        if tag in HEADINGS and self.heading is not None:
            level, ident = self.heading
            text = re.sub(r"\s+", " ", "".join(self.heading_text)).strip()
            self.heading = None
            if level >= 2 and ident:
                self.trail = [(lv, t) for lv, t in self.trail if lv < level]
                self.trail.append((level, text))
                self.sections.append(Section(ident, text, level, [t for _, t in self.trail]))
            elif level == 1:
                self.sections[0].heading = text or self.sections[0].heading
            else:
                self.sections[-1].blocks.append(text)
        elif tag in BLOCK:
            self._flush()
            if tag == "pre":
                self.in_pre -= 1
        elif tag in CELL:
            self.cur.append(" ")

    def handle_data(self, data):
        if self.skip_depth:
            return
        if self.heading is not None:
            self.heading_text.append(data)
        else:
            self.cur.append(data)

    def close(self):
        super().close()
        while self.stack:
            self.handle_endtag(self.stack[-1][0])
        self._flush()


def _split(blocks: list[str], max_words: int) -> list[list[str]]:
    parts, cur, n = [], [], 0
    for b in blocks:
        w = len(b.split())
        if cur and n + w > max_words:
            parts.append(cur)
            cur, n = [], 0
        cur.append(b)
        n += w
    if cur:
        parts.append(cur)
    return parts


def chunk_page(html: str, *, source: str, commit: str, page: str, title: str, url: str,
               max_words: int = MAX_WORDS) -> list[Chunk]:
    """Chunks of one page. ``page`` is its source path, ``url`` its absolute URL."""
    p = _Parser(title)
    p.feed(html)
    p.close()
    title = p.sections[0].heading or title
    out = []
    for s in p.sections:
        if not s.blocks:
            continue
        base = f"{source}@{commit}:{page}" + (f"#{s.anchor}" if s.anchor else "")
        link = url + (f"#{s.anchor}" if s.anchor else "")
        for i, blocks in enumerate(_split(s.blocks, max_words)):
            out.append(Chunk(
                id=base + (f"~{i + 1}" if i else ""), page=page, title=title,
                heading=s.heading if s.anchor else title, path=s.path, url=link,
                text="\n".join(blocks)))
    return out
