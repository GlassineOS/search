# SPDX-License-Identifier: GPL-2.0-or-later
"""MkDocs plugin ``glassine-search``: builds the search files in the docs build.

    plugins:
      - search            # the theme's search: keep it, and list it first
      - glassine-search

Event order (mkdocs/commands/build.py, 1.6): every page is rendered to HTML
(``on_page_content``) before ``on_env``; templates are rendered after it. So
``on_env`` chunks and embeds all pages and adds the related-pages block to
each page's HTML, ``on_post_page`` adds the few inline lines that load the
search UI on first use, and ``on_post_build`` writes ``search/``.
"""
from __future__ import annotations

import html
import os
from pathlib import Path

from mkdocs.config import config_options as c
from mkdocs.config.base import Config
from mkdocs.plugins import BasePlugin, get_plugin_logger
from mkdocs.utils import get_relative_url

from .builder import PROFILES, Builder, Options, Page

log = get_plugin_logger(__name__)

BOOT = """<script>(function(){var b=%s,u=%s,on=0;function go(e){var t=e.target;if(on||!t||!t.matches||!t.matches('[data-md-component="search-query"]'))return;on=1;var s=document.createElement("script");s.src=b+u;s.setAttribute("data-base",b);document.head.appendChild(s)}document.addEventListener("focusin",go,true);document.addEventListener("keydown",go,true)})()</script>"""


class _Config(Config):
    profiles = c.ListOfItems(c.Choice(PROFILES), default=list(PROFILES))
    related_profile = c.Choice(PROFILES, default="static")
    related_count = c.Type(int, default=5)
    related_title = c.Type(str, default="Related pages")
    model_cache = c.Type(str, default=".cache/glassine-search/models")
    embed_cache = c.Type(str, default=".cache/glassine-search/embeddings")
    mirror = c.Optional(c.Type(str))
    hf_fallback = c.Type(bool, default=False)
    source = c.Type(str, default="docs")
    out_dir = c.Type(str, default="search")


def _env_bool(name: str) -> bool | None:
    v = os.environ.get(name)
    return None if v is None else v.strip().lower() in ("1", "true", "yes", "on")


class SearchPlugin(BasePlugin[_Config]):
    def on_config(self, config, **kw):
        cfg = self.config
        root = Path(config.config_file_path or ".").resolve().parent
        profiles = cfg.profiles
        if os.environ.get("GLASSINE_SEARCH_PROFILES"):
            profiles = [p for p in os.environ["GLASSINE_SEARCH_PROFILES"].split(",") if p]
        related = cfg.related_profile if cfg.related_profile in profiles else profiles[0]
        hf = _env_bool("GLASSINE_MODELS_HF")
        self.opts = Options(
            profiles=tuple(profiles), related_profile=related, related_count=cfg.related_count,
            model_cache=Path(os.environ.get("GLASSINE_SEARCH_MODEL_CACHE") or root / cfg.model_cache),
            embed_cache=Path(os.environ.get("GLASSINE_SEARCH_EMBED_CACHE") or root / cfg.embed_cache),
            mirror=os.environ.get("GLASSINE_MODELS_MIRROR") or cfg.mirror,
            hf_fallback=cfg.hf_fallback if hf is None else hf,
            source=cfg.source, site_url=config.site_url or "", repo_dir=root,
            static_model_file=os.environ.get("GLASSINE_SEARCH_STATIC_MODEL") or None)
        self.builder = Builder(self.opts)
        self.result = None
        return config

    def on_env(self, env, config, files, **kw):
        pages, objs = [], []
        for f in files.documentation_pages():
            p = f.page
            if p is None or p.content is None:
                continue
            pages.append(Page(src=f.src_uri, title=p.title or f.src_uri, url=p.url, html=p.content))
            objs.append(p)
        self.result = res = self.builder.build(pages)
        log.info(f"{len(res.chunks)} chunks from {len(pages)} pages, profiles "
                 f"{', '.join(self.opts.profiles)} ({res.stats})")
        by_src = {pg.src: (pg, o) for pg, o in zip(pages, objs)}
        title = html.escape(self.config.related_title)
        for src, rel in res.related.items():
            if src not in by_src or not rel:
                continue
            pg, page = by_src[src]
            items = "".join(
                f'<li><a href="{html.escape(get_relative_url(by_src[t][0].url or "./", pg.url or "./"))}">'
                f"{html.escape(by_src[t][0].title)}</a></li>" for t, _ in rel if t in by_src)
            page.content += (f'\n<nav class="gds-related" data-search-exclude aria-label="{title}">'
                             f"<h2>{title}</h2><ul>{items}</ul></nav>\n")
        return env

    def on_post_page(self, output, page, config, **kw):
        if not self.result or "</body>" not in output:
            return output
        base = get_relative_url(self.opts_out() + "/", page.url or "./")
        if not base.endswith("/"):
            base += "/"
        boot = BOOT % (_js(base), _js(self.result.manifest["web"]["ui.js"]))
        i = output.rfind("</body>")
        return output[:i] + boot + output[i:]

    def opts_out(self) -> str:
        return self.config.out_dir.strip("/")

    def on_post_build(self, config, **kw):
        if not self.result:
            return
        out = Path(config.site_dir) / self.opts_out()
        Builder.write(self.result, out)
        log.info(f"wrote {len(self.result.files) + 1} files to {out}")


def _js(s: str) -> str:
    import json
    return json.dumps(s).replace("</", "<\\/")
