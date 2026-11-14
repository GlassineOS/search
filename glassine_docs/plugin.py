# SPDX-License-Identifier: GPL-2.0-or-later
"""MkDocs plugin ``glassine-search``: builds the search files in the docs build.

    plugins:
      - search            # the theme's search: keep it, and list it first
      - glassine-search

Event order (mkdocs/commands/build.py, 1.6): every page is rendered to HTML
(``on_page_content``) before ``on_env``; templates are rendered after it. So
``on_env`` chunks and embeds all pages (with the pages of the repositories in
``sources_file``, if set) and adds the related-pages block to each page's HTML,
``on_post_page`` points Material's search worker (the "search" key of its
``#__config``) at ours, and ``on_post_build`` writes ``search/`` and the worker. The theme's search box, results, keyboard handling and look stay Material's.
"""
from __future__ import annotations

import html
import json
import os
import re
import subprocess
from pathlib import Path
from urllib.parse import quote

from mkdocs.config import config_options as c
from mkdocs.config.base import Config
from mkdocs.exceptions import PluginError
from mkdocs.plugins import BasePlugin, get_plugin_logger
from mkdocs.utils import get_relative_url

from .builder import PROFILES, Builder, Options, Page
from .sources import SourceError

log = get_plugin_logger(__name__)

# Material's config element; its "search" key names the search worker.
CONFIG_RE = re.compile(r'(<script id="__config" type="application/json">)(.*?)(</script>)', re.S)


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
    # other repositories' documentation folders (decision 0012), relative to mkdocs.yml
    sources_file = c.Optional(c.Type(str))
    source_cache = c.Type(str, default=".cache/glassine-search/sources")
    out_dir = c.Type(str, default="search")


def _env_bool(name: str) -> bool | None:
    v = os.environ.get(name)
    return None if v is None else v.strip().lower() in ("1", "true", "yes", "on")


def repo_path(path: Path, root: Path) -> str:
    """``path`` relative to the top of the git repository holding ``root`` (mkdocs.yml's
    directory, which need not be the top) if that repository tracks files in ``path``,
    else relative to ``root``; "" for the top."""
    top = root
    try:
        out = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=root, check=True,
                             capture_output=True, text=True).stdout.strip()
        # a docs tree that is not checked out, sitting inside another repository, is not
        # that repository's documentation
        tracked = subprocess.run(["git", "ls-files", "--", str(path)], cwd=root, check=True,
                                 capture_output=True, text=True).stdout.strip()
        if out and tracked:
            top = Path(out).resolve()
    except (OSError, subprocess.CalledProcessError):
        pass
    for base in (top, root):
        try:
            rel = path.relative_to(base).as_posix()
        except ValueError:
            continue
        return "" if rel == "." else rel
    return ""


class SearchPlugin(BasePlugin[_Config]):
    def on_config(self, config, **kw):
        cfg = self.config
        root = Path(config.config_file_path or ".").resolve().parent
        profiles = cfg.profiles
        if os.environ.get("GLASSINE_SEARCH_PROFILES"):
            profiles = [p for p in os.environ["GLASSINE_SEARCH_PROFILES"].split(",") if p]
        related = cfg.related_profile if cfg.related_profile in profiles else profiles[0]
        hf = _env_bool("GLASSINE_MODELS_HF")
        sources_file = os.environ.get("GLASSINE_SEARCH_SOURCES") or cfg.sources_file
        # <page> in chunk IDs is the path in the repository: the docs folder comes first
        self.folder = repo_path(Path(config.docs_dir).resolve(), root)
        self.opts = Options(
            profiles=tuple(profiles), related_profile=related, related_count=cfg.related_count,
            model_cache=Path(os.environ.get("GLASSINE_SEARCH_MODEL_CACHE") or root / cfg.model_cache),
            embed_cache=Path(os.environ.get("GLASSINE_SEARCH_EMBED_CACHE") or root / cfg.embed_cache),
            mirror=os.environ.get("GLASSINE_MODELS_MIRROR") or cfg.mirror,
            hf_fallback=cfg.hf_fallback if hf is None else hf,
            source=cfg.source, folder=self.folder, site_url=config.site_url or "", repo_dir=root,
            static_model_file=os.environ.get("GLASSINE_SEARCH_STATIC_MODEL") or None,
            sources_file=root / sources_file if sources_file else None,
            source_cache=Path(os.environ.get("GLASSINE_SEARCH_SOURCE_CACHE") or root / cfg.source_cache))
        if os.environ.get("GLASSINE_SEARCH_FETCH_URL"):  # a mirror, or a local repository in tests
            self.opts.fetch_url = os.environ["GLASSINE_SEARCH_FETCH_URL"]
        self.builder = Builder(self.opts)
        self.result = None
        self._warned = False
        return config

    def on_env(self, env, config, files, **kw):
        pages, objs = [], []
        prefix = f"{self.folder}/" if self.folder else ""
        for f in files.documentation_pages():
            p = f.page
            if p is None or p.content is None:
                continue
            pages.append(Page(src=prefix + f.src_uri, title=p.title or f.src_uri, url=p.url, html=p.content))
            objs.append(p)
        try:
            self.result = res = self.builder.build(pages)
        except SourceError as e:
            raise PluginError(f"glassine-search: {e}") from e
        log.info(f"{len(res.chunks)} chunks from {len(pages)} pages and "
                 f"{len(res.manifest['sources']) - 1} other sources, profiles "
                 f"{', '.join(self.opts.profiles)} ({res.stats})")
        own = self.opts.source
        by_src = {(own, pg.src): (pg, o) for pg, o in zip(pages, objs)}
        # where a related page links: this site's pages by their own title and URL, the
        # pages of other sources by their first chunk (an absolute URL)
        target = {}
        for ch in res.chunks:
            key = (ch.source, ch.page)
            if key in by_src:
                target[key] = (by_src[key][0].title, by_src[key][0].url or "./")
            elif key not in target:
                target[key] = (ch.title, ch.url.split("#", 1)[0])
        title = html.escape(self.config.related_title)
        for key, rel in res.related.items():
            if key not in by_src or not rel:
                continue
            pg, page = by_src[key]
            links = []
            for t, _ in rel:
                name, url = target[t]
                href = url if "://" in url else get_relative_url(url, pg.url or "./")
                links.append(f'<li><a href="{html.escape(href)}">{html.escape(name)}</a></li>')
            page.content += (f'\n<nav class="gds-related" data-search-exclude aria-label="{title}">'
                             f"<h2>{title}</h2><ul>{''.join(links)}</ul></nav>\n")
        return env

    def on_post_page(self, output, page, config, **kw):
        """Point Material's search worker at ours (Material's worker runs inside it)."""
        if not self.result or config.theme.name != "material":
            return output
        m = CONFIG_RE.search(output)
        try:
            cfg = json.loads(m.group(2)) if m else None
        except ValueError:
            cfg = None
        url = cfg.get("search") if isinstance(cfg, dict) else None
        if not url or "/workers/" not in url:
            if not self._warned:
                self._warned = True
                log.warning("no search worker in Material's #__config: semantic search not wired "
                            "(has Material changed its config?)")
            return output
        site_root = url.split("assets/javascripts/workers/")[0]
        material = url.rsplit("/", 1)[1]
        worker = self.result.manifest["web"]["worker"]
        # from the worker's directory to the site root, then to search/
        up = "../" * worker.count("/")
        cfg["search"] = (f"{site_root}{worker}?material={quote(material)}"
                         f"&search={quote(up + self.opts_out() + '/')}")
        blob = json.dumps(cfg, ensure_ascii=False).replace("</", "<\\/")
        return output[:m.start(2)] + blob + output[m.end(2):]

    def opts_out(self) -> str:
        return self.config.out_dir.strip("/")

    def on_post_build(self, config, **kw):
        if not self.result:
            return
        out = Path(config.site_dir) / self.opts_out()
        Builder.write(self.result, out)
        log.info(f"wrote {len(self.result.files) + 1} files to {out} and the search worker")

