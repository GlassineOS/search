# SPDX-License-Identifier: GPL-2.0-or-later
"""Search client: fetches the published search files, caches them, answers queries.

The manifest is revalidated with its ETag on every search (one short request);
every other file is named by its content hash, so it is downloaded once and
checked against the sha256 in the manifest. Without a network the cached
manifest is used. ``base`` may also be a local directory (a built
``site/search/``), which is read in place.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin

from . import bm25, fusion, models, quant, store
from .staticmodel import StaticModel

DEFAULT_BASE = "https://docs.glassine.org/search/"


def cache_root() -> Path:
    xdg = os.environ.get("XDG_CACHE_HOME")
    return (Path(xdg) if xdg else Path.home() / ".cache") / "glassine" / "docs-search"


class FetchError(RuntimeError):
    pass


def _note(msg: str) -> None:
    print(f"glassine-docs: {msg}", file=sys.stderr)


@dataclass
class Hit:
    rank: int
    score: float
    chunk: dict
    url: str
    bm25_rank: int | None
    vector_rank: int | None
    vector_score: float | None


class Site:
    """The search files of one published site (or a local directory)."""

    def __init__(self, base: str = DEFAULT_BASE, cache: Path | None = None, offline: bool = False):
        self.local = None
        if "://" not in base:
            self.local = Path(base)
            if not (self.local / "manifest.json").is_file():
                raise FetchError(f"{base}: no manifest.json")
            base = self.local.resolve().as_uri() + "/"
        self.base = base if base.endswith("/") else base + "/"
        self.cache = Path(cache or cache_root())
        self.dir = self.cache / "sites" / hashlib.sha256(self.base.encode()).hexdigest()[:16]
        self.offline = offline
        self.manifest = self._manifest()

    # --- manifest ---------------------------------------------------------
    def _manifest(self) -> dict:
        if self.local:
            return json.loads((self.local / "manifest.json").read_text(encoding="utf-8"))
        mpath, epath = self.dir / "manifest.json", self.dir / "manifest.etag"
        if not self.offline:
            headers = {"User-Agent": "glassine-docs"}
            if mpath.exists() and epath.exists():
                headers["If-None-Match"] = epath.read_text().strip()
            req = urllib.request.Request(self.base + "manifest.json", headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=20) as r:
                    data = r.read()
                    json.loads(data)
                    self.dir.mkdir(parents=True, exist_ok=True)
                    _atomic(mpath, data)
                    etag = r.headers.get("ETag")
                    if etag:
                        _atomic(epath, etag.encode())
                    else:
                        epath.unlink(missing_ok=True)
            except urllib.error.HTTPError as e:
                if e.code != 304:
                    if not mpath.exists():
                        raise FetchError(f"{self.base}manifest.json: HTTP {e.code}") from e
                    _note(f"manifest: HTTP {e.code}, using the cached copy")
            except (urllib.error.URLError, OSError, ValueError) as e:
                if not mpath.exists():
                    raise FetchError(f"{self.base}manifest.json: {e}") from e
                _note(f"offline ({getattr(e, 'reason', e)}), using the cached manifest")
        if not mpath.exists():
            raise FetchError("no cached manifest (run once online)")
        return json.loads(mpath.read_text(encoding="utf-8"))

    # --- files ------------------------------------------------------------
    def file(self, entry: dict) -> bytes:
        """The bytes of a manifest file entry ({file, sha256}), from cache or network."""
        name, sha = entry["file"], entry["sha256"]
        if self.local:
            data = (self.local / name).read_bytes()
            if hashlib.sha256(data).hexdigest() != sha:
                raise FetchError(f"{name}: sha256 mismatch")
            return data
        p = self.dir / "files" / name
        if p.exists():
            data = p.read_bytes()
            if hashlib.sha256(data).hexdigest() == sha:
                return data
        if self.offline:
            raise FetchError(f"{name} is not cached (offline)")
        req = urllib.request.Request(urljoin(self.base, name), headers={"User-Agent": "glassine-docs"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
        except (urllib.error.URLError, OSError) as e:
            raise FetchError(f"{name}: {e}") from e
        if hashlib.sha256(data).hexdigest() != sha:
            raise FetchError(f"{name}: sha256 mismatch")
        p.parent.mkdir(parents=True, exist_ok=True)
        _atomic(p, data)
        self._prune()
        return data

    def _prune(self) -> None:
        """Drop cached files the current manifest no longer names."""
        keep = {self.manifest["chunks"]["file"], self.manifest["bm25"]["file"]}
        for prof in self.manifest["profiles"].values():
            keep.add(prof["vectors"]["file"])
            if "file" in prof["model"]:
                keep.add(prof["model"]["file"])
        d = self.dir / "files"
        for p in d.iterdir() if d.exists() else []:
            if p.name not in keep:
                p.unlink(missing_ok=True)

    def url(self, rel: str) -> str:
        root = self.manifest.get("site_url") or urljoin(self.base, "../")
        return urljoin(root if root.endswith("/") else root + "/", rel)


def _atomic(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".part")
    tmp.write_bytes(data)
    os.replace(tmp, path)


class Searcher:
    def __init__(self, site: Site, profile: str = "transformer", models_mirror: str | None = None,
                 hf: bool = False):
        self.site = site
        m = site.manifest
        if profile not in m["profiles"]:
            raise FetchError(f"profile {profile!r} not published (have: {', '.join(m['profiles'])})")
        self.profile = profile
        self.prof = m["profiles"][profile]
        self.chunks = json.loads(site.file(m["chunks"]))["chunks"]
        self.index = json.loads(site.file(m["bm25"]))
        meta, arrays = store.unpack(site.file(self.prof["vectors"]))
        self.arrays = arrays
        if profile == "static":
            self.model = StaticModel(site.file(self.prof["model"]))
        else:
            self.model = self._transformer(models_mirror, hf)

    def _transformer(self, mirror, hf):
        from .transformer import TransformerModel
        info = self.prof["model"]
        lock = models.load_lock()
        locked = lock["models"]["transformer"]
        # Source files the lock knows let --hf rebuild a derived file (the int8 model).
        known = {(x["path"], x["sha256"]): x for x in locked["files"] + locked.get("derived", [])}
        entry = {"id": info["id"], "revision": info["revision"], "files": locked["files"]}
        mirror = mirror or models.default_mirror(lock)
        paths = {}
        for f in info["files"]:
            f = known.get((f["path"], f["sha256"]), f)
            paths[f["path"]] = models.fetch_file(self.site.cache / "models", entry, f, mirror, hf=hf, log=_note)
        return TransformerModel(paths[info["onnx"]], paths["tokenizer.json"],
                                query_prefix=info["query_prefix"], max_length=info["max_length"])

    def search(self, query: str, k: int = 5) -> tuple[list[Hit], float]:
        t0 = time.perf_counter()
        fz = self.site.manifest["fusion"]
        kw = bm25.scores(self.index, query)
        if self.profile == "static":
            has_meaning = bool(self.model.ids(query))
            vec = quant.static_scores(self.model.embed(query), self.arrays["bits"], self.arrays["q8"],
                                      self.arrays["scale"], self.prof["quantization"].get("candidates", 100))
        else:
            has_meaning = bool(query.strip())
            vec = quant.int8_scores(self.model.embed_query(query), self.arrays["q8"], self.arrays["scale"])
        lists = [[d for d, _ in kw]]
        if has_meaning:
            lists.append([d for d, _ in vec])
        fused = fusion.rrf(lists, fz["k"], fz["depth"])
        kwr = {d: r + 1 for r, (d, _) in enumerate(kw)}
        vr = {d: (r + 1, s) for r, (d, s) in enumerate(vec)} if has_meaning else {}
        hits = []
        for r, (d, s) in enumerate(fused[:k]):
            c = self.chunks[d]
            v = vr.get(d)
            hits.append(Hit(rank=r + 1, score=s, chunk=c, url=self.site.url(c["url"]),
                            bm25_rank=kwr.get(d), vector_rank=v[0] if v else None,
                            vector_score=v[1] if v else None))
        return hits, time.perf_counter() - t0
