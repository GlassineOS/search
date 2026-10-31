# SPDX-License-Identifier: GPL-2.0-or-later
"""Build the search files from rendered pages (see docs/format.md).

The MkDocs plugin calls ``Builder.build`` once all pages are rendered and
``Builder.write`` after the site is written. Without MkDocs, ``build`` takes
any list of ``Page`` records.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import FORMAT_VERSION, __version__, bm25, chunk, fusion, models, quant, store
from .staticmodel import StaticModel, convert

WEB_DIR = Path(__file__).with_name("web")
WEB_FILES = ["text.js", "score.js"]
# The worker replaces Material's search worker and sits next to it (Material's worker
# loads its language files relative to the worker's own URL).
WORKER_DIR = "assets/javascripts/workers"
PROFILES = ("static", "transformer")
STATIC_CANDIDATES = 100


@dataclass
class Page:
    src: str      # source path, e.g. "guides/recovery.md"
    title: str
    url: str      # site-relative URL, e.g. "guides/recovery/"
    html: str


@dataclass
class Options:
    profiles: tuple[str, ...] = PROFILES
    related_profile: str = "static"
    related_count: int = 5
    model_cache: Path = Path(".cache/glassine-search/models")
    embed_cache: Path = Path(".cache/glassine-search/embeddings")
    mirror: str | None = None
    hf_fallback: bool = False
    source: str = "docs"
    site_url: str = ""
    repo_dir: Path = Path(".")
    max_words: int = chunk.MAX_WORDS
    static_model_file: Path | None = None  # a ready static-model.bin (tests, offline dev)


@dataclass
class Result:
    chunks: list
    files: dict[str, bytes] = field(default_factory=dict)
    site_files: dict[str, bytes] = field(default_factory=dict)  # paths from the site root
    manifest: dict = field(default_factory=dict)
    related: dict[str, list[tuple[str, float]]] = field(default_factory=dict)
    stats: dict = field(default_factory=dict)


def log(msg: str) -> None:
    print(f"glassine-search: {msg}", file=sys.stderr)


def docs_commit(repo_dir: Path) -> tuple[str, bool]:
    """The docs commit: GITHUB_SHA in CI (a clean checkout), else git HEAD and whether
    the work tree has uncommitted changes."""
    sha = os.environ.get("GITHUB_SHA")
    if sha:
        return sha, False
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_dir, check=True,
                             capture_output=True, text=True).stdout.strip()
        st = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=repo_dir,
                            check=True, capture_output=True, text=True).stdout
        return sha, bool(st.strip())
    except (OSError, subprocess.CalledProcessError):
        return "unknown", False


def hashed_name(name: str, data: bytes) -> str:
    stem, dot, ext = name.rpartition(".")
    return f"{stem}.{hashlib.sha256(data).hexdigest()[:12]}.{ext}"


def _file_entry(name: str, data: bytes) -> dict:
    return {"file": name, "sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}


class EmbedCache:
    """One file per (model, text) pair: float32 vector bytes, keyed by sha256."""

    def __init__(self, root: Path, model_key: str):
        self.dir = Path(root) / model_key
        self.hits = self.misses = 0

    def _path(self, text: str) -> Path:
        k = hashlib.sha256(text.encode()).hexdigest()
        return self.dir / k[:2] / (k + ".f32")

    def get(self, text: str, dims: int):
        p = self._path(text)
        try:
            v = np.frombuffer(p.read_bytes(), dtype="<f4")
        except OSError:
            return None
        return v if v.size == dims else None

    def put(self, text: str, vec: np.ndarray) -> None:
        p = self._path(text)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_bytes(np.asarray(vec, dtype="<f4").tobytes())
        os.replace(tmp, p)

    def embed(self, texts: list[str], dims: int, fn) -> np.ndarray:
        out: list = [None] * len(texts)
        todo = []
        for i, t in enumerate(texts):
            v = self.get(t, dims)
            if v is None:
                todo.append(i)
            else:
                out[i] = v.astype(np.float64)
        self.hits += len(texts) - len(todo)
        self.misses += len(todo)
        if todo:
            vecs = fn([texts[i] for i in todo])
            for i, v in zip(todo, vecs):
                self.put(texts[i], v)
                # read back through float32, so a cold and a warm build give the same bytes
                out[i] = np.asarray(v, dtype="<f4").astype(np.float64)
        return np.stack(out) if out else np.zeros((0, dims))


class Builder:
    def __init__(self, opts: Options):
        self.opts = opts
        self.lock = models.load_lock()
        self._static: tuple[StaticModel, bytes] | None = None
        self._transformer = None

    # --- models ---------------------------------------------------------
    def static_model(self) -> tuple[StaticModel, bytes]:
        if self._static is None and self.opts.static_model_file:
            data = Path(self.opts.static_model_file).read_bytes()
            self._static = (StaticModel(data), data)
        if self._static is None:
            entry = self.lock["models"]["static"]
            paths = models.fetch_profile(self.opts.model_cache, "static", self.opts.mirror,
                                         self.opts.hf_fallback, lock=self.lock)
            data = convert(paths["model.safetensors"].parent, entry)
            self._static = (StaticModel(data), data)
        return self._static

    def transformer_model(self):
        if self._transformer is None:
            from .transformer import TransformerModel
            entry = self.lock["models"]["transformer"]
            need = [entry["onnx"], "tokenizer.json"]
            paths = models.fetch_profile(self.opts.model_cache, "transformer", self.opts.mirror,
                                         self.opts.hf_fallback, only=need, lock=self.lock)
            self._transformer = TransformerModel(paths[entry["onnx"]], paths["tokenizer.json"],
                                                 query_prefix=entry["query_prefix"],
                                                 max_length=entry["max_length"])
        return self._transformer

    # --- build ----------------------------------------------------------
    def build(self, pages: list[Page]) -> Result:
        o = self.opts
        t0 = time.monotonic()
        commit, dirty = docs_commit(o.repo_dir)
        cid = commit + ("-dirty" if dirty else "")
        chunks = []
        for p in pages:
            chunks.extend(chunk.chunk_page(p.html, source=o.source, commit=cid, page=p.src,
                                           title=p.title, url=p.url, max_words=o.max_words))
        texts = [c.embed_text() for c in chunks]
        res = Result(chunks=chunks)
        res.stats["chunks"] = len(chunks)
        files = {}

        chunks_doc = {"format": FORMAT_VERSION, "commit": cid, "chunks": [
            {"id": c.id, "page": c.page, "title": c.title, "heading": c.heading,
             "path": c.path, "url": c.url, "text": c.text} for c in chunks]}
        cdata = json.dumps(chunks_doc, ensure_ascii=False, separators=(",", ":")).encode()
        cname = hashed_name("chunks.json", cdata)
        files[cname] = cdata

        idx = bm25.build(texts)
        bdata = json.dumps(idx, ensure_ascii=False, separators=(",", ":")).encode()
        bname = hashed_name("bm25.json", bdata)
        files[bname] = bdata

        manifest = {
            "format": FORMAT_VERSION, "generator": f"glassine-docs {__version__}",
            "site_url": o.site_url, "commit": cid, "dirty": dirty,
            "sources": [{"name": o.source, "commit": cid}],
            "chunks": {**_file_entry(cname, cdata), "count": len(chunks)},
            "bm25": {**_file_entry(bname, bdata), "k1": idx["k1"], "b": idx["b"]},
            "fusion": {"method": "rrf", "k": fusion.K, "depth": fusion.DEPTH},
            "profiles": {},
        }
        vectors = {}
        for prof in o.profiles:
            t1 = time.monotonic()
            if prof == "static":
                model, mdata = self.static_model()
                mname = hashed_name("static-model.bin", mdata)
                files[mname] = mdata
                mkey = "static-" + hashlib.sha256(mdata).hexdigest()[:16]
                cache = EmbedCache(o.embed_cache, mkey)
                vecs = cache.embed(texts, model.dims, model.embed_many)
                q8, scale = quant.quantize_rows(vecs)
                vdata = store.pack({"kind": "vectors", "profile": "static", "dims": model.dims,
                                    "count": len(chunks)},
                                   {"bits": quant.sign_bits(vecs), "q8": q8, "scale": scale})
                model_info = {"id": model.meta["model"], "revision": model.meta["revision"],
                              "license": self.lock["models"]["static"]["license"], **_file_entry(mname, mdata)}
                quantization = {"vectors": "binary sign bits + int8 per-vector scale (rescoring)",
                                "model": "int8 per-row scale", "candidates": STATIC_CANDIDATES}
                dims = model.dims
            elif prof == "transformer":
                tm = self.transformer_model()
                entry = self.lock["models"]["transformer"]
                f = {x["path"]: x for x in entry["files"] + entry.get("derived", [])}
                onnx = f[entry["onnx"]]
                mkey = "transformer-" + onnx["sha256"][:16]
                cache = EmbedCache(o.embed_cache, mkey)
                dims = entry["dims"]
                vecs = cache.embed(texts, dims, tm.embed_many)
                q8, scale = quant.quantize_rows(vecs)
                vdata = store.pack({"kind": "vectors", "profile": "transformer", "dims": dims,
                                    "count": len(chunks)}, {"q8": q8, "scale": scale})
                model_info = {"id": entry["id"], "revision": entry["revision"], "license": entry["license"],
                              "sha256": onnx["sha256"],
                              "files": [{k: f[p][k] for k in ("path", "asset", "sha256", "size")}
                                        for p in (entry["onnx"], "tokenizer.json")],
                              "onnx": entry["onnx"], "query_prefix": entry["query_prefix"],
                              "pooling": entry["pooling"], "max_length": entry["max_length"]}
                quantization = {"vectors": "int8 per-vector scale",
                                "model": "int8 dynamic" if "int8" in entry["onnx"] else "float32"}
            else:
                raise ValueError(f"unknown profile {prof}")
            vname = hashed_name(f"{prof}.bin", vdata)
            files[vname] = vdata
            vectors[prof] = vecs
            manifest["profiles"][prof] = {"model": model_info, "dims": dims,
                                          "quantization": quantization,
                                          "vectors": _file_entry(vname, vdata)}
            res.stats[f"embed_{prof}_s"] = round(time.monotonic() - t1, 3)
            res.stats[f"cache_{prof}"] = {"hits": cache.hits, "misses": cache.misses}

        web = {}
        for name in WEB_FILES:
            data = (WEB_DIR / name).read_bytes()
            hn = hashed_name(name, data)
            files[hn] = data
            web[name] = hn
        wdata = (WEB_DIR / "worker.js").read_bytes()
        wpath = f"{WORKER_DIR}/{hashed_name('glassine-search.js', wdata)}"
        res.site_files[wpath] = wdata
        web["worker"] = wpath
        manifest["web"] = web

        if o.related_profile and o.related_profile in vectors:
            res.related = related_pages(chunks, vectors[o.related_profile], o.related_count)
        manifest["related"] = {"profile": o.related_profile, "count": o.related_count}

        res.files = files
        res.manifest = manifest
        res.stats["total_s"] = round(time.monotonic() - t0, 3)
        return res

    @staticmethod
    def write(res: Result, out_dir: Path) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, data in res.files.items():
            (out_dir / name).write_bytes(data)
        for rel, data in res.site_files.items():  # the search directory sits in the site root
            p = out_dir.parent / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)
        (out_dir / "manifest.json").write_text(json.dumps(res.manifest, indent=1, ensure_ascii=False) + "\n",
                                               encoding="utf-8")


def related_pages(chunks, vecs: np.ndarray, count: int) -> dict[str, list[tuple[str, float]]]:
    """Per page, the closest pages. Page similarity: each chunk's best match in the other
    page, averaged, in both directions (a mean page vector favours long, general pages)."""
    pages: dict[str, list[int]] = {}
    for i, c in enumerate(chunks):
        pages.setdefault(c.page, []).append(i)
    names = list(pages)
    if not names:
        return {}
    v = quant.normalize(np.asarray(vecs, dtype=np.float64))
    sim = v @ v.T
    n = len(names)
    ps = np.zeros((n, n))
    for a, na in enumerate(names):
        for b, nb in enumerate(names):
            sub = sim[np.ix_(pages[na], pages[nb])]
            ps[a, b] = (sub.max(axis=1).mean() + sub.max(axis=0).mean()) / 2
    out = {}
    for a, name in enumerate(names):
        order = [b for b in quant.rank(ps[a]) if b != a][:count]
        out[name] = [(names[b], float(ps[a, b])) for b in order]
    return out


def sizes(out_dir: Path) -> list[tuple[str, int, int]]:
    rows = []
    for p in sorted(Path(out_dir).iterdir()):
        if p.is_file():
            d = p.read_bytes()
            rows.append((p.name, len(d), len(gzip.compress(d, 9))))
    return rows
