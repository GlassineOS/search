# SPDX-License-Identifier: GPL-2.0-or-later
import hashlib
import json

import numpy as np

from conftest import COMMIT
from glassine_docs import cli, store
from glassine_docs.builder import EmbedCache
from glassine_docs.staticmodel import StaticModel
from helpers import build_site


def test_outputs_and_manifest(site):
    out, res = site
    m = json.loads((out / "manifest.json").read_text())
    assert m["format"] == 1 and m["commit"] == COMMIT and not m["dirty"]
    assert m["sources"] == [{"name": "docs", "commit": COMMIT}]
    names = {p.name for p in out.iterdir()}
    for entry in [m["chunks"], m["bm25"], m["profiles"]["static"]["vectors"], m["profiles"]["static"]["model"]]:
        data = (out / entry["file"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == entry["sha256"] and len(data) == entry["size"]
        assert entry["sha256"][:12] in entry["file"]  # content-hashed name
    assert set(m["web"].values()) <= names
    chunks = json.loads((out / m["chunks"]["file"]).read_text())["chunks"]
    assert chunks[0]["id"] == f"docs@{COMMIT}:index.md"
    assert any(c["id"] == f"docs@{COMMIT}:guides/recovery.md#edl" for c in chunks)
    meta, arr = store.unpack((out / m["profiles"]["static"]["vectors"]["file"]).read_bytes())
    assert arr["bits"].shape == (len(chunks), 8) and arr["q8"].shape == (len(chunks), 64)


def test_vectors_come_from_the_published_table(site, model_file):
    out, res = site
    m = json.loads((out / "manifest.json").read_text())
    published = StaticModel((out / m["profiles"]["static"]["model"]["file"]).read_bytes())
    _, arr = store.unpack((out / m["profiles"]["static"]["vectors"]["file"]).read_bytes())
    v = published.embed(res.chunks[1].embed_text())
    back = arr["q8"][1].astype(float) * float(arr["scale"][1])
    assert float(v @ back / np.linalg.norm(back)) > 0.999


def test_related_pages(site):
    _, res = site
    assert set(res.related) == {"index.md", "guides/recovery.md", "decisions/0006-data-encryption.md",
                                "architecture/updates.md", "guides/eink.md"}
    for page, rel in res.related.items():
        assert len(rel) == 2 and page not in [p for p, _ in rel]


def test_embed_cache_warm_build_is_identical(tmp_path, model_file):
    a = build_site(tmp_path / "a", model_file, tmp_path / "emb")
    b = build_site(tmp_path / "b", model_file, tmp_path / "emb")
    assert a.stats["cache_static"] == {"hits": 0, "misses": len(a.chunks)}
    assert b.stats["cache_static"] == {"hits": len(b.chunks), "misses": 0}
    assert a.files.keys() == b.files.keys()


def test_cache_keys_on_text(tmp_path):
    c = EmbedCache(tmp_path, "m")
    calls = []
    fn = lambda ts: (calls.extend(ts), [np.ones(4) * len(t) for t in ts])[1]  # noqa: E731
    c.embed(["a", "bb"], 4, fn)
    c.embed(["a", "ccc"], 4, fn)
    assert calls == ["a", "bb", "ccc"]


def test_cli_local_dir_json(site, capsys):
    out, _ = site
    assert cli.main(["search", "how do I enter recovery", "--profile", "static", "--base", str(out),
                     "--json", "-k", "3"]) == 0
    r = json.loads(capsys.readouterr().out)
    assert r["commit"] == COMMIT and r["profile"] == "static" and len(r["results"]) == 3
    top = r["results"][0]
    assert top["url"].startswith("https://docs.example.org/guides/recovery/")
    assert {"rank", "score", "id", "title", "heading", "text", "bm25_rank", "vector_rank"} <= set(top)


def test_cli_exact_terms(site, capsys):
    out, _ = site
    for q, want in [("mmcblk0p48", "#edl"), ("glassine-deploy rollback", "#failed-update"),
                    ("LUKS2", "#decision")]:
        assert cli.main(["search", q, "--profile", "static", "--base", str(out), "--json"]) == 0
        assert json.loads(capsys.readouterr().out)["results"][0]["url"].endswith(want)


def test_cli_text_output(site, capsys):
    out, _ = site
    assert cli.main(["search", "EDL", "--profile", "static", "--base", str(out), "-k", "2"]) == 0
    text = capsys.readouterr().out
    assert text.startswith("1. ") and "Recovery › EDL" in text and f"docs commit {COMMIT}" in text


def test_cli_missing_profile(site, capsys):
    out, _ = site
    assert cli.main(["search", "x", "--profile", "transformer", "--base", str(out)]) == 2
    assert "not published" in capsys.readouterr().err
