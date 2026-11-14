# SPDX-License-Identifier: GPL-2.0-or-later
"""A real MkDocs build of a small site with the plugin (static profile, synthetic model)."""
import json

import pytest

from glassine_docs.plugin import repo_path
from helpers import SOURCE_PAGES, git, git_repo, sources_toml

mkdocs = pytest.importorskip("mkdocs")
pytest.importorskip("material")
from mkdocs.commands.build import build  # noqa: E402
from mkdocs.config import load_config  # noqa: E402

PAGES = {
    "index.md": "# Home\n\nGlassineOS for e-ink tablets.\n\n## Slots\n\nSlot B boots `boot_b`.\n",
    "guides/recovery.md": "# Recovery\n\nWhen the system does not boot, use recovery.\n\n"
                          "## Enter recovery\n\nClose the cover and press power.\n",
    "guides/edl.md": "# EDL\n\nEmergency download mode writes partitions over USB.\n",
}


def mkdocs_site(tmp_path, model_file, monkeypatch, extra="", sub=""):
    """A site in its own git repository (as the docs repository is); mkdocs.yml in ``sub``."""
    git(tmp_path, "init", "-q")
    tmp_path = tmp_path / sub
    docs = tmp_path / "docs"
    for name, text in PAGES.items():
        (docs / name).parent.mkdir(parents=True, exist_ok=True)
        (docs / name).write_text(text)
    (tmp_path / "mkdocs.yml").write_text(
        "site_name: T\nsite_url: https://docs.example.org/\nuse_directory_urls: true\n"
        "theme:\n  name: material\n"
        "markdown_extensions:\n  - toc:\n      permalink: true\n"
        "plugins:\n  - search\n  - glassine-search:\n      related_count: 1\n" + extra)
    monkeypatch.setenv("GLASSINE_SEARCH_STATIC_MODEL", str(model_file))
    monkeypatch.setenv("GLASSINE_SEARCH_PROFILES", "static")
    monkeypatch.setenv("GLASSINE_SEARCH_EMBED_CACHE", str(tmp_path / "emb"))
    return load_config(str(tmp_path / "mkdocs.yml"), strict=True)


def test_mkdocs_build(tmp_path, model_file, monkeypatch):
    build(mkdocs_site(tmp_path, model_file, monkeypatch))
    site = tmp_path / "site"
    m = json.loads((site / "search" / "manifest.json").read_text())
    assert m["sources"] == [{"name": "docs", "commit": m["commit"], "folder": "docs"}]
    chunks = json.loads((site / "search" / m["chunks"]["file"]).read_text())["chunks"]
    ids = [c["id"].split(":", 1)[1] for c in chunks]
    # <page> is the path in the repository: the docs folder comes first
    assert "docs/guides/recovery.md#enter-recovery" in ids and "docs/index.md#slots" in ids
    assert {c["page"] for c in chunks} == {f"docs/{p}" for p in PAGES}
    assert all("¶" not in c["text"] for c in chunks)
    assert (site / "search" / "search_index.json").exists()  # the theme's own index stays
    page = (site / "guides" / "recovery" / "index.html").read_text()
    assert 'class="gds-related" data-search-exclude' in page
    cfg = json.loads(page.split('<script id="__config" type="application/json">')[1].split("</script>")[0])
    worker = m["web"]["worker"]
    assert cfg["search"].startswith(f"../../{worker}?material=search.")
    assert cfg["search"].endswith("&search=../../../search/")
    assert (site / worker).is_file()
    material = cfg["search"].split("material=")[1].split("&")[0]
    assert (site / "assets/javascripts/workers" / material).is_file()


def test_mkdocs_build_with_sources(tmp_path, model_file, monkeypatch):
    commit = git_repo(tmp_path / "repo", {**SOURCE_PAGES, "docs/recovery.md":
                                          "# Recovery\n\nWhen the system does not boot, use recovery.\n"})
    sources_toml(tmp_path / "search-sources.toml", [
        {"repository": "GlassineOS/example", "folder": "docs", "commit": commit}])
    monkeypatch.setenv("GLASSINE_SEARCH_FETCH_URL", f"file://{tmp_path}/repo")
    build(mkdocs_site(tmp_path, model_file, monkeypatch, "      sources_file: search-sources.toml\n"))
    site = tmp_path / "site"
    m = json.loads((site / "search" / "manifest.json").read_text())
    assert [s["name"] for s in m["sources"]] == ["docs", "example"]
    chunks = json.loads((site / "search" / m["chunks"]["file"]).read_text())["chunks"]
    assert f"example@{commit}:docs/recovery.md" in [c["id"] for c in chunks]
    assert list((tmp_path / ".cache/glassine-search/sources").glob(f"example@{commit}-*/docs"))
    # the closest page to Recovery is the other repository's: an absolute link
    page = (site / "guides" / "recovery" / "index.html").read_text()
    related = page.split('class="gds-related"')[1].split("</nav>")[0]
    assert f'href="https://github.com/GlassineOS/example/blob/{commit}/docs/recovery.md"' in related


def test_mkdocs_build_fails_on_a_source_it_cannot_fetch(tmp_path, model_file, monkeypatch, caplog):
    from mkdocs.exceptions import Abort
    sources_toml(tmp_path / "search-sources.toml", [
        {"repository": "GlassineOS/private", "folder": "docs", "commit": "c" * 40}])
    monkeypatch.setenv("GLASSINE_SEARCH_FETCH_URL", f"file://{tmp_path}/nothing")
    cfg = mkdocs_site(tmp_path, model_file, monkeypatch, "      sources_file: search-sources.toml\n")
    with pytest.raises(Abort):  # MkDocs logs the plugin's error and aborts the build
        build(cfg)
    assert "glassine-search: GlassineOS/private at " in caplog.text
    assert "fetched without credentials" in caplog.text
    assert not (tmp_path / "site" / "search").exists()


def test_page_is_the_path_from_the_top_of_the_repository(tmp_path, model_file, monkeypatch):
    """mkdocs.yml need not sit at the top: <page> still starts at the repository's top."""
    build(mkdocs_site(tmp_path, model_file, monkeypatch, sub="site"))
    out = tmp_path / "site" / "site"
    m = json.loads((out / "search" / "manifest.json").read_text())
    assert m["sources"][0]["folder"] == "site/docs"
    chunks = json.loads((out / "search" / m["chunks"]["file"]).read_text())["chunks"]
    assert {c["page"] for c in chunks} == {f"site/docs/{p}" for p in PAGES}


def test_repo_path(tmp_path):
    git(tmp_path, "init", "-q")
    (tmp_path / "a" / "docs").mkdir(parents=True)
    assert repo_path(tmp_path / "a" / "docs", tmp_path / "a") == "a/docs"
    assert repo_path(tmp_path, tmp_path) == ""
    assert repo_path(tmp_path.parent / "elsewhere", tmp_path) == ""
