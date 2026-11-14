# SPDX-License-Identifier: GPL-2.0-or-later
"""Other repositories' documentation (decision 0012): the sources file, fetching, rendering,
and what the build and the client make of several sources. Offline: local git repositories."""
import hashlib
import json
import re
import shutil

import pytest

from conftest import COMMIT
from glassine_docs import cli, client, sources
from helpers import build_site, git_repo, pages, sources_toml

C = "a" * 40


def entry(**kw):
    return {"repository": "GlassineOS/example", "folder": "docs", "commit": C, **kw}


# --- the sources file -----------------------------------------------------------
def test_load(tmp_path):
    f = sources_toml(tmp_path / "s.toml", [entry(), entry(repository="GlassineOS/other", folder="doc/")])
    got = sources.load(f, reserved=("docs",))
    assert [(s.name, s.folder, s.commit) for s in got] == [("example", "docs", C), ("other", "doc", C)]
    assert got[0].url == f"https://github.com/GlassineOS/example/blob/{C}/"
    assert got[0].manifest() == {"name": "example", "commit": C, "repository": "GlassineOS/example",
                                 "folder": "docs", "url": got[0].url}
    (tmp_path / "empty.toml").write_text("# no entries yet\n")
    assert sources.load(tmp_path / "empty.toml") == []


@pytest.mark.parametrize("bad, msg", [
    (entry(repository="someone/example"), "is not GlassineOS/<name>"),
    (entry(repository="GlassineOS/a/b"), "is not GlassineOS/<name>"),
    (entry(repository="GlassineOS/.."), "is not GlassineOS/<name>"),
    (entry(commit="abc123"), "full 40-character commit"),
    (entry(commit="A" * 40), "full 40-character commit"),
    (entry(folder="../x"), "inside the repository"),
    (entry(folder="/docs"), "inside the repository"),
    (entry(folder="."), "inside the repository"),
    (entry(folder=""), "inside the repository"),
    (entry(folder="docs/[x]"), "wildcards"),
    (entry(folder="docs/*"), "wildcards"),
    (entry(folder="#docs"), "a leading #"),
    (entry(repository="GlassineOS/docs"), "the site itself"),
    ({"repository": "GlassineOS/example", "commit": C}, "exactly the fields"),
    (entry(branch="main"), "exactly the fields"),
])
def test_load_rejects(tmp_path, bad, msg):
    f = sources_toml(tmp_path / "s.toml", [bad])
    with pytest.raises(sources.SourceError, match=msg):
        sources.load(f, reserved=("docs",))


def test_load_rejects_duplicates_and_other_keys(tmp_path):
    with pytest.raises(sources.SourceError, match="listed twice"):
        sources.load(sources_toml(tmp_path / "s.toml", [entry(), entry(folder="other")]))
    (tmp_path / "k.toml").write_text('sources = []\n')
    with pytest.raises(sources.SourceError, match="unknown key"):
        sources.load(tmp_path / "k.toml")
    (tmp_path / "x.toml").write_text('[[source]\n')
    with pytest.raises(sources.SourceError, match="x.toml"):
        sources.load(tmp_path / "x.toml")


# --- fetching -------------------------------------------------------------------
def src_for(repo, commit, folder="docs"):
    return sources.Source(name="example", commit=commit, repository="GlassineOS/example", folder=folder,
                          url=f"https://github.com/GlassineOS/example/blob/{commit}/")


def test_fetch_checks_out_only_the_folder_and_caches(tmp_path):
    commit = git_repo(tmp_path / "repo")
    url = f"file://{tmp_path}/repo"
    folder = sources.fetch(src_for(tmp_path / "repo", commit), tmp_path / "cache", url)
    tree = folder.parent
    assert tree.parent == tmp_path / "cache" and tree.name.startswith(f"example@{commit}-")
    assert (folder / "guides" / "flashing.md").is_file()
    assert not (tree / "README.md").exists() and not (tree / ".git").exists()
    shutil.rmtree(tmp_path / "repo")  # a pinned commit never changes: the cache is enough
    assert sources.fetch(src_for(None, commit), tmp_path / "cache", url) == folder
    assert [p.name for p in (tmp_path / "cache").iterdir()] == [tree.name]


def test_fetch_cache_follows_the_folder(tmp_path):
    """Only the folder is checked out: the same commit with another folder is fetched anew."""
    commit = git_repo(tmp_path / "repo", {"docs/a.md": "# A\n", "docs/sub/b.md": "# B\n",
                                          "manual/c.md": "# C\n"})
    url, cache = f"file://{tmp_path}/repo", tmp_path / "cache"
    def names(folder):
        s = src_for(None, commit, folder=folder)
        return [p["page"] for p in sources.pages(s, sources.fetch(s, cache, url))]
    assert names("docs/sub") == ["docs/sub/b.md"]
    assert names("docs") == ["docs/a.md", "docs/sub/b.md"]
    assert names("manual") == ["manual/c.md"]
    assert len(list(cache.iterdir())) == 3


def test_fetch_tolerates_another_build_storing_first(tmp_path, monkeypatch):
    commit = git_repo(tmp_path / "repo")
    url, cache = f"file://{tmp_path}/repo", tmp_path / "cache"
    s = src_for(None, commit)
    first = sources.fetch(s, cache, url)
    (first.parent / sources.DONE).unlink()  # as if this build had not seen the other's checkout
    real_rmtree = shutil.rmtree

    def keep_dest(path, *a, **kw):  # the other build's checkout lands between rmtree and replace
        if path != first.parent:
            real_rmtree(path, *a, **kw)
        else:
            (first.parent / sources.DONE).write_text(commit)
    monkeypatch.setattr(sources.shutil, "rmtree", keep_dest)
    assert sources.fetch(s, cache, url) == first


def test_fetch_failures_fail(tmp_path):
    commit = git_repo(tmp_path / "repo")
    url = f"file://{tmp_path}/repo"
    with pytest.raises(sources.SourceError, match="without credentials"):
        sources.fetch(src_for(None, commit), tmp_path / "c1", f"file://{tmp_path}/missing")
    with pytest.raises(sources.SourceError, match="without credentials"):
        sources.fetch(src_for(None, "b" * 40), tmp_path / "c2", url)
    with pytest.raises(sources.SourceError, match="no folder 'manual'"):
        sources.fetch(src_for(None, commit, folder="manual"), tmp_path / "c3", url)
    for c in ("c1", "c2", "c3"):  # nothing half-fetched is left to be taken for a checkout
        assert not [p for p in (tmp_path / c).iterdir()]


def test_fetch_uses_no_user_config_or_credentials(tmp_path, monkeypatch):
    """A user's git config (credential helpers, URL rewrites) and GIT_* settings are not
    used: here a rewrite would turn the unreachable URL into a working one."""
    commit = git_repo(tmp_path / "repo")
    home = tmp_path / "home"
    home.mkdir()
    (home / ".gitconfig").write_text(f'[url "file://{tmp_path}/repo"]\n\tinsteadOf = https://unreachable.invalid/x\n'
                                     "[credential]\n\thelper = store\n")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(home / ".gitconfig"))
    monkeypatch.setenv("GIT_CONFIG_PARAMETERS", f"'url.file://{tmp_path}/repo.insteadof'='https://unreachable.invalid/x'")
    with pytest.raises(sources.SourceError):
        sources.fetch(src_for(None, commit), tmp_path / "cache", "https://unreachable.invalid/x")
    env = sources._git_env(tmp_path)
    assert env["GIT_CONFIG_GLOBAL"] == "/dev/null" and env["GIT_TERMINAL_PROMPT"] == "0"
    assert env["HOME"] == str(tmp_path) and "GIT_CONFIG_PARAMETERS" not in env


# --- rendering ------------------------------------------------------------------
@pytest.mark.parametrize("text, slug", [
    # checked against GitHub's rendering of GlassineOS/search docs/format.md and
    # GlassineOS/docs pages
    ("manifest.json", "manifestjson"), ("The site's search worker", "the-sites-search-worker"),
    ("Vector and model files (.bin)", "vector-and-model-files-bin"),
    ("Search file format, version 1", "search-file-format-version-1"),
    ("What A/B needs", "what-ab-needs"),
    ("0012 — Where documentation lives", "0012--where-documentation-lives"),
    # github-slugger's rule (letters of any script stay; "_" stays)
    ("Über Größe", "über-größe"), ("snake_case", "snake_case"),
])
def test_github_slug(text, slug):
    assert sources.github_slug(text) == slug


def test_pages_render_with_github_anchors(tmp_path):
    commit = git_repo(tmp_path / "repo")
    s = src_for(None, commit)
    folder = sources.fetch(s, tmp_path / "cache", f"file://{tmp_path}/repo")
    got = sources.pages(s, folder)
    assert [p["page"] for p in got] == ["docs/guides/flashing.md", "docs/index.md"]  # no hidden file
    fl = got[0]
    assert fl["url"] == f"https://github.com/GlassineOS/example/blob/{commit}/docs/guides/flashing.md"
    assert '<h1 id="flashing">' in fl["html"]
    assert '<h2 id="write-the-image">' in fl["html"] and '<h2 id="write-the-image-1">' in fl["html"]
    assert "<code>example flash</code>" in fl["html"]


def test_page_urls_are_percent_encoded(tmp_path):
    folder = tmp_path / "tree" / "docs" / "sub"
    folder.mkdir(parents=True)
    (folder / "my page#1.md").write_text("# P\n\n## Über\n\ntext\n")
    s = src_for(None, C)
    (p,) = sources.pages(s, folder.parent)
    assert p["page"] == "docs/sub/my page#1.md"
    assert p["url"] == f"{s.url}docs/sub/my%20page%231.md"
    from glassine_docs.chunk import chunk_page
    cs = chunk_page(p["html"], source="example", commit=C, page=p["page"], title=p["title"], url=p["url"])
    assert cs[-1].url.split("#", 1) == [p["url"], "über"]  # the page is everything before the first "#"
    assert cs[-1].id == f"example@{C}:docs/sub/my page#1.md#über"


def test_empty_folder_fails_the_build(tmp_path, model_file):
    commit = git_repo(tmp_path / "repo", {"docs/notes.txt": "no Markdown here\n"})
    f = sources_toml(tmp_path / "s.toml", [entry(commit=commit)])
    with pytest.raises(sources.SourceError, match="no .md files in docs/"):
        build_site(tmp_path / "out", model_file, tmp_path / "emb", sources_file=f,
                   source_cache=tmp_path / "sources", fetch_url=f"file://{tmp_path}/repo")


def test_pages_skip_symlinks(tmp_path):
    (tmp_path / "outside.md").write_text("# Secret\n\nnot in the repository\n")
    folder = tmp_path / "tree" / "docs"
    folder.mkdir(parents=True)
    (folder / "a.md").write_text("# A\n")
    (folder / "link.md").symlink_to(tmp_path / "outside.md")
    assert [p["page"] for p in sources.pages(src_for(None, C), folder)] == ["docs/a.md"]


# --- the build and the client with several sources ------------------------------
@pytest.fixture
def multi(tmp_path, model_file):
    commit = git_repo(tmp_path / "repo")
    f = sources_toml(tmp_path / "search-sources.toml", [entry(commit=commit)])
    own = pages()
    own[0].src = "docs/index.md"  # the same page path as the other source's index
    out = tmp_path / "site-search"
    res = build_site(out, model_file, tmp_path / "emb", page_list=own, sources_file=f,
                     source_cache=tmp_path / "sources", fetch_url=f"file://{tmp_path}/repo")
    return out, res, commit


def test_build_with_sources(multi):
    out, res, commit = multi
    m = json.loads((out / "manifest.json").read_text())
    assert m["format"] == 2 and m["commit"] == COMMIT
    base = f"https://github.com/GlassineOS/example/blob/{commit}/"
    assert m["sources"] == [{"name": "docs", "commit": COMMIT},
                            {"name": "example", "commit": commit, "repository": "GlassineOS/example",
                             "folder": "docs", "url": base}]
    chunks = json.loads((out / m["chunks"]["file"]).read_text())["chunks"]
    by_id = {c["id"]: c for c in chunks}
    c = by_id[f"example@{commit}:docs/guides/flashing.md#write-the-image-1"]
    assert c["source"] == "example" and c["page"] == "docs/guides/flashing.md"
    assert c["url"] == base + "docs/guides/flashing.md#write-the-image-1"
    own = by_id[f"docs@{COMMIT}:guides/recovery.md#edl"]
    assert own["source"] == "docs" and own["url"] == "guides/recovery/#edl"
    assert len(chunks) == len(res.chunks) == len({c["id"] for c in chunks})


def test_related_pages_keep_sources_apart(multi):
    _, res, _ = multi
    keys = set(res.related)
    assert ("docs", "docs/index.md") in keys and ("example", "docs/index.md") in keys
    for key, rel in res.related.items():
        assert len(rel) == 2 and key not in [k for k, _ in rel]


def test_cli_with_sources(multi, capsys):
    out, _, commit = multi
    assert cli.main(["search", "example flash EDL", "--profile", "static", "--base", str(out),
                     "--json", "-k", "10"]) == 0
    r = json.loads(capsys.readouterr().out)
    hit = next(h for h in r["results"] if h["source"] == "example")
    assert hit["url"].startswith(f"https://github.com/GlassineOS/example/blob/{commit}/docs/")
    assert {h["source"] for h in r["results"]} == {"docs", "example"}
    assert cli.main(["search", "flash", "--profile", "static", "--base", str(out)]) == 0
    assert f"other sources example@{commit[:12]}" in capsys.readouterr().out


def test_failing_source_fails_the_build(tmp_path, model_file):
    f = sources_toml(tmp_path / "s.toml", [entry()])
    with pytest.raises(sources.SourceError, match="GlassineOS/example"):
        build_site(tmp_path / "out", model_file, tmp_path / "emb", sources_file=f,
                   source_cache=tmp_path / "sources", fetch_url=f"file://{tmp_path}/nothing")


def test_client_rejects_unknown_formats(site, tmp_path, capsys):
    out, _ = site
    for f in (1, 2):
        assert client.check_format({"format": f}, "x") == {"format": f}
    m = json.loads((out / "manifest.json").read_text())
    m["format"] = 3
    (out / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(client.FetchError, match="format 3"):
        client.Site(str(out))
    assert cli.main(["search", "x", "--profile", "static", "--base", str(out)]) == 2
    assert "upgrade glassine-docs" in capsys.readouterr().err


def test_format_1_site_still_reads(site, capsys):
    """The published site stays at format 1 until the docs CI's pin moves."""
    out, _ = site
    m = json.loads((out / "manifest.json").read_text())
    chunks_doc = json.loads((out / m["chunks"]["file"]).read_text())
    chunks_doc["format"] = 1
    for c in chunks_doc["chunks"]:
        del c["source"]
    data = json.dumps(chunks_doc).encode()
    (out / m["chunks"]["file"]).write_bytes(data)
    m.update(format=1)
    m["chunks"].update(sha256=hashlib.sha256(data).hexdigest(), size=len(data))
    (out / "manifest.json").write_text(json.dumps(m))
    assert cli.main(["search", "recovery", "--profile", "static", "--base", str(out), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["results"][0]["source"] == "docs"


# Where a Markdown renderer can read a page otherwise than GitHub does. The expected
# anchors follow GFM (CommonMark: "#55" without a space is text) and github-slugger.
TRICKY = """# Block-list hand-off

Intro text.

#55 also says what the bootloader does, which is text on GitHub, not a heading.

##55 x is text too

## Layout {#custom-id}

1) one
2) two

Setext heading
--------------

## Repeated

| a | b |
|---|---|
| 1 | 2 |

## Repeated

<div>raw <b>HTML</b> block</div>

## `code` and *emphasis* ~~gone~~

# A second h1
"""


def test_render_reads_markdown_as_github_does():
    html = sources.render(TRICKY)
    ids = re.findall(r'<h[1-6] id="([^"]*)"', html)
    assert ids == ["block-list-hand-off", "layout-custom-id", "setext-heading", "repeated", "repeated-1",
                   "code-and-emphasis-gone", "a-second-h1"]
    assert "<p>#55 also says" in html and "<p>##55 x is text too</p>" in html
    assert "<ol>" in html and "<table>" in html and "<s>gone</s>" in html
    from glassine_docs.chunk import chunk_page
    cs = chunk_page(html, source="x", commit=C, page="docs/p.md", title="p", url="u/")
    assert {c.title for c in cs} == {"Block-list hand-off"}  # the first h1, not the last
    assert "#55 also says" in cs[0].text
