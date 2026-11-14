# SPDX-License-Identifier: GPL-2.0-or-later
"""Other repositories' documentation folders: the sources file, fetching, rendering.

Decision 0012: the docs repository keeps a list of public repositories, each with its
documentation folder and a pinned commit, and the build indexes those folders too.

    [[source]]
    repository = "GlassineOS/example-repo"   # public, fetched without credentials
    folder = "docs"                          # relative to the repository root
    commit = "<full 40-character commit>"

Each repository is fetched with git at its commit, without any credentials (no global
or system git config, no credential helper, no netrc, no prompts), so a private or
missing repository fails the build instead of being skipped. Its Markdown files are
rendered as GitHub reads them (CommonMark with GFM tables and strikethrough, by
markdown-it-py) and get GitHub's heading anchors, since its pages link to their
rendering on GitHub at the pinned commit.
"""
from __future__ import annotations

import hashlib
import html
import os
import re
import shutil
import subprocess
import tempfile
import unicodedata
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from urllib.parse import quote

FETCH_URL = "https://github.com/{repository}.git"
LINK_URL = "https://github.com/{repository}/blob/{commit}/"
OWNER = "GlassineOS"
FIELDS = ("repository", "folder", "commit")
COMMIT_RE = re.compile(r"[0-9a-f]{40}")
NAME_RE = re.compile(r"[A-Za-z0-9._-]+")
GLOB = set("*?[]\\!")  # the folder is a sparse-checkout pattern: no wildcards
DONE = ".glassine-fetched"  # written last into a complete checkout


class SourceError(RuntimeError):
    pass


@dataclass
class Source:
    name: str             # the chunk ID prefix: the repository's name ("docs", "search")
    commit: str
    repository: str = ""  # "GlassineOS/<name>"; empty for the site itself
    folder: str = ""      # the documentation folder, relative to the repository root
    url: str = ""         # where its pages link: page paths are relative to this; empty = the site

    def manifest(self) -> dict:
        d = {"name": self.name, "commit": self.commit}
        for k in ("repository", "folder", "url"):
            if getattr(self, k):
                d[k] = getattr(self, k)
        return d


def _toml(path: Path) -> dict:
    try:
        import tomllib
    except ImportError:  # Python 3.10
        import tomli as tomllib
    try:
        return tomllib.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise SourceError(f"{path}: {e}") from e


def load(path: Path, link_url: str = LINK_URL, reserved: tuple[str, ...] = ()) -> list[Source]:
    """The sources file's entries, checked. ``reserved`` names are taken (the site's own)."""
    doc = _toml(path)
    extra = set(doc) - {"source"}
    if extra:
        raise SourceError(f"{path}: unknown key(s) {', '.join(sorted(extra))} (entries are [[source]])")
    entries = doc.get("source", [])
    if not isinstance(entries, list):
        raise SourceError(f"{path}: 'source' must be an array of tables ([[source]])")
    out, names = [], set(reserved)
    for n, e in enumerate(entries, 1):
        where = f"{path}: source {n}"
        if not isinstance(e, dict):
            raise SourceError(f"{where}: not a table")
        if set(e) != set(FIELDS):
            raise SourceError(f"{where}: needs exactly the fields {', '.join(FIELDS)}, has {', '.join(sorted(e))}")
        repo, folder, commit = (e[k] for k in FIELDS)
        if not all(isinstance(v, str) for v in (repo, folder, commit)):
            raise SourceError(f"{where}: fields must be strings")
        owner, _, name = repo.partition("/")
        if owner != OWNER or not NAME_RE.fullmatch(name) or name in (".", ".."):
            raise SourceError(f"{where}: repository {repo!r} is not {OWNER}/<name>")
        if not COMMIT_RE.fullmatch(commit):
            raise SourceError(f"{where}: commit {commit!r} is not a full 40-character commit")
        f = PurePosixPath(folder)
        if not folder or f.is_absolute() or ".." in f.parts or str(f) == "." or "\\" in folder:
            raise SourceError(f"{where}: folder {folder!r} must be a folder inside the repository")
        if GLOB & set(folder) or folder.startswith("#") or folder != folder.strip():
            raise SourceError(f"{where}: folder {folder!r}: wildcards (* ? [ ] \\ !), a leading # and "
                              "surrounding spaces are not supported")
        if name in names:
            raise SourceError(f"{where}: {name!r} is listed twice, or is the site itself")
        names.add(name)
        out.append(Source(name=name, commit=commit, repository=repo, folder=str(f),
                          url=link_url.format(repository=repo, commit=commit)))
    return out


# --- fetching ---------------------------------------------------------------
def _git_env(home: Path) -> dict:
    """git with no credentials: no user or system config, no helper, no netrc, no prompt."""
    env = {k: v for k, v in os.environ.items() if not k.startswith(("GIT_", "SSH_"))}
    env.update(HOME=str(home), XDG_CONFIG_HOME=str(home), GIT_CONFIG_NOSYSTEM="1",
               GIT_CONFIG_GLOBAL=os.devnull, GIT_TERMINAL_PROMPT="0", GCM_INTERACTIVE="never",
               LC_ALL="C")
    return env


def _git(args: list[str], cwd: Path, env: dict) -> str:
    cmd = ["git", "-c", "credential.helper=", "-c", "core.askPass=", "-c", "http.extraHeader=",
           "-c", "advice.detachedHead=false", *args]
    r = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)
    if r.returncode:
        raise SourceError(f"git {args[0]}: {(r.stderr or r.stdout).strip()}")
    return r.stdout.strip()


def fetch(src: Source, cache: Path, fetch_url: str = FETCH_URL) -> Path:
    """The source's documentation folder at its commit, checked out under ``cache``."""
    cache = Path(cache)
    # only the folder is checked out: an entry whose folder changes needs its own checkout
    folder_key = hashlib.sha256(src.folder.encode()).hexdigest()[:12]
    dest = cache / f"{src.name}@{src.commit}-{folder_key}"
    if (dest / DONE).is_file():
        return _folder(src, dest)
    cache.mkdir(parents=True, exist_ok=True)
    url = fetch_url.format(repository=src.repository, commit=src.commit)
    tmp = Path(tempfile.mkdtemp(dir=cache, prefix=f".{src.name}-"))
    try:
        tree, home = tmp / "tree", tmp / "home"
        tree.mkdir()
        home.mkdir()
        env = _git_env(home)
        try:
            _git(["init", "-q"], tree, env)
            # only the documentation folder is checked out
            _git(["config", "core.sparseCheckout", "true"], tree, env)
            (tree / ".git" / "info").mkdir(exist_ok=True)
            (tree / ".git" / "info" / "sparse-checkout").write_text(f"/{src.folder}/\n")
            _git(["fetch", "-q", "--depth", "1", "--no-tags", url, src.commit], tree, env)
            got = _git(["rev-parse", "FETCH_HEAD^{commit}"], tree, env)
            if got != src.commit:
                raise SourceError(f"fetched {got}, not {src.commit}")
            _git(["checkout", "-q", "--detach", "FETCH_HEAD"], tree, env)
        except SourceError as e:
            raise SourceError(f"{src.repository} at {src.commit}: {e} (sources are fetched without "
                              "credentials: a private or missing repository fails the build)") from e
        _folder(src, tree)
        shutil.rmtree(tree / ".git")
        (tree / DONE).write_text(src.commit + "\n")
        if (dest / DONE).is_file():  # another build sharing the cache finished first: keep its tree,
            return _folder(src, dest)  # which it may be reading
        shutil.rmtree(dest, ignore_errors=True)  # an unfinished leftover
        try:
            os.replace(tree, dest)
        except OSError as e:  # another build sharing the cache got there first
            if not (dest / DONE).is_file():
                raise SourceError(f"{src.repository} at {src.commit}: cannot store the checkout: {e}") from e
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return _folder(src, dest)


def _folder(src: Source, tree: Path) -> Path:
    d = tree / src.folder
    if d.is_symlink() or not d.is_dir():
        raise SourceError(f"{src.repository} at {src.commit}: no folder {src.folder!r}")
    return d


# --- rendering --------------------------------------------------------------
HEADING_RE = re.compile(r"<h([1-6])(?:\s[^>]*)?>(.*?)</h\1>", re.S)
TAG_RE = re.compile(r"<[^>]+>")


def github_slug(text: str) -> str:
    """GitHub's heading anchor: lower case, drop everything but letters, marks, numbers,
    "_", "-" and spaces, then spaces become hyphens (github-slugger's rule)."""
    keep = (ch for ch in text.lower()
            if ch in "- " or unicodedata.category(ch)[0] in "LMN" or unicodedata.category(ch) == "Pc")
    return "".join(keep).replace(" ", "-")


def github_anchors(body: str) -> str:
    """Give every heading GitHub's id; repeats get -1, -2, ... as on GitHub."""
    seen: dict[str, int] = {}

    def sub(m):
        slug = base = github_slug(html.unescape(TAG_RE.sub("", m.group(2))).strip())
        if base in seen:
            while slug in seen:
                seen[base] += 1
                slug = f"{base}-{seen[base]}"
        seen[slug] = 0
        return f'<h{m.group(1)} id="{html.escape(slug)}">{m.group(2)}</h{m.group(1)}>'

    return HEADING_RE.sub(sub, body)


def renderer():
    """Markdown as GitHub reads it: CommonMark (so "#55 x" is text, not a heading, and
    "{#id}" stays text) with GFM's tables and strikethrough; raw HTML is kept."""
    try:
        from markdown_it import MarkdownIt
    except ImportError as e:  # in the build extra
        raise SourceError("rendering other repositories' pages needs markdown-it-py "
                          "(pip install 'glassine-docs[build]')") from e
    return MarkdownIt("commonmark", {"html": True}).enable(["table", "strikethrough"])


def render(text: str) -> str:
    return github_anchors(renderer().render(text))


def pages(src: Source, folder: Path) -> list[dict]:
    """The folder's Markdown files rendered: page (path in the repository), title, url, html.
    Hidden files and folders and symbolic links are left out."""
    md = renderer()
    out = []
    for dirpath, dirnames, filenames in os.walk(folder):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        for fn in sorted(filenames):
            p = Path(dirpath) / fn
            if fn.startswith(".") or not fn.endswith(".md") or p.is_symlink() or not p.is_file():
                continue
            rel = p.relative_to(folder).as_posix()
            body = md.render(p.read_text(encoding="utf-8", errors="replace"))
            page = f"{src.folder}/{rel}"
            # a file name may hold spaces or "#": the link is the path percent-encoded
            out.append({"page": page, "title": PurePosixPath(rel).stem,
                        "url": src.url + quote(page, safe="/"), "html": github_anchors(body)})
    out.sort(key=lambda d: d["page"])
    return out
