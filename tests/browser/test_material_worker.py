# SPDX-License-Identifier: GPL-2.0-or-later
"""The worker swap in a real browser, against the Material version the docs use.

Catches Material changing its search worker protocol or result rendering. Run with
``make check-browser`` (needs Playwright and its headless Chromium; GDS_BROWSER=1).

The site is built with a synthetic static model in which the word "rescue" means the
"Enter recovery" section, a word that appears on no page: only the vector side can
find it. With another repository among the sources, "lifeboat" means a section of it,
which Material's index doesn't have.
"""
import http.server
import json
import os
import subprocess
import sys
import threading
from functools import partial
from importlib.metadata import version

import numpy as np
import pytest

if not os.environ.get("GDS_BROWSER"):
    pytest.skip("browser test: make check-browser", allow_module_level=True)
playwright = pytest.importorskip("playwright.sync_api")

from glassine_docs import quant, store  # noqa: E402
from glassine_docs.staticmodel import StaticModel  # noqa: E402
from helpers import git_repo, sources_toml, synthetic_static_model  # noqa: E402

PAGES = {
    "index.md": "# Home\n\nGlassineOS runs on e-ink tablets.\n\n## Slots\n\n"
                "The tablet keeps Android in one slot and GlassineOS in the other.\n",
    "guides/recovery.md": "# Recovery\n\nThe recovery partition holds a small system for repairs.\n\n"
                          "## Enter recovery\n\nClose the cover, then press power. The recovery menu opens.\n\n"
                          "## Backups\n\nThe recovery writes backups of the system partition.\n",
    "guides/edl.md": "# EDL\n\nEmergency download mode writes partitions over USB.\n",
    "guides/display.md": "# Display\n\nThe e-ink panel refreshes in A2 mode for scrolling.\n",
}
MKDOCS_YML = """site_name: T
site_url: https://docs.example.org/
theme:
  name: material
  features: [search.suggest]
markdown_extensions:
  - toc:
      permalink: true
plugins:
  - search
  - glassine-search
"""


def build(root, model_file, emb, env=None):
    env = {**os.environ, "GLASSINE_SEARCH_STATIC_MODEL": str(model_file),
           "GLASSINE_SEARCH_PROFILES": "static", "GLASSINE_SEARCH_EMBED_CACHE": str(emb),
           "NO_MKDOCS_2_WARNING": "1", **(env or {})}
    subprocess.run([sys.executable, "-m", "mkdocs", "build", "--strict", "-q"], cwd=root, env=env, check=True)


def rescue_model(root, emb, words=(("rescue", "#enter-recovery"),), env=None) -> None:
    """Build with a random model, then point each word at the chunk whose URL ends so."""
    data = synthetic_static_model()
    (root / "model.bin").write_bytes(data)
    build(root, root / "model.bin", emb, env)
    site = root / "site" / "search"
    m = json.loads((site / "manifest.json").read_text())
    chunks = json.loads((site / m["chunks"]["file"]).read_text())["chunks"]
    sm = StaticModel(data)
    meta, arrays = store.unpack(data)
    table, scale = arrays["table"].copy(), arrays["scale"].copy()
    for word, end in words:
        target = next(c for c in chunks if c["url"].endswith(end))
        trail = " › ".join([target["title"]] + [p for p in target["path"] if p != target["title"]])
        vec = sm.embed(trail + "\n" + target["text"]) * 50  # outweighs any other word in a query
        tid = sm.tokenizer.vocab[word]
        table[tid], scale[tid] = quant.quantize_rows(vec)
    (root / "model.bin").write_bytes(store.pack(meta, {"scale": scale, "table": table}))
    build(root, root / "model.bin", emb, env)


class Handler(http.server.SimpleHTTPRequestHandler):
    release = threading.Event()

    def log_message(self, *a):
        pass

    def do_GET(self):
        if "static-model." in self.path:
            Handler.release.wait(20)  # hold the model: keyword results must show first
        super().do_GET()


def serve(tmp_path):
    Handler.release = threading.Event()
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), partial(Handler, directory=str(tmp_path / "site")))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


def write_site(tmp_path, yml=MKDOCS_YML):
    for name, text in PAGES.items():
        (tmp_path / "docs" / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / "docs" / name).write_text(text)
    (tmp_path / "mkdocs.yml").write_text(yml)


@pytest.fixture
def site(tmp_path):
    write_site(tmp_path)
    rescue_model(tmp_path, tmp_path / "emb")
    httpd = serve(tmp_path)
    yield f"http://127.0.0.1:{httpd.server_address[1]}/"
    Handler.release.set()
    httpd.shutdown()


OTHER = {
    "docs/index.md": "# Flasher\n\nThe flasher writes images.\n",
    "docs/guides/images.md": "# Images & slots\n\nHow images are written.\n\n"
                             "## Write the `<image>` — safely\n\nKeep the battery above half while writing.\n",
}


@pytest.fixture
def site_with_source(tmp_path):
    commit = git_repo(tmp_path / "repo", OTHER)
    sources_toml(tmp_path / "search-sources.toml",
                 [{"repository": "GlassineOS/flasher", "folder": "docs", "commit": commit}])
    write_site(tmp_path, MKDOCS_YML + "      sources_file: search-sources.toml\n")
    (tmp_path / "mkdocs.yml").write_text((tmp_path / "mkdocs.yml").read_text().replace(
        "  - glassine-search\n", "  - glassine-search:\n"))
    env = {"GLASSINE_SEARCH_FETCH_URL": f"file://{tmp_path}/repo"}
    rescue_model(tmp_path, tmp_path / "emb", [("rescue", "#enter-recovery"), ("lifeboat", "--safely")], env)
    Handler.release.set()  # the model is not held back here
    httpd = serve(tmp_path)
    Handler.release.set()
    yield f"http://127.0.0.1:{httpd.server_address[1]}/", commit
    httpd.shutdown()


def test_material_worker_swap(site):
    print("mkdocs-material", version("mkdocs-material"))
    with playwright.sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        requests = []
        page.on("request", lambda r: requests.append(r.url))
        page.on("worker", lambda w: requests.append("worker:" + w.url))
        page.goto(site)
        page.wait_for_load_state("networkidle")

        # our worker replaced Material's; nothing of the vector side loaded yet
        assert any(u.startswith("worker:") and "glassine-search." in u for u in requests), requests
        assert not [u for u in requests if "static-model." in u or "/search/static." in u or "/search/chunks." in u]

        box = page.locator('[data-md-component="search-query"]')
        box.click()
        page.keyboard.type("tablet rescue", delay=20)
        titles = page.locator(".md-search-result__list h1, .md-search-result__list h2")
        # keyword results while the model is held back: Home (tablet), no Enter recovery
        page.wait_for_function("document.querySelectorAll('.md-search-result__list li').length > 0")
        kw = titles.all_inner_texts()
        assert kw and "Enter recovery" not in kw
        assert any("static-model." in u for u in requests)  # the first query started the load

        Handler.release.set()
        page.wait_for_function("[...document.querySelectorAll('.md-search-result__list h2')]"
                               ".some(h => h.textContent.includes('Enter recovery'))", timeout=10000)
        assert titles.all_inner_texts()[0] == "Home"  # fused: the keyword hit stays on top

        # a query only the vectors can answer ranks its section first
        box.fill("")
        page.keyboard.type("rescue", delay=20)
        page.wait_for_function("(document.querySelector('.md-search-result__list h2') || {}).textContent"
                               " === 'Enter recovery'", timeout=10000)
        first = page.locator(".md-search-result__list li").first
        assert first.locator("h1").inner_text() == "Recovery"
        assert page.locator(".md-search-result__meta").inner_text() != "No matching documents"

        # Material's keyboard handling still works on the fused list
        page.keyboard.press("ArrowDown")
        focused = page.evaluate("document.activeElement.getAttribute('href') || ''")
        assert "guides/recovery/" in focused
        page.keyboard.press("ArrowUp")
        assert page.evaluate("document.activeElement.dataset.mdComponent") == "search-query"
        with page.expect_navigation():
            page.keyboard.press("Enter")
        assert page.url.split("?")[0].endswith("guides/recovery/#enter-recovery"), page.url
        browser.close()


def test_other_repository_results(site_with_source):
    """A section of another repository, which Material's index doesn't have, is listed
    under its page, escaped, and links to GitHub at the pinned commit."""
    url, commit = site_with_source
    with playwright.sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        errors = []
        page.on("console", lambda m: errors.append(m.text) if m.type in ("error", "warning") else None)
        page.goto(url)
        page.wait_for_load_state("networkidle")
        box = page.locator('[data-md-component="search-query"]')
        box.click()
        page.keyboard.type("lifeboat", delay=20)
        page.wait_for_function("(document.querySelector('.md-search-result__list h2') || {}).textContent"
                               " === 'Write the <image> — safely'", timeout=10000)
        first = page.locator(".md-search-result__list li").first
        assert first.locator("h1").inner_text() == "Images & slots"
        base = f"https://github.com/GlassineOS/flasher/blob/{commit}/docs/guides/images.md"
        hrefs = first.locator("a").evaluate_all("as => as.map(a => a.getAttribute('href'))")
        assert [h.split("?")[0] for h in hrefs] == [base, base + "#write-the-image--safely"], hrefs
        assert "battery above half" in first.locator("h2 + *").first.inner_text() or \
            "battery above half" in first.inner_text()
        assert not page.locator(".md-search-result__list image").count()  # text, not markup

        # the site's own sections still come from Material's index
        box.fill("")
        page.keyboard.type("rescue", delay=20)
        page.wait_for_function("(document.querySelector('.md-search-result__list h2') || {}).textContent"
                               " === 'Enter recovery'", timeout=10000)
        assert page.locator(".md-search-result__list li").first.locator("h1").inner_text() == "Recovery"
        assert not [e for e in errors if "semantic search" in e], errors
        browser.close()
