# SPDX-License-Identifier: GPL-2.0-or-later
"""The worker swap in a real browser, against the Material version the docs use.

Catches Material changing its search worker protocol or result rendering. Run with
``make check-browser`` (needs Playwright and its headless Chromium; GDS_BROWSER=1).

The site is built with a synthetic static model in which the word "rescue" means the
"Enter recovery" section, a word that appears on no page: only the vector side can
find it.
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
from helpers import synthetic_static_model  # noqa: E402

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


def build(root, model_file, emb):
    env = {**os.environ, "GLASSINE_SEARCH_STATIC_MODEL": str(model_file),
           "GLASSINE_SEARCH_PROFILES": "static", "GLASSINE_SEARCH_EMBED_CACHE": str(emb),
           "NO_MKDOCS_2_WARNING": "1"}
    subprocess.run([sys.executable, "-m", "mkdocs", "build", "--strict", "-q"], cwd=root, env=env, check=True)


def rescue_model(root, emb) -> None:
    """Build with a random model, then point "rescue" at the Enter recovery chunk."""
    data = synthetic_static_model()
    (root / "model.bin").write_bytes(data)
    build(root, root / "model.bin", emb)
    site = root / "site" / "search"
    m = json.loads((site / "manifest.json").read_text())
    chunks = json.loads((site / m["chunks"]["file"]).read_text())["chunks"]
    target = next(c for c in chunks if c["url"].endswith("#enter-recovery"))
    sm = StaticModel(data)
    trail = " › ".join([target["title"]] + [p for p in target["path"] if p != target["title"]])
    vec = sm.embed(trail + "\n" + target["text"]) * 50  # outweighs any other word in a query
    meta, arrays = store.unpack(data)
    table, scale = arrays["table"].copy(), arrays["scale"].copy()
    tid = sm.tokenizer.vocab["rescue"]
    table[tid], scale[tid] = quant.quantize_rows(vec)
    (root / "model.bin").write_bytes(store.pack(meta, {"scale": scale, "table": table}))
    build(root, root / "model.bin", emb)


class Handler(http.server.SimpleHTTPRequestHandler):
    release = threading.Event()

    def log_message(self, *a):
        pass

    def do_GET(self):
        if "static-model." in self.path:
            Handler.release.wait(20)  # hold the model: keyword results must show first
        super().do_GET()


@pytest.fixture
def site(tmp_path):
    for name, text in PAGES.items():
        (tmp_path / "docs" / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / "docs" / name).write_text(text)
    (tmp_path / "mkdocs.yml").write_text(MKDOCS_YML)
    rescue_model(tmp_path, tmp_path / "emb")
    Handler.release = threading.Event()
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), partial(Handler, directory=str(tmp_path / "site")))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}/"
    Handler.release.set()
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
