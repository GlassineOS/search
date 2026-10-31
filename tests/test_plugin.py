# SPDX-License-Identifier: GPL-2.0-or-later
"""A real MkDocs build of a small site with the plugin (static profile, synthetic model)."""
import json

import pytest

mkdocs = pytest.importorskip("mkdocs")
from mkdocs.commands.build import build  # noqa: E402
from mkdocs.config import load_config  # noqa: E402

PAGES = {
    "index.md": "# Home\n\nGlassineOS for e-ink tablets.\n\n## Slots\n\nSlot B boots `boot_b`.\n",
    "guides/recovery.md": "# Recovery\n\nWhen the system does not boot, use recovery.\n\n"
                          "## Enter recovery\n\nClose the cover and press power.\n",
    "guides/edl.md": "# EDL\n\nEmergency download mode writes partitions over USB.\n",
}


def test_mkdocs_build(tmp_path, model_file, monkeypatch):
    docs = tmp_path / "docs"
    for name, text in PAGES.items():
        (docs / name).parent.mkdir(parents=True, exist_ok=True)
        (docs / name).write_text(text)
    (tmp_path / "mkdocs.yml").write_text(
        "site_name: T\nsite_url: https://docs.example.org/\nuse_directory_urls: true\n"
        "markdown_extensions:\n  - toc:\n      permalink: true\n"
        "plugins:\n  - search\n  - glassine-search:\n      related_count: 1\n")
    monkeypatch.setenv("GLASSINE_SEARCH_STATIC_MODEL", str(model_file))
    monkeypatch.setenv("GLASSINE_SEARCH_PROFILES", "static")
    monkeypatch.setenv("GLASSINE_SEARCH_EMBED_CACHE", str(tmp_path / "emb"))
    cfg = load_config(str(tmp_path / "mkdocs.yml"), strict=True)
    build(cfg)
    site = tmp_path / "site"
    m = json.loads((site / "search" / "manifest.json").read_text())
    chunks = json.loads((site / "search" / m["chunks"]["file"]).read_text())["chunks"]
    ids = [c["id"].split(":", 1)[1] for c in chunks]
    assert "guides/recovery.md#enter-recovery" in ids and "index.md#slots" in ids
    assert all("¶" not in c["text"] for c in chunks)
    assert (site / "search" / "search_index.json").exists()  # the theme's own index stays
    page = (site / "guides" / "recovery" / "index.html").read_text()
    assert 'class="gds-related" data-search-exclude' in page
    assert f'var b="../../search/",u="{m["web"]["ui.js"]}"' in page
