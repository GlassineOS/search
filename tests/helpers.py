# SPDX-License-Identifier: GPL-2.0-or-later
"""Offline test material: a synthetic static model on the real vocabulary, sample pages."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from glassine_docs import builder, quant, store

FIX = Path(__file__).with_name("fixtures")
DIMS = 64


def tokenizer_config(profile: str = "static") -> dict:
    return json.loads((FIX / f"{profile}-tokenizer.json").read_text(encoding="utf-8"))


def synthetic_static_model(seed: int = 7, dims: int = DIMS) -> bytes:
    """A static-model.bin with random vectors on potion's vocabulary.

    Rows of words that share a stem get correlated vectors, so meaning-ish
    neighbours exist; that is all the tests need.
    """
    tok = tokenizer_config("static")
    rng = np.random.default_rng(seed)
    emb = rng.normal(size=(len(tok["vocab"]), dims))
    q, scale = quant.quantize_rows(emb)
    meta = {"kind": "static-model", "model": "test/synthetic", "revision": "0", "source_sha256": {},
            "dims": dims, "normalize": True, "max_length": 512,
            "median_token_length": int(np.median([len(t) for t in tok["vocab"]])), "tokenizer": tok}
    return store.pack(meta, {"scale": scale, "table": q})


PAGES = [
    ("index.md", "Home", "", """<h1 id="home">Home</h1><p>GlassineOS is a Linux distribution for e-ink
tablets. It boots from <code>boot_b</code> and keeps Android in slot A.</p>"""),
    ("guides/recovery.md", "Recovery", "guides/recovery/", """<h1 id="recovery">Recovery</h1>
<p>When the system does not boot, the bootloader falls back to the recovery partition.</p>
<h2 id="enter-recovery">Enter recovery<a class="headerlink" href="#enter-recovery">¶</a></h2>
<p>Close the cover and press power. The recovery menu opens.</p>
<pre><code>fastboot oem reboot-recovery
</code></pre>
<h2 id="edl">EDL<a class="headerlink" href="#edl">¶</a></h2><p>Qualcomm's emergency download mode
writes partitions such as <code>mmcblk0p48</code> over USB.</p>
<div class="mermaid">graph LR; a-->b</div>"""),
    ("decisions/0006-data-encryption.md", "0006 — User data", "decisions/0006-data-encryption/",
     """<h1 id="0006-user-data">0006 — User data</h1><h2 id="context">Context</h2>
<p>User data must survive what Android does to its own data.</p><h2 id="decision">Decision</h2>
<p>User data lives in a LUKS2 volume, unlocked at login with the user's password.</p>
<table><tr><th>Key</th><th>Where</th></tr><tr><td>recovery key</td><td>printed</td></tr></table>"""),
    ("architecture/updates.md", "Updates", "architecture/updates/", """<h1 id="updates">Updates</h1>
<h2 id="failed-update">When an update fails</h2><p>A failed update leaves the previous deployment
in place; the next boot uses it. The <code>glassine-deploy rollback</code> command selects it by hand.</p>
<h3 id="tries">Tries</h3><p>Two tries, then the old deployment.</p>"""),
    ("guides/eink.md", "E-ink", "guides/eink/", """<h1 id="e-ink">E-ink</h1>
<h2 id="refresh">Refresh modes</h2><p>The panel refreshes fast in A2 mode and slowly with full greys
in GC16 mode. <code>glassine-epdd</code> picks the mode.</p>"""),
]


def pages() -> list[builder.Page]:
    return [builder.Page(src=s, title=t, url=u, html=h) for s, t, u, h in PAGES]


def build_site(out_dir: Path, model_file: Path, embed_cache: Path, page_list=None, profiles=("static",)):
    opts = builder.Options(profiles=profiles, related_profile="static", related_count=2,
                           embed_cache=embed_cache, static_model_file=model_file,
                           site_url="https://docs.example.org/", repo_dir=out_dir)
    res = builder.Builder(opts).build(page_list or pages())
    builder.Builder.write(res, out_dir)
    return res
