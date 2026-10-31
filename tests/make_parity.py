#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
"""Write what the Python side computes, for the node parity tests (tests/js/parity.test.js).

    tests/make_parity.py OUT_DIR [STATIC_MODEL_BIN]

Builds the sample site (static profile) into OUT_DIR/site with the synthetic
model, or with a real static-model.bin when given, and writes OUT_DIR/expected.json.
"""
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE)]
os.environ.setdefault("GITHUB_SHA", "0" * 40)

from glassine_docs import bm25, fusion, quant, store  # noqa: E402
from glassine_docs.staticmodel import StaticModel  # noqa: E402
from glassine_docs.text import bm25_terms  # noqa: E402
from helpers import FIX, build_site, synthetic_static_model  # noqa: E402

QUERIES = [
    "how do I get back into recovery if the system doesn't boot", "why is user data encrypted with LUKS",
    "what happens when an update fails", "e-ink refresh speed", "EDL", "boot_b", "mmcblk0p48",
    "glassine-epdd A2", "Über Größe", "", "zzzzqqq", "[MASK] recovery", "中文 recovery",
]


def main(out: str, model: str | None = None) -> None:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    mfile = Path(model) if model else out / "model.bin"
    if not model:
        mfile.write_bytes(synthetic_static_model())
    site = out / "site"
    build_site(site, mfile, out / "emb")
    m = json.loads((site / "manifest.json").read_text())
    sm = StaticModel((site / m["profiles"]["static"]["model"]["file"]).read_bytes())
    _, arr = store.unpack((site / m["profiles"]["static"]["vectors"]["file"]).read_bytes())
    idx = json.loads((site / m["bm25"]["file"]).read_text())
    texts = json.loads((FIX / "token-texts.json").read_text(encoding="utf-8"))
    rows = []
    for q in QUERIES + texts:
        v = sm.embed(q)
        kw = bm25.scores(idx, q)
        vec = quant.static_scores(v, arr["bits"], arr["q8"], arr["scale"], 100)
        lists = [[d for d, _ in kw]] + ([[d for d, _ in vec]] if sm.ids(q) else [])
        rows.append({
            "q": q, "terms": bm25_terms(q), "ids": sm.ids(q),
            "vec_hex": v.astype("<f8").tobytes().hex(),
            "bm25": kw, "static": vec, "fused": fusion.rrf(lists),
        })
    (out / "expected.json").write_text(json.dumps({"site": str(site), "rows": rows}, ensure_ascii=False))


if __name__ == "__main__":
    main(*sys.argv[1:])
