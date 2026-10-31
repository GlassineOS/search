#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
"""Regenerate the tokenizer golden fixtures (dev only: needs ``tokenizers``).

    tools/make-golden.py MODEL_CACHE_DIR

Reads both models' tokenizer.json from the model cache (``glassine-docs-build
fetch-models`` fills it) and writes, for each model, ``tests/fixtures/
<profile>-tokenizer.json`` (our compact config: vocabulary and flags) and the
ids that Hugging Face ``tokenizers`` gives for every fixture text into
``tests/fixtures/golden-tokens.json``.
"""
import json
import sys
from pathlib import Path

from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from glassine_docs import models, text  # noqa: E402

TEXTS = json.loads((ROOT / "tests/fixtures/token-texts.json").read_text(encoding="utf-8"))


def main(cache: str) -> None:
    lock = models.load_lock()
    golden = {}
    for profile, entry in lock["models"].items():
        tj = models.cached_path(Path(cache), entry, "tokenizer.json")
        cfg = text.hf_tokenizer_config(tj)
        (ROOT / f"tests/fixtures/{profile}-tokenizer.json").write_text(
            json.dumps(cfg, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        tok = Tokenizer.from_file(str(tj))
        golden[profile] = {
            "plain": [tok.encode(t, add_special_tokens=False).ids for t in TEXTS],
            "special": [tok.encode(t, add_special_tokens=True).ids for t in TEXTS],
        }
    (ROOT / "tests/fixtures/golden-tokens.json").write_text(
        json.dumps(golden, separators=(",", ":")), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1])
