# SPDX-License-Identifier: GPL-2.0-or-later
import json

import pytest

from glassine_docs.text import WordPiece, bm25_terms
from helpers import FIX, tokenizer_config

TEXTS = json.loads((FIX / "token-texts.json").read_text(encoding="utf-8"))
GOLDEN = json.loads((FIX / "golden-tokens.json").read_text())


@pytest.mark.parametrize("profile", ["static", "transformer"])
def test_wordpiece_matches_hf_tokenizers(profile):
    wp = WordPiece(tokenizer_config(profile))
    got = [wp.encode(t) for t in TEXTS]
    assert got == GOLDEN[profile]["plain"]


def test_wordpiece_special_tokens_transformer():
    # potion's tokenizer.json post-processor names bge-base ids for [CLS]/[SEP]; model2vec never
    # adds special tokens, so only the transformer's are checked.
    wp = WordPiece(tokenizer_config("transformer"))
    assert [wp.encode(t, add_special_tokens=True) for t in TEXTS] == GOLDEN["transformer"]["special"]


def test_wordpiece_truncation():
    wp = WordPiece(tokenizer_config("transformer"))
    ids = wp.encode("word " * 600, add_special_tokens=True, max_length=512)
    assert len(ids) == 512 and ids[0] == wp.cls_id and ids[-1] == wp.sep_id


@pytest.mark.parametrize("text,terms", [
    ("boot_b", ["boot_b", "boot", "b"]),
    ("glassine-epdd", ["glassine-epdd", "glassine", "epdd"]),
    ("/dev/mmcblk0p48", ["dev/mmcblk0p48", "dev", "mmcblk0p48"]),
    ("Über Größe", ["uber", "große"]),
    ("the EDL mode", ["edl", "mode"]),
    ("v1.6.2, end.", ["v1.6.2", "v1", "6", "2", "end"]),
    ("--force x--y", ["force", "x", "y"]),
    ("ΟΔΟΣ", ["οδοσ"]),
])
def test_bm25_terms(text, terms):
    assert bm25_terms(text) == terms
