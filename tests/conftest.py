# SPDX-License-Identifier: GPL-2.0-or-later
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

COMMIT = "c0ffee" + "0" * 34


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    monkeypatch.setenv("GITHUB_SHA", COMMIT)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    monkeypatch.delenv("GLASSINE_MODELS_MIRROR", raising=False)


@pytest.fixture(scope="session")
def model_file(tmp_path_factory):
    from helpers import synthetic_static_model
    p = tmp_path_factory.mktemp("model") / "static-model.bin"
    p.write_bytes(synthetic_static_model())
    return p


@pytest.fixture
def site(tmp_path, model_file):
    from helpers import build_site
    out = tmp_path / "site-search"
    res = build_site(out, model_file, tmp_path / "emb")
    return out, res


def real_models_dir():
    d = os.environ.get("GDS_MODELS")
    return Path(d) if d else None
