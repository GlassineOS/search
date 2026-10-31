# SPDX-License-Identifier: GPL-2.0-or-later
"""Pinned model files: the lock (``models.json``), download and verification.

Every file is fetched from the mirror (release assets of GlassineOS/search) by
its asset name and checked against the sha256 in the lock. With ``hf=True`` a
file the mirror does not have is fetched from Hugging Face at the pinned
revision instead (a development fallback; the hash check is the same).
Derived files (a quantized ONNX model) are rebuilt from their source when the
mirror lacks them, and must hash to the locked value too.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

LOCK_PATH = Path(__file__).with_name("models.json")
HF_BASE = "https://huggingface.co/"


class ModelError(RuntimeError):
    pass


def load_lock(path: Path | None = None) -> dict:
    return json.loads((path or LOCK_PATH).read_text(encoding="utf-8"))


def default_mirror(lock: dict | None = None) -> str:
    return os.environ.get("GLASSINE_MODELS_MIRROR") or (lock or load_lock())["mirror"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def model_dir(cache: Path, entry: dict) -> Path:
    return Path(cache) / entry["id"].replace("/", "--") / entry["revision"]


def cached_path(cache: Path, entry: dict, path: str) -> Path:
    return model_dir(cache, entry) / path


def _download(url: str, dest: Path, sha256: str) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    h = hashlib.sha256()
    req = urllib.request.Request(url, headers={"User-Agent": "glassine-docs"})
    with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
        for block in iter(lambda: r.read(1 << 20), b""):
            h.update(block)
            f.write(block)
    if h.hexdigest() != sha256:
        tmp.unlink(missing_ok=True)
        raise ModelError(f"{url}: sha256 {h.hexdigest()} != locked {sha256}")
    os.replace(tmp, dest)


def fetch_file(cache: Path, entry: dict, f: dict, mirror: str, hf: bool = False, log=None) -> Path:
    """Return the verified local path of one locked file, downloading it if needed."""
    dest = cached_path(cache, entry, f["path"])
    if dest.exists() and dest.stat().st_size == f["size"] and sha256_file(dest) == f["sha256"]:
        return dest
    errors = []
    url = mirror.rstrip("/") + "/" + f["asset"]
    try:
        if log:
            log(f"fetching {url}")
        _download(url, dest, f["sha256"])
        return dest
    except (urllib.error.URLError, OSError, ModelError) as e:
        errors.append(f"{url}: {e}")
    if hf and "from" in f:
        src = next(s for s in entry["files"] if s["path"] == f["from"])
        src_path = fetch_file(cache, entry, src, mirror, hf=True, log=log)
        if log:
            log(f"deriving {f['path']} from {f['from']} ({f['recipe']})")
        derive(f, src_path, dest)
        got = sha256_file(dest)
        if got != f["sha256"]:
            dest.unlink(missing_ok=True)
            raise ModelError(f"derived {f['path']}: sha256 {got} != locked {f['sha256']} "
                             f"(recipe {f['recipe']} needs onnxruntime {f.get('onnxruntime')})")
        return dest
    if hf:
        url = f"{HF_BASE}{entry['id']}/resolve/{entry['revision']}/{f['path']}"
        try:
            if log:
                log(f"fetching {url}")
            _download(url, dest, f["sha256"])
            return dest
        except (urllib.error.URLError, OSError, ModelError) as e:
            errors.append(f"{url}: {e}")
    raise ModelError("could not fetch " + f["path"] + ":\n  " + "\n  ".join(errors))


def derive(f: dict, src: Path, dest: Path) -> None:
    if f["recipe"] != "onnxruntime-quantize-dynamic-qint8":
        raise ModelError(f"unknown recipe {f['recipe']}")
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
    except ImportError as e:
        raise ModelError(f"rebuilding {f['path']} needs onnxruntime and onnx: "
                         'pip install "glassine-docs[derive]" (or use --profile static)') from e

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part.onnx")
    quantize_dynamic(str(src), str(tmp), weight_type=QuantType.QInt8)
    os.replace(tmp, dest)


def fetch_profile(cache: Path, profile: str, mirror: str | None = None, hf: bool = False,
                  only: list[str] | None = None, lock: dict | None = None) -> dict[str, Path]:
    """Fetch the files a profile needs (all of them, or the paths in ``only``)."""
    lock = lock or load_lock()
    entry = lock["models"][profile]
    mirror = mirror or default_mirror(lock)
    out = {}
    for f in entry["files"] + entry.get("derived", []):
        if only is not None and f["path"] not in only:
            continue
        out[f["path"]] = fetch_file(Path(cache), entry, f, mirror, hf=hf,
                                    log=lambda m: print(m, file=sys.stderr))
    return out


def main(argv=None) -> int:
    """python -m glassine_docs.models fetch [--cache DIR] [--mirror URL] [--hf] [PROFILE...]"""
    import argparse

    p = argparse.ArgumentParser(prog="python -m glassine_docs.models")
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch", help="download and verify the locked model files")
    f.add_argument("profiles", nargs="*", default=None)
    f.add_argument("--cache", default=".cache/glassine-search/models")
    f.add_argument("--mirror")
    f.add_argument("--hf", action="store_true", help="fall back to Hugging Face (and rebuild derived files)")
    a = p.parse_args(argv)
    lock = load_lock()
    for prof in a.profiles or list(lock["models"]):
        for path, local in fetch_profile(Path(a.cache), prof, a.mirror, a.hf, lock=lock).items():
            print(f"{prof}\t{path}\t{local}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
