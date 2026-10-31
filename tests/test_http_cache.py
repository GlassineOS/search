# SPDX-License-Identifier: GPL-2.0-or-later
"""The client against a local HTTP server that sets ETags like GitHub Pages."""
import hashlib
import http.server
import json
import os
import threading
from functools import partial

import pytest

from glassine_docs import cli
from glassine_docs.client import Site, cache_root
from helpers import build_site, pages


class Handler(http.server.SimpleHTTPRequestHandler):
    log = []

    def log_message(self, *a):
        pass

    def do_GET(self):
        path = self.translate_path(self.path)
        if not os.path.isfile(path):
            self.send_error(404)
            return
        data = open(path, "rb").read()
        etag = '"' + hashlib.sha256(data).hexdigest()[:16] + '"'
        if self.headers.get("If-None-Match") == etag:
            Handler.log.append((self.path, 304))
            self.send_response(304)
            self.send_header("ETag", etag)
            self.end_headers()
            return
        Handler.log.append((self.path, 200))
        self.send_response(200)
        self.send_header("ETag", etag)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture
def server(tmp_path):
    root = tmp_path / "www"
    (root / "search").mkdir(parents=True)
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), partial(Handler, directory=str(root)))
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    Handler.log = []
    yield root / "search", f"http://127.0.0.1:{httpd.server_address[1]}/search/", httpd
    httpd.shutdown()


def run(base, *extra):
    return cli.main(["search", "recovery", "--profile", "static", "--base", base, "--json", *extra])


def test_etag_and_hash_caching(server, tmp_path, model_file, capsys):
    out, base, httpd = server
    build_site(out, model_file, tmp_path / "emb")
    assert run(base) == 0
    first = [p for p, _ in Handler.log]
    assert first[0] == "/search/manifest.json" and len(first) == 5  # manifest, chunks, bm25, model, vectors
    assert cache_root() == tmp_path / "xdg" / "glassine" / "docs-search"

    Handler.log = []
    assert run(base) == 0
    assert Handler.log == [("/search/manifest.json", 304)]  # nothing else is fetched

    # change one page: chunks, bm25 and vectors change; the model does not
    ps = pages()
    ps[0].html += "<p>A new sentence about slots.</p>"
    build_site(out, model_file, tmp_path / "emb", ps)
    Handler.log = []
    assert run(base) == 0
    got = sorted(p.split("/")[-1].split(".")[0] for p, code in Handler.log if code == 200)
    assert got == ["bm25", "chunks", "manifest", "static"]
    site = Site(base)
    cached = sorted(p.name.split(".")[0] for p in (site.dir / "files").iterdir())
    assert cached == ["bm25", "chunks", "static", "static-model"]  # old versions pruned
    capsys.readouterr()


def test_offline_from_cache(server, tmp_path, model_file, capsys):
    out, base, httpd = server
    build_site(out, model_file, tmp_path / "emb")
    assert run(base) == 0
    want = json.loads(capsys.readouterr().out)["results"]
    httpd.shutdown()
    httpd.server_close()
    assert run(base) == 0  # server gone: cached manifest
    assert json.loads(capsys.readouterr().out)["results"] == want
    assert run(base, "--offline") == 0
    capsys.readouterr()


def test_no_cache_and_no_network(tmp_path, capsys):
    assert run("http://127.0.0.1:9/search/") == 2
    assert "manifest.json" in capsys.readouterr().err
