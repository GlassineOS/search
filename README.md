# GlassineOS/search

Semantic search over the [GlassineOS documentation](https://docs.glassine.org/): the
build step of the docs site, the search in its search box, and `glassine-docs`, the
command-line search for people and coding agents. The design is decision record
[0011](https://docs.glassine.org/decisions/0011-semantic-search/); the files it
publishes are specified in [docs/format.md](docs/format.md).

## Search from a terminal

```sh
pip install "git+https://github.com/GlassineOS/search"
glassine-docs search "how do I get back into recovery if the system doesn't boot"
glassine-docs search "EDL" --profile static -k 3 --json
```

Each result gives the page and heading, a link to the heading, the text, and the docs
commit it came from. `--json` is for agents. The first search downloads the index
(about 0.3 MB) and the model of its profile once; later searches send one short
request to see whether the docs changed, and work offline from the cache
(`$XDG_CACHE_HOME/glassine/docs-search`, else `~/.cache/glassine/docs-search`).

| Profile | Model | Download | Needs |
|---|---|---|---|
| `transformer` (default) | bge-small-en-v1.5, int8 ONNX | 34 MB once | onnxruntime |
| `static` | potion-base-8M, int8 table | 8 MB once | numpy only |

Where onnxruntime has no wheel (musl): `pip install --no-deps "git+…"`, then
`pip install numpy`, and use `--profile static`.

## In the docs build

The MkDocs plugin `glassine-search` chunks every rendered page at its headings, builds
the BM25 index, embeds the chunks with both profiles, adds a "Related pages" block to
each page and writes `search/` into the site.

On the site it replaces Material's search worker. The plugin rewrites the `search` key
of Material's `#__config` to our worker, which loads Material's own worker inside
itself and speaks Material's protocol to the page, so the search box, results,
highlighting, suggestions and keyboard handling stay Material's. The keyword results
come from Material's worker at once; the first non-empty query loads the static model
(8 MB) and vectors, and from then on each result is the keyword list and the vector
list fused. Pages load nothing extra but the worker (6 KB).

```yaml
plugins:
  - search            # the theme's search: keep it, and first
  - glassine-search:
      related_profile: static    # or transformer
      related_count: 5
```

| Variable | Effect |
|---|---|
| `GLASSINE_SEARCH_MODEL_CACHE` | model download cache (default `.cache/glassine-search/models`) |
| `GLASSINE_SEARCH_EMBED_CACHE` | embeddings by chunk text and model (default `.cache/glassine-search/embeddings`) |
| `GLASSINE_SEARCH_PROFILES` | e.g. `static` for a quick local preview |
| `GLASSINE_MODELS_MIRROR` | where model files come from (default the `models-1` release of this repository) |
| `GLASSINE_MODELS_HF=1` | fall back to Hugging Face at the pinned revision (development) |

## Contents

| Path | What |
|---|---|
| `glassine_docs/text.py`, `web/text.js` | BERT WordPiece tokenizer and BM25 terms, one per language, same behaviour |
| `glassine_docs/{quant,bm25,fusion,staticmodel}.py`, `web/score.js` | quantization, BM25, reciprocal rank fusion, the static embedder |
| `glassine_docs/transformer.py` | the transformer profile (onnxruntime, CPU) |
| `glassine_docs/chunk.py`, `builder.py`, `plugin.py` | chunking, the build, the MkDocs plugin |
| `glassine_docs/client.py`, `cli.py` | `glassine-docs` |
| `glassine_docs/web/worker.js` | replaces Material's search worker: runs it for keywords, fuses in the vector ranking |
| `glassine_docs/models.json` | the model lock: Hugging Face revision, files, sha256, release asset names |
| `tools/mirror-models.sh` | prepares the release assets of the model mirror |
| `tests/` | pytest and `node --test` suites |

## Tests

```sh
make check            # offline: synthetic model, sample pages, Python and JS parity
make check-models     # with the real models (downloads them once; HF=1 to allow Hugging Face)
make check-browser    # the worker swap in headless Chromium (Playwright)
```

`check-browser` builds a small site with Material, types into its search box and checks
that a hit only the vectors can find is listed and ranked, that the keyword results
show before the model arrives, and that arrows and Enter work. It relies on Material's
internal worker protocol, so it runs against the Material range the docs repository
uses (`mkdocs-material>=9.7,<10`, the `dev` extra); keep the two in step.

The JS tokenizer and the Python one are both checked against Hugging Face
`tokenizers` ids for both models. The node tests check the browser's query vectors bit
for bit, its candidate scores and its fused ranking against the Python side.

Licence: GPL-2.0-or-later (`LICENSE`). The models are MIT-licensed by their authors and
are not part of this repository.
