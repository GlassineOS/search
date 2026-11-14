# Search file format, version 2

The docs build writes these files into `search/` next to the site. Every file except
`manifest.json` carries the first 12 hex digits of its sha256 in its name
(`chunks.83ab54c59e5f.json`), so a name never changes its content. Clients read
`manifest.json` first (revalidated with its ETag) and fetch only files whose names
they don't have yet.

The index covers the site's own pages and, per decision
[0012](https://docs.glassine.org/decisions/0012-where-documentation-lives/), the
documentation folders of other public repositories listed in the build's sources file
(see [Sources](#sources)). Each of these is a *source*.

## manifest.json

```jsonc
{
  "format": 2,
  "generator": "glassine-docs 0.1.0",
  "site_url": "https://docs.glassine.org/",
  "commit": "<docs commit>",           // the site's own source; "-dirty" for a tree with changes
  "dirty": false,
  "sources": [
    {"name": "docs", "commit": "<docs commit>", "folder": "docs"},
    {"name": "search", "commit": "<full commit>", "repository": "GlassineOS/search",
     "folder": "docs", "url": "https://github.com/GlassineOS/search/blob/<full commit>/"}
  ],
  "chunks": {"file": "chunks.<h>.json", "sha256": "…", "size": 0, "count": 0},
  "bm25":   {"file": "bm25.<h>.json",   "sha256": "…", "size": 0, "k1": 1.2, "b": 0.75},
  "fusion": {"method": "rrf", "k": 60, "depth": 100},
  "profiles": {
    "static": {
      "model": {"id": "minishlab/potion-base-8M", "revision": "…", "license": "MIT",
                "file": "static-model.<h>.bin", "sha256": "…", "size": 0},
      "dims": 256,
      "quantization": {"vectors": "…", "model": "…", "candidates": 100},
      "vectors": {"file": "static.<h>.bin", "sha256": "…", "size": 0}
    },
    "transformer": {
      "model": {"id": "BAAI/bge-small-en-v1.5", "revision": "…", "license": "MIT",
                "sha256": "<sha256 of the ONNX file>", "onnx": "onnx/model.int8.onnx",
                "files": [{"path": "…", "asset": "…", "sha256": "…", "size": 0}],
                "query_prefix": "Represent this sentence for searching relevant passages: ",
                "pooling": "cls", "max_length": 512},
      "dims": 384,
      "quantization": {"vectors": "…", "model": "int8 dynamic"},
      "vectors": {"file": "transformer.<h>.bin", "sha256": "…", "size": 0}
    }
  },
  "web": {"text.js": "text.<h>.js", "score.js": "score.<h>.js",
          "worker": "assets/javascripts/workers/glassine-search.<h>.js"},
  "related": {"profile": "static", "count": 5}
}
```

A profile's model files are either on the site (`model.file`, the static profile) or
release assets of the mirror (`model.files[].asset`, the transformer profile; the
default mirror is `https://github.com/GlassineOS/search/releases/download/models-1/`).
Clients check every file against its sha256. A reader must reject a `format` it does
not know; `glassine-docs` and the site's worker read formats 1 and 2.

- `commit` and `dirty` are those of the site's own source, the checkout the build ran
  in. Every other source is at the commit its entry pins, so it is never dirty.
- `sources` lists every source the chunks come from, the site's own first. `name` is
  the source's name in chunk IDs (the repository's name). `folder` is its documentation
  folder in its repository, `repository` and `url` are given for other repositories:
  `url` is where their pages link, the page's path in the repository follows it.

## chunks.json

```json
{"format": 2, "commit": "…", "chunks": [
  {"id": "docs@<commit>:docs/guides/recovery.md#enter-recovery", "source": "docs",
   "page": "docs/guides/recovery.md", "title": "Recovery", "heading": "Enter recovery",
   "path": ["Enter recovery"], "url": "guides/recovery/#enter-recovery", "text": "…"},
  {"id": "search@<commit>:docs/format.md#chunksjson", "source": "search",
   "page": "docs/format.md", "title": "Search file format, version 2", "heading": "chunks.json",
   "path": ["chunks.json"],
   "url": "https://github.com/GlassineOS/search/blob/<commit>/docs/format.md#chunksjson",
   "text": "…"}]}
```

- `commit` is the manifest's `commit`.
- `id` is `<source>@<commit>:<page>#<anchor>`, with the commit of its source. `<page>`
  is the source file's path in its repository, documentation folder included (it was
  the path inside the documentation folder in format 1). `<anchor>` is the heading's
  HTML id as the page's link target renders it: MkDocs' for the site's own pages,
  GitHub's for other repositories' (see [Sources](#sources)). The text before the first
  `h2` has no `#<anchor>`. A section longer than 300 words is split at block
  boundaries; its parts after the first get `~2`, `~3`, … after the anchor and link to
  the same `url`.
- `source` is the `name` of its entry in the manifest's `sources` (new in format 2).
  `source` and `page` together name a page; `page` alone can repeat across sources.
- `url` is relative to the site root for the site's own pages, and absolute for other
  repositories' pages, which are not on the site: their rendering on GitHub at the
  pinned commit, `<source url><page>#<anchor>`.
- `path` is the trail of headings from `h2` down to this one.
- The order of `chunks` is the row order of every vector file and the document number
  of the BM25 index.

What gets indexed and embedded for a chunk is its trail and its text:
`<title> › <path…>` on the first line, then `text`.

## bm25.json

```json
{"k1": 1.2, "b": 0.75, "n": 167, "avgdl": 81.3, "doclen": [..],
 "stopwords": ["a", "an", ..],
 "postings": {"boot_b": [doc, tf, doc, tf, ..], ..}}
```

Terms: the text is decomposed (NFD), combining marks are dropped, each character is
lower-cased on its own. A term is a run of letters and digits; runs joined by one of
`_ - . : /` form a compound term, which is indexed whole and as its parts (`boot_b` →
`boot_b`, `boot`, `b`). Stop words are dropped as parts, never inside a compound.
Score: Okapi BM25 with `idf = ln(1 + (n - df + 0.5) / (df + 0.5))`, summed over the
query's distinct terms in sorted order.

## Vector and model files (`.bin`)

One container for all binary files, little endian:

| Bytes | Content |
|---|---|
| 8 | magic `GLDSRCH1` |
| 4 | u32: length of the header |
| n | header, UTF-8 JSON: `{"meta": {..}, "arrays": {name: {"dtype", "shape", "offset"}}}` |
| … | arrays at their offsets (multiples of 8 from the file start); dtypes `i8`, `u8`, `f32` |

- **static-model** (`meta.kind = "static-model"`): `table` i8 `[vocab, dims]`, `scale` f32
  `[vocab]`. `meta.tokenizer` holds the vocabulary and the tokenizer flags; `meta` also
  has `normalize`, `max_length`, `median_token_length`, the source model and the sha256
  of its files.
- **static vectors**: `bits` u8 `[n, dims/8]` (sign bits, bit set when the value is
  above 0, most significant bit first), `q8` i8 `[n, dims]`, `scale` f32 `[n]`.
- **transformer vectors**: `q8` i8 `[n, dims]`, `scale` f32 `[n]`.

int8 rows: `scale = max|x| / 127` (as f32), `q = round(x / scale)` with halves rounded
away from zero, clipped to ±127. All vectors are L2-normalised before quantization.

## Embedding

- **static**: the text cut to `max_length × median_token_length` characters, BERT
  WordPiece without `[CLS]`/`[SEP]`, at most `max_length` tokens, `[UNK]` dropped. The
  vector is the mean of `table[id] × scale[id]` (summed in token order), then divided
  by its L2 norm plus 1e-32. These are model2vec's rules, applied to the int8 table.
- **transformer**: `[CLS] tokens [SEP]` (at most 512), `[CLS]` output, L2-normalised.
  Queries are prefixed with `query_prefix`, passages are not.

## Sources

The build's input is a TOML file (the plugin's `sources_file`; the docs repository's
`search-sources.toml`) with one `[[source]]` table per repository:

| Field | Meaning |
|---|---|
| `repository` | a public repository, `GlassineOS/<name>`; `<name>` becomes the source's name |
| `folder` | its documentation folder, relative to the repository root |
| `commit` | the full 40-character commit to index |

Each repository is fetched at its commit with git, without credentials, so a private or
missing repository, a commit it doesn't have or a missing folder fails the build; no
entry is skipped. Every `.md` file in the folder and below is indexed (hidden files and
folders and symbolic links are not). Pages are rendered with Python-Markdown (`extra`,
`sane_lists`), and headings get GitHub's anchors: the heading text in lower case,
everything but letters, marks, numbers, `_`, `-` and spaces dropped, spaces turned into
`-`; a repeated anchor gets `-1`, `-2`, …. A page's title is its first `h1`, else its
file name.

## The site's search worker

`web.worker` (a path from the site root) replaces Material's search worker. The plugin
sets Material's `#__config` `search` key to
`<worker>?material=<Material's worker file>&search=<search/ relative to the worker>`.
The worker imports Material's worker and relays Material's messages (`0` SETUP,
`1` READY, `2` QUERY, `3` RESULT with `{items: SearchItem[][], suggest?}`). The first
non-empty QUERY loads `text.js`, `score.js`, the static model, the static vectors and
the chunks. From then on a RESULT holds Material's items and the sections the vectors
find (up to 10, as Material's documents without highlighting), ordered by reciprocal
rank fusion of Material's keyword order and the vector order, grouped by page as
Material groups them, with `score` = fused score × 1000. Sections of other
repositories are not in Material's index: they join as documents of their own (the
heading as title, the first 300 characters of the text) with their page as the parent,
linking to their absolute `url`. A section of the site that Material's index leaves out
is not shown. A RESULT is only sent for the
newest query; when the model arrives, the newest query's RESULT is sent again, fused.

## Search

1. BM25 over all chunks.
2. Vector scores. static: the `candidates` chunks nearest by Hamming distance of the
   sign bits, rescored by `dot(q8_query, q8_chunk) × scale_query × scale_chunk`
   (the query quantized like a chunk). transformer: `dot(query, q8_chunk) × scale_chunk`
   over all chunks.
3. Reciprocal rank fusion: `Σ 1 / (k + rank)` over the BM25 list and the vector list,
   each cut to `depth`. A query without any known token (static) uses the BM25 list only.

Ties sort by chunk number everywhere.
