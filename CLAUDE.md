# CLAUDE.md — GlassineOS/search

Semantic search over the docs: builder + MkDocs plugin, browser worker, `glassine-docs`
CLI (see `README.md`, file format in `docs/format.md`). Every change: run `make check`
and extend the tests for new behaviour.

- Python (`glassine_docs/*.py`) and JavaScript (`glassine_docs/web/*.js`) implement the
  tokenizer, BM25, quantization and fusion twice: change both, keep the parity tests
  bit-exact. A change to what gets embedded or stored bumps `FORMAT_VERSION` and
  `docs/format.md`.
- Models only under MIT or Apache-2.0, pinned in `glassine_docs/models.json` by revision
  and sha256. Never commit model files or other binaries (`*.bin` is blocked).
- Only published documentation is indexed. No private notes, local paths or vendor
  material in code, tests or fixtures.
- Commits carry an `Assisted-by:` or `Co-Authored-By:` trailer. Never add
  `Signed-off-by`. Run `.glassine/org/scripts/safety-scan.sh --staged` before committing.
- Organization guidance for coding agents, including the project knowledge search
  (run `git submodule update --init` if the import below is missing):

@.glassine/org/AGENTS.md
