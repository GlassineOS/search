# SPDX-License-Identifier: GPL-2.0-or-later
PYTHON ?= python3
NODE ?= node
TMP ?= .cache-local
MODELS ?= .cache/glassine-search/models

.PHONY: check test-py test-js models check-models golden

# Offline: synthetic model, sample pages; no network, no model download.
check: test-py test-js

test-py:
	mkdir -p $(TMP)
	$(PYTHON) -m pytest -q --basetemp=$(TMP)/pytest

test-js:
	for f in glassine_docs/web/*.js; do $(NODE) --check $$f || exit 1; done
	$(PYTHON) -I tests/make_parity.py $(TMP)/parity
	GDS_PARITY_DIR=$(TMP)/parity $(NODE) --test tests/js/*.test.js

# The real models (downloads them once into $(MODELS); HF=1 allows Hugging Face).
models:
	$(PYTHON) -m glassine_docs.models fetch --cache $(MODELS) $(if $(HF),--hf)

check-models: models
	GDS_MODELS=$(MODELS) $(PYTHON) -m pytest -q --basetemp=$(TMP)/pytest tests/test_real_models.py
	$(PYTHON) -c "import sys; from pathlib import Path; from glassine_docs import models, staticmodel; e = models.load_lock()['models']['static']; Path('$(TMP)').mkdir(exist_ok=True); Path('$(TMP)/static-model.bin').write_bytes(staticmodel.convert(models.model_dir(Path('$(MODELS)'), e), e))"
	$(PYTHON) -I tests/make_parity.py $(TMP)/parity-real $(TMP)/static-model.bin
	GDS_PARITY_DIR=$(TMP)/parity-real $(NODE) --test tests/js/*.test.js

# Regenerate the tokenizer golden fixtures (needs the dev extra: tokenizers).
golden: models
	$(PYTHON) -I tools/make-golden.py $(MODELS)
