#!/bin/sh
# SPDX-License-Identifier: GPL-2.0-or-later
# Prepare the model release assets (release "models-1" of GlassineOS/search).
#
#   tools/mirror-models.sh OUT_DIR [MODEL_CACHE]
#
# Downloads every file of models.json from Hugging Face at the pinned revision
# (or takes it from MODEL_CACHE), rebuilds the derived int8 model, checks each
# sha256 against the lock, and copies the files into OUT_DIR under their asset
# names, with SHA256SUMS. It uploads nothing; the last lines print the command.
set -eu
out=${1:?usage: tools/mirror-models.sh OUT_DIR [MODEL_CACHE]}
cache=${2:-.cache/glassine-search/models}
here=$(cd "$(dirname "$0")/.." && pwd)
py=${PYTHON:-python3}

cd "$here"
"$py" -m glassine_docs.models fetch --cache "$cache" --hf --mirror "file:///nonexistent/" >/dev/null
mkdir -p "$out"
"$py" - "$cache" "$out" <<'PY'
import shutil, sys
from pathlib import Path
from glassine_docs import models
cache, out = Path(sys.argv[1]), Path(sys.argv[2])
lock = models.load_lock()
sums = []
for prof, entry in lock["models"].items():
    for f in entry["files"] + entry.get("derived", []):
        src = models.cached_path(cache, entry, f["path"])
        got = models.sha256_file(src)
        if got != f["sha256"]:
            sys.exit(f"{src}: sha256 {got} != locked {f['sha256']}")
        shutil.copyfile(src, out / f["asset"])
        sums.append(f"{f['sha256']}  {f['asset']}")
(out / "SHA256SUMS").write_text("\n".join(sums) + "\n")
print("\n".join(sums))
PY
(cd "$out" && sha256sum -c SHA256SUMS >/dev/null)
echo "assets ready in $out. To publish (licences: both models MIT, see models.json):"
echo "  gh release create models-1 --repo GlassineOS/search --title 'Models 1' \\"
echo "    --notes 'Pinned model files for glassine-docs search (sha256 in SHA256SUMS).' $out/*"
