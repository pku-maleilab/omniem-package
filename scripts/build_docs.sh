#!/usr/bin/env bash
# Build the omniem API documentation from in-code docstrings using pdoc.
#
# pdoc imports the package and renders its public surface (the classes / functions
# in each module's ``__all__``) plus their Google-style docstrings to static HTML.
# Because it *imports* omniem, run this in the same environment where omniem and
# its dependencies (torch, numpy, monai, …) are installed.
#
# Usage:
#   scripts/build_docs.sh [OUTPUT_DIR]
#
# OUTPUT_DIR defaults to docs/api/ (a gitignored build artifact). Open the result
# at <OUTPUT_DIR>/omniem.html. Regenerate any time the docstrings change.
#
# Install the generator first:
#   python -m pip install "pdoc>=14"

set -euo pipefail

# Resolve the repo root from this script's location so the command works from any
# working directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Output directory — first arg or the default gitignored docs/api/.
OUTPUT_DIR="${1:-${REPO_ROOT}/docs/api}"

cd "${REPO_ROOT}"

# -d google      : parse Google-style docstring sections (Args/Returns/Raises).
# --no-show-source: keep the rendered pages focused on the API contract, not the
#                   implementation body.
# -o             : write static HTML into OUTPUT_DIR.
python -m pdoc omniem \
    -d google \
    --no-show-source \
    -o "${OUTPUT_DIR}"

echo "API docs written to: ${OUTPUT_DIR}/omniem.html"
