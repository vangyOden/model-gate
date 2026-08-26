#!/usr/bin/env bash
# Build the whole site into web/_site:
#
#   _site/index.html   the landing page
#   _site/docs/...     the MkDocs build
#
#   ./web/build.sh          build
#   ./web/build.sh serve    build, then serve the docs with live reload
set -euo pipefail
cd "$(dirname "$0")"

# The notebooks live in examples/ and are executed there by run_all.sh. The
# docs build consumes copies so there is exactly one source of truth.
echo "==> syncing notebooks from examples/"
for nb in ../examples/*.ipynb; do
    cp "$nb" "docs/examples/$(basename "$nb")"
done

if [ "${1:-}" = "serve" ]; then
    exec mkdocs serve
fi

echo "==> building docs"
rm -rf _site          # before mkdocs writes into _site/docs, not after
mkdocs build --strict

echo "==> copying landing page"
cp -R landing/. _site/

echo
echo "built: web/_site"
echo "  landing  _site/index.html"
echo "  docs     _site/docs/index.html"
