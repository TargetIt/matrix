#!/usr/bin/env sh
set -eu

repo_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo_dir"

node tests/model.test.js
node tests/static.test.js

echo "All MATRIX regression checks passed."
