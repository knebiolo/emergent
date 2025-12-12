#!/usr/bin/env bash
set -euo pipefail

TEST_PATTERN=${1:-src/emergent/fish_passage/tests}
mkdir -p tmp/pytest_tmp
export PYTEST_TMPDIR=tmp/pytest_tmp
python -m pytest -q "$TEST_PATTERN"
