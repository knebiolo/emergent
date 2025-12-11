param(
    [string]$TestPattern = 'src/emergent/fish_passage/tests'
)

# Ensure project temp dir exists
New-Item -ItemType Directory -Path tmp/pytest_tmp -Force | Out-Null

$env:PYTEST_TMPDIR = 'tmp/pytest_tmp'
python -m pytest -q $TestPattern
