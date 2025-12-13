Developer testing instructions

- Run the main test suite (excludes GUI/OpenGL tools tests on Windows via conftest.py):

```bash
python -X faulthandler -m pytest --basetemp=tmp/pytest_basetemp
```

- Run the tools GUI/OpenGL diagnostics (run these on a machine with GPU or using Xvfb on Linux):

```bash
# On Linux (headless), start Xvfb first
Xvfb :99 -screen 0 1024x768x24 &
export DISPLAY=:99
python tools/run_tools_tests.py
```

- Run the main tests only (explicit subset):

```bash
python -X faulthandler -m pytest -q --basetemp=tmp/pytest_basetemp @tmp/subset_no_tools.txt
```

- CI notes: The repository includes a GitHub Actions workflow `.github/workflows/ci.yml` that runs main tests on `ubuntu-latest` and runs tools tests in a separate job under Xvfb.
