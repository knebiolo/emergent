"""Run only the angle/heading related tests without importing heavy GUI/GIS deps.
This imports the test modules directly and runs them using unittest to avoid pytest collection.
"""
import os
import sys
import unittest
import importlib

# ensure repo root (cwd) is importable so 'tests' package can be found
sys.path.insert(0, os.getcwd())

# Load test modules directly
mods = [
    'tests.test_angle_utils',
    'tests.test_heading_wrap',
    'tests.test_heading_error'
]
loader = unittest.TestLoader()
suite = unittest.TestSuite()
for m in mods:
    mod = importlib.import_module(m)
    suite.addTests(loader.loadTestsFromModule(mod))

runner = unittest.TextTestRunner(verbosity=2)
result = runner.run(suite)
if not result.wasSuccessful():
    raise SystemExit(1)
print('Focused tests passed')
