import os
import sys
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
# Run pytest programmatically so the environment is set in-process.
import pytest
rc = pytest.main(['-vv', '-r', 'a', '-s'])
print('PYTEST_EXIT_CODE:', rc)
sys.exit(rc)
