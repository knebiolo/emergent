import sys
import traceback

from test_heading_wrap import test_wrapping_examples, test_wrap_rad_consistent_with_deg

failed = False
try:
    test_wrapping_examples()
    print('test_wrapping_examples: PASS')
except Exception:
    failed = True
    print('test_wrapping_examples: FAIL')
    traceback.print_exc()

try:
    test_wrap_rad_consistent_with_deg()
    print('test_wrap_rad_consistent_with_deg: PASS')
except Exception:
    failed = True
    print('test_wrap_rad_consistent_with_deg: FAIL')
    traceback.print_exc()

if failed:
    sys.exit(2)
else:
    sys.exit(0)
