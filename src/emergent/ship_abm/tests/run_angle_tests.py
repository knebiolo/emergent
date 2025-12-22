import sys, traceback
from test_angle_utils import test_wrap_rad_basic, test_wrap_deg_basic, test_heading_diff_rad_examples, test_heading_diff_broadcasting

failed = False
for fn in [test_wrap_rad_basic, test_wrap_deg_basic, test_heading_diff_rad_examples, test_heading_diff_broadcasting]:
    try:
        fn()
        print(f"{fn.__name__}: PASS")
    except Exception:
        failed = True
        print(f"{fn.__name__}: FAIL")
        traceback.print_exc()

sys.exit(2 if failed else 0)
