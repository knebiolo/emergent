import traceback
import sys

try:
    import emergent.salmon_abm.sockeye_SoA as mod
    print('IMPORT_OK')
    # print some identifying info
    print('module file:', getattr(mod, '__file__', '<unknown>'))
    print('available attrs:', [a for a in dir(mod) if not a.startswith('_')][:50])
except Exception:
    print('IMPORT_FAILED')
    traceback.print_exc()
    sys.exit(2)
