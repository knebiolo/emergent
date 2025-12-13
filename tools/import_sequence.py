"""Import test modules sequentially from a file list to find which import triggers a native crash.
Usage: python tools/import_sequence.py tmp/subset_second_half.txt
"""
import sys
from pathlib import Path
import importlib.util

if len(sys.argv) < 2:
    print('usage: import_sequence.py <file_with_paths>')
    sys.exit(2)

p = Path(sys.argv[1])
if not p.exists():
    print('list file missing', p)
    sys.exit(2)

lines = [l.strip() for l in p.read_text(encoding='utf-8').splitlines() if l.strip()]
for i, lp in enumerate(lines):
    print(f'IMPORT {i} {lp}')
    # derive module name from path
    mp = Path(lp)
    modname = ('test_' + mp.stem)
    try:
        spec = importlib.util.spec_from_file_location(modname, str(mp))
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        print('OK', lp)
    except Exception as e:
        print('EXC', lp, repr(e))
        # continue to next
print('done')
