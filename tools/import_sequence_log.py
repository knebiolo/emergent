"""Import test modules sequentially and append progress to tmp/import_seq_progress.txt so we capture progress even if process dies."""
import sys
from pathlib import Path
import importlib.util

out = Path('tmp/import_seq_progress.txt')
if out.exists():
    out.unlink()

if len(sys.argv) < 2:
    out.write_text('missing arg')
    sys.exit(2)

p = Path(sys.argv[1])
if not p.exists():
    out.write_text('list missing')
    sys.exit(2)

lines = [l.strip() for l in p.read_text(encoding='utf-8').splitlines() if l.strip()]
for i, lp in enumerate(lines):
    mode = 'w' if i == 0 else 'a'
    with open(out, mode, encoding='utf-8') as fh:
        fh.write(f'IMPORT {i} {lp}\n')
        fh.flush()
    mp = Path(lp)
    modname = ('seq_import_' + mp.stem)
    try:
        spec = importlib.util.spec_from_file_location(modname, str(mp))
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        with open(out, 'a', encoding='utf-8') as fh:
            fh.write(f'OK {lp}\n')
    except Exception as e:
        with open(out, 'a', encoding='utf-8') as fh:
            fh.write(f'EXC {lp} {repr(e)}\n')
print('done')
