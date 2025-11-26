import re
import statistics
from pathlib import Path

fn = Path(__file__).with_name('headless_run_log.txt')
if not fn.exists():
    print('Log file not found:', fn)
    raise SystemExit(1)

pat_err = re.compile(r'err=([\-0-9\.]+)')
pat_rud = re.compile(r'rud=([\-0-9\.]+)')

errs = []
ruds = []
lines = fn.read_text(encoding='utf-8', errors='replace').splitlines()
for L in lines:
    m = pat_err.search(L)
    if m:
        try:
            errs.append(float(m.group(1)))
        except ValueError:
            pass
    m2 = pat_rud.search(L)
    if m2:
        try:
            ruds.append(float(m2.group(1)))
        except ValueError:
            pass

print('lines:', len(lines))
print('err_count', len(errs))
if errs:
    print('err_mean', statistics.mean(errs), 'err_std', statistics.pstdev(errs) if len(errs)>1 else 0)
print('rud_count', len(ruds))
if ruds:
    print('rud_mean', statistics.mean(ruds), 'rud_std', statistics.pstdev(ruds) if len(ruds)>1 else 0)

print('\nSample (first 20 DR/CTRL lines):')
for L in lines[:40]:
    print(L)
