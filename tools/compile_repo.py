#!/usr/bin/env python3
"""Byte-compile repository Python files and collect syntax errors.

Writes a report to `outputs/compile_errors.txt` and prints a summary to
stdout. Excludes `src/emergent/ship_abm` per project constraints.
"""
import sys
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[1]
out_dir = root / 'outputs'
out_dir.mkdir(exist_ok=True)
report = []

for p in sorted(root.rglob('*.py')):
    sp = str(p)
    if 'src\\emergent\\ship_abm' in sp or 'src/emergent/ship_abm' in sp:
        continue
    try:
        subprocess.check_output([sys.executable, '-m', 'py_compile', sp], stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        report.append((sp, e.output.decode('utf-8', errors='replace')))

report_path = out_dir / 'compile_errors.txt'
with report_path.open('w', encoding='utf-8') as f:
    f.write('FILES_WITH_ERRORS: {}\n\n'.format(len(report)))
    for fp, out in report:
        f.write('--- {}\n'.format(fp))
        f.write(out)
        f.write('\n\n')

print('Checked files, errors found:', len(report))
print('Detailed report written to', str(report_path))
