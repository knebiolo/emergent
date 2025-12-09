"""Create an inventory of all `except Exception` occurrences in sockeye.py

Writes `tmp_replace_backups/sockeye_except_inventory-<timestamp>.md` with line numbers
and 3 lines of context for each match.

Usage: python tools\inventory_broad_excepts.py
"""
from pathlib import Path
import re
from datetime import datetime
ROOT = Path(__file__).resolve().parents[1]
SOCKEYE = ROOT / 'src' / 'emergent' / 'salmon_abm' / 'sockeye.py'
OUTDIR = ROOT / 'tmp_replace_backups'
OUTDIR.mkdir(exist_ok=True)
text = SOCKEYE.read_text(encoding='utf-8')
lines = text.splitlines()
pattern = re.compile(r'except\s+Exception(?:\s+as\s+\w+)?\s*:')
matches = []
for i, line in enumerate(lines, start=1):
    if pattern.search(line):
        start = max(1, i-3)
        end = min(len(lines), i+3)
        context = '\n'.join(f'{ln:5d}: {lines[ln-1]}' for ln in range(start, end+1))
        matches.append((i, line.strip(), context))

now = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
out_path = OUTDIR / f'sockeye_except_inventory-{now}.md'
with open(out_path, 'w', encoding='utf-8') as f:
    f.write('# sockeye.py except Exception inventory\n\n')
    f.write(f'Generated: {now}\n\n')
    f.write(f'Total matches: {len(matches)}\n\n')
    for ln, ltxt, ctx in matches:
        f.write(f'## Line {ln}\n')
        f.write('''```\n''')
        f.write(ctx)
        f.write('\n```\n\n')
print('Inventory written to', out_path)
