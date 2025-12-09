import re
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOCKEYE = ROOT / 'src' / 'emergent' / 'salmon_abm' / 'sockeye.py'
OUT = Path(__file__).resolve().parent / 'except_exception_inventory.csv'

lines = SOCKEYE.read_text(encoding='utf-8').splitlines()

matches = []
for i, line in enumerate(lines, start=1):
    if 'except Exception:' in line:
        prev_line = lines[i-2] if i-2 >= 0 else ''
        next_line = lines[i] if i < len(lines) else ''
        # Try to find function/class context (search upward)
        context = ''
        for j in range(i-1, max(0, i-40), -1):
            l = lines[j-1]
            m = re.match(r"\s*def\s+(\w+)\s*\(|\s*class\s+(\w+)\s*[:(]", l)
            if m:
                context = m.group(1) or m.group(2)
                break
        # Heuristic risk tagging
        tag = 'low'
        clue = (prev_line + ' ' + next_line + ' ' + context).lower()
        high_words = ['timestep', 'collision', 'hecras', 'hdf', 'flush', 'write', 'query_pairs', 'kdtre', 'kdtree', 'numba', 'jit', 'precompile', 'warmup', 'run', 'initialize', 'init', 'apply_hecras_mapping', 'map_', 'compute_']
        med_words = ['numba', 'kernel', 'compile', 'cache', 'warmup']
        if any(w in clue for w in high_words):
            tag = 'high'
        elif any(w in clue for w in med_words):
            tag = 'medium'

        recommendation = 'Replace broad except with specific exceptions (e.g., ValueError, TypeError, IndexError, OSError) and use logger.exception; re-raise unexpected errors.'
        matches.append((len(matches)+1, i, context, prev_line.strip(), line.strip(), next_line.strip(), tag, recommendation))

# write CSV
with OUT.open('w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['id','line','context','prev_line','match_line','next_line','risk','recommendation'])
    for row in matches:
        writer.writerow(row)

print(f'Wrote {len(matches)} entries to {OUT}')
