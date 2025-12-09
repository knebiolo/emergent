# Quick diagnostic: find unbalanced try/except/finally blocks in a Python file
import sys
from pathlib import Path
p = Path(r"c:\Users\Kevin.Nebiolo\OneDrive - Kleinschmidt Associates\Software\emergent\src\emergent\salmon_abm\sockeye.py")
s = p.read_text()
lines = s.splitlines()
stack = []
issues = []
for i, l in enumerate(lines, start=1):
    stripped = l.lstrip()
    # ignore commented lines
    if stripped.startswith('#'):
        continue
    if stripped.startswith('try:'):
        stack.append(('try', i))
    elif stripped.startswith('except'):
        if stack and stack[-1][0] == 'try':
            stack.pop()
        else:
            issues.append((i, 'unmatched except'))
    elif stripped.startswith('finally:'):
        if stack and stack[-1][0] == 'try':
            stack.pop()
        else:
            issues.append((i, 'unmatched finally'))

print('Total try blocks still open:', len(stack))
for typ, ln in stack:
    print('Unmatched try at line', ln)
if issues:
    print('\nOther issues:')
    for ln, msg in issues:
        print(msg, 'at line', ln)
else:
    print('No stray except/finally detected')
sys.exit(0)