import sys
from pathlib import Path
p = Path(r'src/emergent/salmon_abm/sockeye_SoA_OpenGL_RL.py')
s = p.read_text()
lines = s.splitlines()
occ = []
for i,l in enumerate(lines, start=1):
    if '"""' in l:
        occ.append((i, '"""', l))
    if "'''" in l:
        occ.append((i, "'''", l))
print('Found', len(occ), 'triple-quote occurrences')
for ln, t, l in occ[:200]:
    print(f'{ln:5d} {t} -> {l.strip()}')
# Now try to find unbalanced by scanning characters
in_dq = False
in_sq = False
dq_start = None
sq_start = None
idx = 0
slen = len(s)
while idx < slen:
    if s[idx:idx+3] == '"""' and not in_sq:
        if not in_dq:
            in_dq = True
            dq_start = s[:idx].count('\n') + 1
            idx += 3
            continue
        else:
            in_dq = False
            dq_start = None
            idx += 3
            continue
    if s[idx:idx+3] == "'''" and not in_dq:
        if not in_sq:
            in_sq = True
            sq_start = s[:idx].count('\n') + 1
            idx += 3
            continue
        else:
            in_sq = False
            sq_start = None
            idx += 3
            continue
    idx += 1
# After scan
if in_dq:
    print('Unclosed double-quote triple starting at line', dq_start)
if in_sq:
    print("Unclosed single-quote triple starting at line", sq_start)
if not in_dq and not in_sq:
    print('No unmatched triple quotes detected')
