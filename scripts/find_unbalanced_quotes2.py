from pathlib import Path
p = Path(r"c:\Users\Kevin.Nebiolo\OneDrive - Kleinschmidt Associates\Software\emergent\src\emergent\salmon_abm\sockeye_SoA_OpenGL_RL.py")
s = p.read_text()
print('File:', p)
DD = '"""'
SS = "'''"
# find all occurrences
occ = []
i = 0
while True:
    di = s.find(DD, i)
    si = s.find(SS, i)
    if di == -1 and si == -1:
        break
    if di != -1 and (si == -1 or di < si):
        occ.append((di, DD))
        i = di + 3
    else:
        occ.append((si, SS))
        i = si + 3

print('Total occurrences:', len(occ))
# simulate stack
stack = []
unmatched = []
for pos, ty in occ:
    if stack and stack[-1][1] == ty:
        stack.pop()
    else:
        stack.append((pos, ty))
# After processing, any remaining on stack are unmatched opens
if not stack:
    print('All triple-quote blocks appear balanced (by simple pairing).')
else:
    print('Unmatched triple-quote openings:')
    for pos, ty in stack:
        ln = s[:pos].count('\n') + 1
        print(f"  type={ty} pos={pos} line={ln}")
        # show context
        lines = s.splitlines()
        start = max(0, ln - 10)
        end = min(len(lines), ln + 10)
        print('--- context ---')
        for i in range(start, end):
            print(f"{i+1:5d}: {lines[i]}")
        print('---------------')

# Also show the 2030-2040 lines
lines = s.splitlines()
print('\nLines 2028..2042:')
for i in range(2027, 2042):
    print(f"{i+1:5d}: {lines[i]}")
