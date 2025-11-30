from pathlib import Path

p = Path('src/emergent/salmon_abm/sockeye_SoA.py')
s = p.read_text()
lines = s.splitlines()
occ = []
for i, l in enumerate(lines, start=1):
    if '"""' in l or "'''" in l:
        occ.append((i, l))

print('Found', len(occ), 'triple-quote lines')
for i, l in occ[:40]:
    print(i, l)

# naive parity check
count = s.count('"""') + s.count("'''")
print('Total triple quote tokens:', count)

# find open/close by scanning
stack = []
for i, l in enumerate(lines, start=1):
    if '"""' in l:
        # count occurrences
        for _ in range(l.count('"""')):
            if stack and stack[-1] == '"""':
                stack.pop()
            else:
                stack.append('"""')
    if "'''" in l:
        for _ in range(l.count("'''")):
            if stack and stack[-1] == "'''":
                stack.pop()
            else:
                stack.append("'''")
    if len(stack) > 0 and len(stack) < 5:
        last = stack[-1]
        if last:
            # print first unclosed occurrence context
            start = i - 3
            if start < 1:
                start = 1
            print('--- context around first unmatched at line', i)
            for j in range(start, start+10):
                if j <= len(lines):
                    print(j, lines[j-1])
            break

if stack:
    print('Unclosed triple-quote tokens remain:', stack)
else:
    print('No unclosed triple quotes detected')
