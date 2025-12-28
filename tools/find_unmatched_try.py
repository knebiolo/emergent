import re
p='src/emergent/salmon_abm/behavior.py'
with open(p,'r',encoding='utf-8') as f:
    lines=f.readlines()
stack=[]
for i,l in enumerate(lines, start=1):
    s=l.rstrip('\n')
    # find tokens at line
    m_try=re.match(r"^(\s*)try:\s*(#.*)?$", s)
    m_except=re.match(r"^(\s*)except\b.*:\s*(#.*)?$", s)
    m_finally=re.match(r"^(\s*)finally\b.*:\s*(#.*)?$", s)
    if m_try:
        indent=len(m_try.group(1))
        stack.append(('try', indent, i))
    if m_except or m_finally:
        indent=len((m_except or m_finally).group(1))
        # pop nearest try with same indent
        found=False
        for j in range(len(stack)-1, -1, -1):
            if stack[j][1]==indent and stack[j][0]=='try':
                stack.pop(j)
                found=True
                break
        if not found:
            print('Unmatched except/finally at', i, 'indent', indent)

if stack:
    print('\nUnmatched try blocks remain:')
    for t in stack:
        print(t)
else:
    print('All try matched')
