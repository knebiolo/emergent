import re
p='src/emergent/salmon_abm/behavior.py'
with open(p,'r',encoding='utf-8') as f:
    lines=f.readlines()
for i,l in enumerate(lines, start=1):
    s=l.rstrip('\n')
    m=re.match(r"^(\s*)(try:|except\b.*:|finally\b.*:)", s)
    if m:
        indent=len(m.group(1))
        tok=m.group(2)
        print(f"{i:5d} indent={indent:2d} {tok}")
