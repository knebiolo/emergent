p='src/emergent/salmon_abm/behavior.py'
with open(p,'r',encoding='utf-8') as f:
    lines=f.readlines()
for i in range(1440,1500):
    line=lines[i-1]
    leading=len(line)-len(line.lstrip(' '))
    print(f"{i:4d}: indent={leading:2d} |{line.rstrip()}|")
