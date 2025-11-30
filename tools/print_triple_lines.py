from pathlib import Path
p=Path('src/emergent/salmon_abm/sockeye_SoA.py')
lines=p.read_text().splitlines()
for i,l in enumerate(lines, start=1):
    if '"""' in l or "'''" in l:
        print(i, repr(l))
print('total lines:', len(lines))
