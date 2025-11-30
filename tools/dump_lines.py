from pathlib import Path
p=Path('src/emergent/salmon_abm/sockeye_SoA.py')
lines=p.read_text().splitlines()
for i in range(8184,8206):
    print(f'{i:5d}: {lines[i-1]!r}')
