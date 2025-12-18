from pathlib import Path
p = Path('outputs')
print('cwd:', Path('.').resolve())
print('outputs exists:', p.exists())
if p.exists():
    for f in sorted(p.iterdir()):
        try:
            print(' -', f.name, 'size', f.stat().st_size)
        except Exception as e:
            print(' -', f.name, 'stat error', e)
else:
    print('No outputs directory.')
