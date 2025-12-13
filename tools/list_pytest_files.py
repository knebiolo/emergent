from pathlib import Path
root=Path('.')
search_dirs=[root/'tests', root/'src', root/'tools']
files=[]
for d in search_dirs:
    if d.exists():
        for p in d.rglob('test_*.py'):
            files.append(str(p))
files=sorted(set(files))
print(len(files))
for i,f in enumerate(files):
    print(i, f)
