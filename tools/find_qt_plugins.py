import site, os
found = set()
for p in site.getsitepackages():
    for root, dirs, files in os.walk(p):
        if any(f.lower().endswith('.dll') and f.lower().startswith('q') for f in files) or 'platforms' in root.lower():
            found.add(root)
for f in sorted(found):
    print(f)
