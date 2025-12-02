from pathlib import Path
p=Path(r"c:\Users\Kevin.Nebiolo\OneDrive - Kleinschmidt Associates\Software\emergent\src\emergent\salmon_abm\sockeye_SoA_OpenGL_RL.py")
s=p.read_text()

positions=[]
state=None  # None, 'd' for triple double, "s" for triple single
i=0
while i<len(s):
    if s.startswith('"""', i):
        positions.append((i, '"""'))
        if state is None:
            state='d'
            start=i
        elif state=='d':
            state=None
            end=i
        else:
            # inside single but found double triple
            positions.append(('conflict', i))
        i+=3
    elif s.startswith("'''", i):
        positions.append((i, "''""))
        if state is None:
            state='s'
            start=i
        elif state=='s':
            state=None
            end=i
        else:
            positions.append(('conflict', i))
        i+=3
    else:
        i+=1

print('Final state:', state)
print('Total triple-double occurrences:', s.count('"""'))
print('Total triple-single occurrences:', s.count("'''"))
# show around the pycompile error line ~2037
lines=s.splitlines()
for ln in range(2015, 2045):
    print(f"{ln+1}: {lines[ln]}")

# If unclosed, show earlier triple quotes
if state is not None:
    print('Unclosed triple quote type:', state)
    # find last occurrence of that quote type
    q='"""' if state=='d' else "'''"
    last = s.rfind(q)
    print('Last occurrence pos', last, 'line', s[:last].count('\n')+1)
else:
    print('All triple quotes appear balanced (by toggle heuristic)')
