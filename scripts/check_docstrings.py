from pathlib import Path
p=Path(r"c:\Users\Kevin.Nebiolo\OneDrive - Kleinschmidt Associates\Software\emergent\src\emergent\salmon_abm\sockeye_SoA_OpenGL_RL.py")
s=p.read_text()
print('File:',p)
print('Length:',len(s))
print('triple double quotes count:', s.count('"""'))
print("triple single quotes count:", s.count("'''"))
lines=s.splitlines()
start=2000
end=2060
for i in range(start-1, end):
    print(f"{i+1:5d}: {repr(lines[i])}")
# show context of unclosed quotes by scanning for unclosed triple quotes
for q in ['"""', "'''"]:
    idxs=[]
    i=0
    while True:
        j=s.find(q,i)
        if j==-1: break
        idxs.append(j)
        i=j+len(q)
    print(f"\nPositions of {q}: {len(idxs)} occurrences")
    for k,pos in enumerate(idxs[:10]):
        # show line number
        ln=s[:pos].count('\n')+1
        print(f"  {k+1}: line {ln} pos {pos}")
