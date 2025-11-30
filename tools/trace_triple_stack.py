from pathlib import Path
p=Path('src/emergent/salmon_abm/sockeye_SoA.py')
s=p.read_text()
lines=s.splitlines()
stack=[]
for i,l in enumerate(lines, start=1):
    changed=False
    idx=0
    while True:
        i1=l.find('"""', idx)
        i2=l.find("'''", idx)
        if i1==-1 and i2==-1:
            break
        if i1!=-1 and (i2==-1 or i1<i2):
            tok='"""'
            pos=i1
        else:
            tok="'''"
            pos=i2
        if stack and stack[-1]==tok:
            stack.pop()
            print(i, 'pop', tok, 'stack now', stack)
        else:
            stack.append(tok)
            print(i, 'push', tok, 'stack now', stack)
        idx=pos+3
print('FINAL STACK', stack)
