from pathlib import Path
p=Path('src/emergent/salmon_abm/sockeye_SoA.py')
s=p.read_text()
count_double = s.count('"""')
count_single = s.count("'''")
print('double triple quotes:', count_double)
print("single triple quotes:", count_single)
# scan to detect unmatched
stack=[]
lines=s.splitlines()
for i,l in enumerate(lines, start=1):
    # process occurrences left-to-right
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
        else:
            stack.append(tok)
        idx=pos+3
if stack:
    print('UNMATCHED stack:', stack)
else:
    print('All matched')

# print contexts of stack last if present
if stack:
    # find first occurrence of the first unmatched token
    target=stack[0]
    for i,l in enumerate(lines, start=1):
        if target in l:
            start=max(1, i-3)
            end=min(len(lines), i+3)
            print('context around', i)
            for j in range(start, end+1):
                print(j, lines[j-1])
            break
