import sys
p='src/emergent/salmon_abm/behavior.py'
start= int(sys.argv[1]) if len(sys.argv)>1 else 1
end = int(sys.argv[2]) if len(sys.argv)>2 else start+50
with open(p,'r',encoding='utf-8') as f:
    lines=f.readlines()
for i in range(start-1, min(end, len(lines))):
    print(f"{i+1:5d}: {lines[i].rstrip()}")
