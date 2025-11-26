#!/usr/bin/env python3
import csv,sys,math
path='logs/pid_trace_forced.csv'
start=440.0
end=460.0
cols=['t','agent','hd_cmd_deg','raw_deg','rud_preinv_deg','rud_deg','I_raw_deg','psi_deg','flagged_give_way','crossing_lock','event']
print('Reading',path,'window',start,end)
with open(path,'r',newline='',encoding='utf-8',errors='ignore') as f:
    r=csv.DictReader(f)
    out=[]
    for row in r:
        try:
            t=float(row.get('t','nan'))
        except:
            continue
        if t<start-1e-6 or t> end+1e-6: continue
        out.append(row)
    if not out:
        print('No rows found in window')
        sys.exit(0)
    print('Found',len(out),'rows')
    # print header
    hdr=' | '.join([c.ljust(12) for c in cols])
    print(hdr)
    print('-'*len(hdr))
    for row in out:
        vals=[]
        for c in cols:
            v=row.get(c,'')
            if v is None: v=''
            if len(v)>12: v=v[:12]
            vals.append(str(v).ljust(12))
        print(' | '.join(vals))
    # print nearest to collision time 454.3
    tgt=454.3
    best=min(out,key=lambda r:abs(float(r.get('t') or 1e9)-tgt))
    print('\nNearest row to {:.2f}s:'.format(tgt))
    for c in cols:
        print(' ',c,':',best.get(c,''))
