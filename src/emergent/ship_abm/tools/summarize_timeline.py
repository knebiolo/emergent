#!/usr/bin/env python3
import csv,math,sys,re

CSV='logs/pid_trace_300_500.csv'
RUNLOG='logs/run_ship_Seattle_gui.out'
START=300.0
END=500.0
COLL_T=454.3
TH=1.0

by_agent={}
with open(CSV,'r',newline='') as f:
    r=csv.DictReader(f)
    for row in r:
        try:
            t=float(row.get('t','nan'))
        except:
            continue
        if t<START-1e-6 or t>END+1e-6: continue
        a=int(row.get('agent') or 0)
        try:
            rud=float(row.get('rud_deg') or 'nan')
        except:
            rud=math.nan
        hd=float(row.get('hd_cmd_deg') or 'nan')
        psi=float(row.get('psi_deg') or 'nan')
        I_raw=float(row.get('I_raw_deg') or 'nan')
        flag=row.get('flagged_give_way','').strip()
        by_agent.setdefault(a,[]).append({'t':t,'rud':rud,'hd':hd,'psi':psi,'I_raw':I_raw,'flag':flag})

print('\nPer-agent timeline summary (threshold {:.1f}°, collision {:.2f}s)'.format(TH,COLL_T))
for a in sorted(by_agent.keys()):
    arr=sorted(by_agent[a],key=lambda x:x['t'])
    first=None
    last_before_coll=None
    for it in arr:
        if first is None and not math.isnan(it['rud']) and abs(it['rud'])>=TH:
            first=it['t']
        if it['t']<COLL_T and not math.isnan(it['rud']) and abs(it['rud'])>=TH:
            last_before_coll=it['t']
    # nearest to collision
    best=None; bdt=1e9
    for it in arr:
        dt=abs(it['t']-COLL_T)
        if dt<bdt:
            best=it; bdt=dt
    print('\nAgent',a)
    print('  rows:',len(arr),'time: {:.2f}-> {:.2f}'.format(arr[0]['t'],arr[-1]['t']))
    print('  first evasive (|rud|>={:.1f}°): {}'.format(TH, first))
    print('  last evasive before {:.2f}s: {}'.format(COLL_T, last_before_coll))
    if best:
        print("  nearest row to {:.2f}s -> t={:.2f} rud={:+.2f} hd_cmd={:+.2f} psi={:+.2f} I_raw={:+.2f} flag='{}'".format(COLL_T, best['t'], best['rud'], best['hd'], best['psi'], best['I_raw'], best['flag']))

# scan run log for flagged_give_way lines in [440,460]
print('\nRun-log flagged_give_way lines in 440..460s:')
text=open(RUNLOG,'rb').read()
# decode robustly
if text.startswith(b'\xff\xfe') or text.startswith(b'\xfe\xff'):
    s=text.decode('utf-16',errors='ignore')
else:
    try:
        s=text.decode('utf-8')
    except:
        s=text.decode('utf-8',errors='ignore')

lines=s.splitlines()
pat=re.compile(r'flagged_give_way',re.IGNORECASE)
for L in lines:
    if pat.search(L):
        # try to extract nearby numeric time if present on same or next lines
        if 't=' in L:
            print(' ',L)
            continue
        # some flagged lines are multi-line; just print if they are in the collision band text nearby
        # if the line includes a distance/time numeric we attempt to show it
        print(' ',L)

print('\n(End of summary)')
