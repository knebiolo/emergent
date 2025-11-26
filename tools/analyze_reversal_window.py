#!/usr/bin/env python3
import csv,math
path='logs/pid_trace_forced.csv'
start=440.0
end=460.0
agents_data={}
with open(path,'r',newline='',encoding='utf-8',errors='ignore') as f:
    r=csv.DictReader(f)
    for row in r:
        try:
            t=float(row.get('t','nan'))
        except:
            continue
        if t<start-1e-6 or t> end+1e-6: continue
        a=int(row.get('agent') or 0)
        hd=float(row.get('hd_cmd_deg') or 'nan')
        rud=float(row.get('rud_deg') or 'nan')
        raw=float(row.get('raw_deg') or 'nan')
        I_raw=float(row.get('I_raw_deg') or 'nan')
        agents_data.setdefault(a,[]).append({'t':t,'hd':hd,'rud':rud,'raw':raw,'I_raw':I_raw})

for a,arr in sorted(agents_data.items()):
    print('\nAgent',a,'rows',len(arr))
    # detect rud sign flips
    last_rud=None
    last_hd=None
    flips=[]
    hd_jumps=[]
    for it in arr:
        if last_rud is None:
            last_rud=it['rud']
        else:
            if not math.isnan(it['rud']) and not math.isnan(last_rud):
                if (last_rud==0 and abs(it['rud'])>0.5) or (last_rud*it['rud']<0 and abs(it['rud'])>0.5):
                    flips.append((it['t'], last_rud, it['rud']))
            last_rud=it['rud']
        if last_hd is None:
            last_hd=it['hd']
        else:
            if not math.isnan(it['hd']) and not math.isnan(last_hd):
                d=it['hd']-last_hd
                if abs(d)>30: # large heading command jump >30deg
                    hd_jumps.append((it['t'], last_hd, it['hd'], d))
            last_hd=it['hd']
    print('  RUD sign flips (>0.5deg change and sign change):',len(flips))
    for f in flips[:10]:
        print('   flip at t={:.3f}: rud_prev={:+.3f} rud_now={:+.3f}'.format(*f))
    print('  Large HD_CMD jumps (>30deg):',len(hd_jumps))
    for h in hd_jumps[:10]:
        print('   jump at t={:.3f}: hd_prev={:+.2f} hd_now={:+.2f} delta={:+.2f}'.format(*h))
    # print last few rows
    print('  last 5 rows:')
    for it in arr[-5:]:
        print('   t={:.3f} hd={:+.3f} rud={:+.3f} raw={:+.3f} I_raw={:+.3f}'.format(it['t'],it['hd'],it['rud'],it['raw'],it['I_raw']))
