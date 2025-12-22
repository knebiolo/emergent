#!/usr/bin/env python3
"""Analyze the full logs/pid_trace_forced.csv and produce a concise report.
"""
import csv
from collections import defaultdict
import math

IN = 'logs/pid_trace_forced.csv'

# quick reader to collect per-agent stats and event timestamps
per_agent = defaultdict(lambda: {
    'n_rows':0,
    'max_raw_pre':0.0,
    'max_raw':0.0,
    'max_rud_pre':0.0,
    'max_rud':0.0,
    'max_I_raw':0.0,
    'events':[],
    'times':[],
    'roles':[],
    'locks':[],
    'flags':[]
})

collisions = []

with open(IN,'r',newline='') as f:
    r = csv.DictReader(f)
    for row in r:
        try:
            t = float(row.get('t','nan'))
        except:
            continue
        agen = int(row.get('agent',-1))
        try:
            raw_pre = abs(float(row.get('raw_preinv_deg','0') or 0.0))
            raw = abs(float(row.get('raw_deg','0') or 0.0))
            rud_pre = abs(float(row.get('rud_preinv_deg','0') or 0.0))
            rud = abs(float(row.get('rud_deg','0') or 0.0))
            I_raw = abs(float(row.get('I_raw_deg','0') or 0.0))
        except Exception:
            raw_pre,raw,rud_pre,rud,I_raw = 0,0,0,0,0
        per_agent[agen]['n_rows'] += 1
        per_agent[agen]['max_raw_pre'] = max(per_agent[agen]['max_raw_pre'], raw_pre)
        per_agent[agen]['max_raw'] = max(per_agent[agen]['max_raw'], raw)
        per_agent[agen]['max_rud_pre'] = max(per_agent[agen]['max_rud_pre'], rud_pre)
        per_agent[agen]['max_rud'] = max(per_agent[agen]['max_rud'], rud)
        per_agent[agen]['max_I_raw'] = max(per_agent[agen]['max_I_raw'], I_raw)
        ev = (row.get('event') or '').strip()
        if ev:
            per_agent[agen]['events'].append((t,ev))
        per_agent[agen]['times'].append(t)
        per_agent[agen]['roles'].append(row.get('role',''))
        try:
            per_agent[agen]['locks'].append(int(row.get('crossing_lock','-1') or -1))
        except Exception:
            per_agent[agen]['locks'].append(-1)
        try:
            per_agent[agen]['flags'].append(int(row.get('flagged_give_way','0') or 0))
        except Exception:
            per_agent[agen]['flags'].append(0)

# print summary
print('Full PID trace summary:', IN)
for a,stats in sorted(per_agent.items()):
    print('\nAgent',a)
    print('  rows:',stats['n_rows'])
    print('  max raw_preinv_deg: {:.2f}  max raw_deg: {:.2f}'.format(stats['max_raw_pre'], stats['max_raw']))
    print('  max rud_preinv_deg: {:.2f}  max rud_deg: {:.2f}'.format(stats['max_rud_pre'], stats['max_rud']))
    print('  max I_raw_deg: {:.2f}'.format(stats['max_I_raw']))
    if stats['events']:
        print('  events (sample 10):', stats['events'][:10])
    else:
        print('  events: none')
    # detect long intervals of flagged_give_way
    if any(stats['flags']):
        # compute total flagged time as count*dt (approx)
        dt = (stats['times'][1]-stats['times'][0]) if len(stats['times'])>1 else 0.1
        flagged_count = sum(stats['flags'])
        print('  flagged_give_way total (approx): {:.1f}s ({} ticks)'.format(flagged_count*dt, flagged_count))

# quick collision search in run log
import re
LOG = 'logs/run_ship_Seattle_gui.out'
try:
    with open(LOG,'r',encoding='utf-8',errors='ignore') as f:
        for line in f:
            if 'Collision: Ships' in line:
                m = re.search(r'at t=([0-9\.]+)s', line)
                if m:
                    collisions.append(float(m.group(1)))
except Exception:
    pass

if collisions:
    print('\nCollisions found in run log (times):')
    for c in collisions[:20]:
        print(' ',c)
else:
    print('\nNo collisions found in run log')

print('\nDone')
