#!/usr/bin/env python3
"""Analyze logs/pid_trace_window.csv and print concise per-agent diagnostics."""
import csv
from collections import defaultdict
import math

IN='logs/pid_trace_window.csv'

rows=[]
with open(IN,'r',newline='') as f:
    r=csv.DictReader(f)
    for row in r:
        # convert numeric fields
        try:
            row['t']=float(row['t'])
        except:
            continue
        for k in ['err_deg','r_des_deg','derr_deg','P_deg','I_deg','I_raw_deg','D_deg',
                  'raw_preinv_deg','raw_deg','rud_preinv_deg','rud_deg','psi_deg','hd_cmd_deg','r_meas_deg','x_m','y_m']:
            try:
                row[k]=float(row.get(k,''))
            except:
                row[k]=float('nan')
        rows.append(row)

by_agent=defaultdict(list)
for r in rows:
    by_agent[int(r['agent'])].append(r)

summary=[]
for agent,rs in sorted(by_agent.items()):
    max_raw_pre=max(abs(r['raw_preinv_deg']) for r in rs)
    max_raw=max(abs(r['raw_deg']) for r in rs)
    max_rud_pre=max(abs(r['rud_preinv_deg']) for r in rs)
    max_rud=max(abs(r['rud_deg']) for r in rs)
    max_I_raw=max(abs(r['I_raw_deg']) for r in rs if not math.isnan(r['I_raw_deg']))
    events=[(r['t'],r['event']) for r in rs if r['event'].strip()!='']
    # find discrepancies where raw_preinv large but rud_deg small
    discrep=[]
    for r in rs:
        if abs(r['raw_preinv_deg'])>1.0 and abs(r['rud_deg'])<0.1:
            discrep.append((r['t'],r['raw_preinv_deg'],r['rud_deg'],r['event']))
    # find times where rud_preinv != rud_deg by >0.5 deg
    mismatch=[]
    for r in rs:
        if abs(r['rud_preinv_deg'] - r['rud_deg'])>0.5:
            mismatch.append((r['t'],r['rud_preinv_deg'],r['rud_deg'],r['event']))
    summary.append({
        'agent':agent,
        'max_raw_pre_deg':max_raw_pre,
        'max_raw_deg':max_raw,
        'max_rud_pre_deg':max_rud_pre,
        'max_rud_deg':max_rud,
        'max_I_raw_deg':max_I_raw,
        'events':events,
        'discrepancies':discrep,
        'mismatches':mismatch,
        'n_rows':len(rs),
        't_start':rs[0]['t'],
        't_end':rs[-1]['t']
    })

# Print concise report
print('PID window analysis (logs/pid_trace_window.csv)')
for s in summary:
    print('\nAgent',s['agent'])
    print('  rows:',s['n_rows'],'  time:',s['t_start'],'->',s['t_end'])
    print('  max raw_preinv_deg: {:.2f}°  max raw_deg: {:.2f}°'.format(s['max_raw_pre_deg'],s['max_raw_deg']))
    print('  max rud_preinv_deg: {:.2f}°  max rud_deg: {:.2f}°'.format(s['max_rud_pre_deg'],s['max_rud_deg']))
    print('  max I_raw_deg: {:.2f}°'.format(s['max_I_raw_deg']))
    if s['events']:
        print('  events:',s['events'])
    else:
        print('  events: none')
    if s['discrepancies']:
        print('  Potential controller->plant mismatch (raw_cmd present but applied rudder small):')
        for d in s['discrepancies'][:5]:
            print('    t={:.2f}s raw_preinv={:+.2f}° rud_deg={:+.2f}° evt={}'.format(d[0],d[1],d[2],d[3]))
    else:
        print('  No raw->applied suppression detected (within thresholds)')
    if s['mismatches']:
        print('  Times where rud_preinv != rud_deg by >0.5° (possible saturation/backcalc):')
        for m in s['mismatches'][:5]:
            print('    t={:.2f}s rud_preinv={:+.2f}° rud_deg={:+.2f}° evt={}'.format(m[0],m[1],m[2],m[3]))
    else:
        print('  No large rud preinv->applied mismatches')

# Also print the rows around collision time (502.5s) for quick inspection
ct=502.5
print('\nSample rows near collision t~{:.2f}s:'.format(ct))
for r in rows:
    if abs(r['t']-ct)<=0.5:
        print('t={:.2f} ag={} hd_cmd={:.1f} psi={:.1f} err={:.2f} raw_preinv={:.2f} raw={:.2f} rud_preinv={:.2f} rud={:.2f} I_raw={:.2f} evt={}'.format(
            r['t'],int(r['agent']),r['hd_cmd_deg'],r['psi_deg'],r['err_deg'],r['raw_preinv_deg'],r['raw_deg'],r['rud_preinv_deg'],r['rud_deg'],r['I_raw_deg'],r['event']))
