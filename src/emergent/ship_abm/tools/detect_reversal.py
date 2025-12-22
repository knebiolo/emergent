#!/usr/bin/env python3
"""Detect first evasive rudder sign and persistent reversal times per agent.

Usage: python detect_reversal.py [IN_CSV] [threshold_deg] [persist_seconds]
Defaults: IN_CSV=logs/pid_trace_300_500.csv, threshold_deg=0.5, persist_seconds=2.0
"""
import csv
import sys
from collections import defaultdict

IN = sys.argv[1] if len(sys.argv) > 1 else 'logs/pid_trace_300_500.csv'
THRESH = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
PERSIST = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0

rows = []
with open(IN, 'r', newline='') as f:
    r = csv.DictReader(f)
    for row in r:
        try:
            t = float(row.get('t', 'nan'))
            agent = int(row.get('agent', 0))
            rud = float(row.get('rud_deg', 'nan'))
        except Exception:
            continue
        rows.append({'t': t, 'agent': agent, 'rud': rud, 'hd_cmd': row.get('hd_cmd_deg',''), 'psi': row.get('psi_deg',''), 'event': row.get('event',''), 'flagged_give_way': row.get('flagged_give_way','')})

by_agent = defaultdict(list)
for r in rows:
    by_agent[r['agent']].append(r)

def find_persistent_reversal(rs, thresh, persist):
    # rs sorted by t
    rs = sorted(rs, key=lambda x: x['t'])
    first_evasive_t = None
    first_sign = 0
    # find first time |rud|>=thresh
    for r in rs:
        if abs(r['rud']) >= thresh:
            first_evasive_t = r['t']
            first_sign = 1 if r['rud'] > 0 else -1
            break
    if first_evasive_t is None:
        return None, None, None

    # find last time where sign == first_sign and |rud|>=thresh
    last_same = None
    for r in rs:
        if r['t'] < first_evasive_t:
            continue
        if abs(r['rud']) >= thresh and (1 if r['rud']>0 else -1) == first_sign:
            last_same = r['t']

    # find first time where opposite sign persists for at least `persist` seconds
    reversal_time = None
    for i, r in enumerate(rs):
        if r['t'] <= (last_same or first_evasive_t):
            continue
        sign = 1 if r['rud']>0 else (-1 if r['rud']<0 else 0)
        if sign == 0 or sign == first_sign:
            continue
        # candidate start
        t0 = r['t']
        # check persistence
        t_end_needed = t0 + persist
        ok = True
        for r2 in rs:
            if r2['t'] < t0:
                continue
            if r2['t'] > t_end_needed:
                break
            s2 = 1 if r2['rud']>0 else (-1 if r2['rud']<0 else 0)
            if s2 != sign:
                ok = False
                break
        if ok:
            reversal_time = t0
            break

    return first_evasive_t, last_same, reversal_time

print('Detect reversal in',IN,'threshold=',THRESH,'persist_s=',PERSIST)
for agent, rs in sorted(by_agent.items()):
    fe, last_same, rev = find_persistent_reversal(rs, THRESH, PERSIST)
    print('\nAgent',agent)
    print('  rows:',len(rs),'  t:',rs[0]['t'],'->',rs[-1]['t'])
    print('  flagged_give_way sample:', next((r['flagged_give_way'] for r in rs if r.get('flagged_give_way')!=''), ''))
    if fe is None:
        print('  No significant evasive rudder (|rud|<{})'.format(THRESH))
        continue
    print('  first significant evasive at t={:.2f}s'.format(fe))
    if last_same is not None:
        print('  last time with that evasive sign at t={:.2f}s'.format(last_same))
    else:
        print('  no later times with same sign found')
    if rev is not None:
        print('  persistent reversal detected at t={:.2f}s (>= {:.1f}s)'.format(rev, PERSIST))
    else:
        print('  no persistent reversal detected after last_same (within window)')
