#!/usr/bin/env python3
"""Analyze an extracted PID CSV window and detect evasive start and reversal times.

Usage: python analyze_approach_300_500.py [IN_CSV]
"""
import csv
import sys
from collections import defaultdict

IN = sys.argv[1] if len(sys.argv) > 1 else 'logs/pid_trace_300_500.csv'

rows = []
with open(IN, 'r', newline='') as f:
    r = csv.DictReader(f)
    for row in r:
        try:
            row['t'] = float(row.get('t','nan'))
        except:
            continue
        # numeric cast of key fields
        for k in ['raw_preinv_deg','raw_deg','rud_preinv_deg','rud_deg','hd_cmd_deg','psi_deg','I_raw_deg']:
            try:
                row[k] = float(row.get(k,''))
            except:
                row[k] = float('nan')
        rows.append(row)

by_agent = defaultdict(list)
for r in rows:
    by_agent[int(r.get('agent',0))].append(r)

def detect_evasive_and_reversal(rs, rud_threshold_deg=1.0):
    # returns (evasive_start_t, reversal_t, notes)
    evasive = None
    reversal = None
    last_rud = None
    last_sign = 0
    for r in sorted(rs, key=lambda x: x['t']):
        rud = r['rud_deg']
        t = r['t']
        # detect first significant rudder in magnitude
        if evasive is None and abs(rud) >= rud_threshold_deg:
            evasive = t
        if last_rud is not None:
            # detect change in sign after evasive detected
            if evasive is not None and reversal is None:
                # if rud sign flips and magnitude exceeds threshold, mark reversal
                sign = 1 if rud > 0 else (-1 if rud < 0 else 0)
                if last_sign != 0 and sign != 0 and sign != last_sign:
                    reversal = t
                    break
        last_rud = rud
        if rud > 0:
            last_sign = 1
        elif rud < 0:
            last_sign = -1
    notes = []
    return evasive, reversal, notes

print('Analyzing',IN,'rows=',len(rows))
for agent,rs in sorted(by_agent.items()):
    evasive, reversal, notes = detect_evasive_and_reversal(rs, rud_threshold_deg=0.5)
    print('\nAgent',agent)
    print('  rows:',len(rs),'  t:',(rs[0]['t'] if rs else None),'->',(rs[-1]['t'] if rs else None))
    print('  first significant rudder (|rud|>=0.5°):',evasive)
    print('  reversal time (rud sign flip after evasive):',reversal)
    # show sample around evasive and reversal
    if evasive is not None:
        for r in rs:
            if evasive-1.0 <= r['t'] <= (evasive+3.0):
                print('    t={:.2f} rud={:+.2f} raw_preinv={:+.2f} hd_cmd={:+.2f} psi={:+.2f} evt={}'.format(r['t'],r['rud_deg'],r['raw_preinv_deg'],r['hd_cmd_deg'],r['psi_deg'],r.get('event','')))
    if reversal is not None:
        for r in rs:
            if reversal-1.0 <= r['t'] <= (reversal+3.0):
                print('    (reversal) t={:.2f} rud={:+.2f} raw_preinv={:+.2f} hd_cmd={:+.2f} psi={:+.2f} evt={}'.format(r['t'],r['rud_deg'],r['raw_preinv_deg'],r['hd_cmd_deg'],r['psi_deg'],r.get('event','')))

print('\nDone')
