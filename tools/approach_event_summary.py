#!/usr/bin/env python3
"""Produce a concise per-agent event summary for t=300..500.

Outputs:
 - first evasive (|rud|>=threshold)
 - last evasive before collision_time
 - times of rudder sign flips (instant, and persistent flips)
 - flagged_give_way state change times
"""
import csv
import sys
from collections import defaultdict

IN = sys.argv[1] if len(sys.argv)>1 else 'logs/pid_trace_300_500.csv'
THRESH = float(sys.argv[2]) if len(sys.argv)>2 else 1.0
COLLISION_T = float(sys.argv[3]) if len(sys.argv)>3 else 454.3

rows = []
with open(IN,'r',newline='') as f:
    r = csv.DictReader(f)
    for row in r:
        try:
            t = float(row.get('t','nan'))
        except:
            continue
        try:
            agent = int(row.get('agent',0))
        except:
            agent = 0
        def g(k, default=float('nan')):
            try:
                return float(row.get(k,default))
            except:
                return default
        rows.append({
            't':t,
            'agent':agent,
            'rud':g('rud_deg'),
            'raw_pre':g('raw_preinv_deg'),
            'hd_cmd':g('hd_cmd_deg'),
            'psi':g('psi_deg'),
            'I_raw':g('I_raw_deg'),
            'flag':row.get('flagged_give_way','').strip(),
        })

by_agent=defaultdict(list)
for r in rows:
    by_agent[r['agent']].append(r)

def sign(x):
    return 1 if x>0 else (-1 if x<0 else 0)

print('Approach event summary for',IN,'threshold=',THRESH,'collision_t=',COLLISION_T)
for agent,rs in sorted(by_agent.items()):
    rs = sorted(rs,key=lambda x:x['t'])
    first_evasive=None
    last_before_collision=None
    sign_flips=[]
    last_sign=0
    last_flag=None
    flag_changes=[]
    # For persistent flip detection: require flip sustained for at least persist_s samples (approx sampling dt ~ 0.1s)
    persist_s = 5
    for i,r in enumerate(rs):
        t=r['t']; rud=r['rud']
        s=sign(rud)
        # first evasive
        if first_evasive is None and abs(rud)>=THRESH:
            first_evasive = t
        # last evasive before collision
        if t < COLLISION_T and abs(rud) >= THRESH:
            last_before_collision = t
        # sign flip
        if last_sign!=0 and s!=0 and s!=last_sign:
            # record flip time and neighboring values
            sign_flips.append((t, last_sign, s, rud))
        last_sign = s if s!=0 else last_sign
        # flag changes
        if last_flag is None:
            last_flag = r['flag']
        elif r['flag'] != last_flag:
            flag_changes.append((t, last_flag, r['flag']))
            last_flag = r['flag']

    print('\nAgent',agent)
    print('  rows:',len(rs),'time:',rs[0]['t'],'->',rs[-1]['t'])
    print('  first evasive (|rud|>={:.1f}°): {}'.format(THRESH, first_evasive))
    print('  last evasive before collision (t<{:.2f}): {}'.format(COLLISION_T, last_before_collision))
    if sign_flips:
        print('  sign flips (time, from, to, rud):')
        for sf in sign_flips[:10]:
            print('    t={:.2f} {} -> {} rud={:+.2f}'.format(sf[0],sf[1],sf[2],sf[3]))
    else:
        print('  sign flips: none detected')
    if flag_changes:
        print('  flagged_give_way transitions:')
        for fc in flag_changes:
            print('    t={:.2f} {} -> {}'.format(fc[0],fc[1],fc[2]))
    else:
        print('  flagged_give_way transitions: none in window')
