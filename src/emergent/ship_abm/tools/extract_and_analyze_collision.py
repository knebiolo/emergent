#!/usr/bin/env python3
"""Find earliest collision time in run log, extract PID rows around it, and print diagnostics.
"""
import re, csv
from collections import defaultdict

LOG = 'logs/run_ship_Seattle_gui.out'
PID = 'logs/pid_trace_forced.csv'
OUT = 'logs/pid_trace_collision_start_window.csv'
WINDOW_S = 8.0

# find earliest collision time
coll_times = []
with open(LOG,'r',encoding='utf-8',errors='ignore') as f:
    for line in f:
        if 'Collision: Ships' in line:
            m = re.search(r'at t=([0-9\.]+)s', line)
            if m:
                coll_times.append(float(m.group(1)))
if not coll_times:
    print('No collisions found in run log')
    raise SystemExit(1)

start_t = min(coll_times)
print(f'Earliest collision at t={start_t:.2f}s; extracting ±{WINDOW_S}s window -> {start_t-WINDOW_S:.2f}..{start_t+WINDOW_S:.2f}s')

# read PID trace header
with open(PID,'r',newline='') as f:
    rdr = csv.DictReader(f)
    header = rdr.fieldnames
    rows = [r for r in rdr if r.get('t')]

# filter rows in window
win_rows = []
for r in rows:
    try:
        t = float(r['t'])
    except:
        continue
    if (t >= start_t - WINDOW_S) and (t <= start_t + WINDOW_S):
        win_rows.append(r)

if not win_rows:
    print('No PID rows found in window')
    raise SystemExit(1)

# write out
with open(OUT,'w',newline='') as f:
    w = csv.DictWriter(f, fieldnames=header)
    w.writeheader()
    for r in win_rows:
        w.writerow(r)

print(f'Wrote {len(win_rows)} rows to {OUT}')

# analyze per-agent in window
by_agent = defaultdict(list)
for r in win_rows:
    try:
        a = int(r.get('agent',-1))
    except:
        a = -1
    by_agent[a].append(r)

for a,rs in sorted(by_agent.items()):
    print(f'\nAgent {a}  rows={len(rs)}')
    max_raw_pre = max(abs(float(r.get('raw_preinv_deg') or 0.0)) for r in rs)
    max_raw = max(abs(float(r.get('raw_deg') or 0.0)) for r in rs)
    max_rud_pre = max(abs(float(r.get('rud_preinv_deg') or 0.0)) for r in rs)
    max_rud = max(abs(float(r.get('rud_deg') or 0.0)) for r in rs)
    max_I = max(abs(float(r.get('I_raw_deg') or 0.0)) for r in rs)
    events = [(float(r['t']), (r.get('event') or '').strip()) for r in rs if (r.get('event') or '').strip()]
    roles = [r.get('role','') for r in rs]
    locks = [r.get('crossing_lock','') for r in rs]
    flags = [r.get('flagged_give_way','') for r in rs]
    print(f'  max raw_preinv_deg={max_raw_pre:.2f}  max raw_deg={max_raw:.2f}')
    print(f'  max rud_preinv_deg={max_rud_pre:.2f}  max rud_deg={max_rud:.2f}')
    print(f'  max I_raw_deg={max_I:.2f}')
    print(f'  events: {events[:10]}')
    # role / lock summary
    from collections import Counter
    print('  role counts:', Counter(roles))
    print('  lock counts:', Counter(locks))
    print('  flagged_give_way counts:', Counter(flags))

# print some sample rows around start_t
print('\nSample rows near collision start:')
ct = start_t
for r in win_rows:
    t = float(r['t'])
    if abs(t - ct) <= 1.0:
        print(f"t={t:.2f} ag={r.get('agent')} hd_cmd={r.get('hd_cmd_deg')} psi={r.get('psi_deg')} err={r.get('err_deg')} raw_preinv={r.get('raw_preinv_deg')} raw={r.get('raw_deg')} rud_preinv={r.get('rud_preinv_deg')} rud={r.get('rud_deg')} I_raw={r.get('I_raw_deg')} event={r.get('event')} role={r.get('role')} lock={r.get('crossing_lock')} flag={r.get('flagged_give_way')}")

print('\nDone')
