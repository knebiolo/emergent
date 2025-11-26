import csv
from collections import defaultdict
import math

CSV = r"logs/pid_trace_forced.csv"

rows = []
with open(CSV, newline='') as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append(r)

print(f"Loaded {len(rows)} rows from {CSV}")

agents = sorted({int(r['agent']) for r in rows})
print(f"Agents found: {agents}")

# Find non-empty events
events = [r for r in rows if r.get('event') and r.get('event').strip()]
print(f"Event rows count: {len(events)}")
for r in events[:50]:
    print(f"t={r['t']} agent={r['agent']} event={r['event']} role={r.get('role')} lock={r.get('crossing_lock')} flag={r.get('flagged_give_way')}")

# Find where crossing_lock != -1
locks = [r for r in rows if r.get('crossing_lock') and r.get('crossing_lock').strip() and r.get('crossing_lock') != '-1']
print(f"Rows with crossing_lock != -1: {len(locks)}")
for r in locks[:50]:
    print(f"t={r['t']} agent={r['agent']} lock={r['crossing_lock']} role={r.get('role')} flag={r.get('flagged_give_way')}")

# Find where flagged_give_way not zero/false
flags = [r for r in rows if r.get('flagged_give_way') and r.get('flagged_give_way').strip() and r.get('flagged_give_way') not in ('0','False','false','')]
print(f"Rows with flagged_give_way true: {len(flags)}")
for r in flags[:50]:
    print(f"t={r['t']} agent={r['agent']} flag={r['flagged_give_way']} role={r.get('role')} lock={r.get('crossing_lock')}")

# Find non-neutral roles
non_neutral = [r for r in rows if r.get('role') and r.get('role').strip() and r.get('role') not in ('neutral','')]
print(f"Rows with non-neutral roles: {len(non_neutral)}")
for r in non_neutral[:50]:
    print(f"t={r['t']} agent={r['agent']} role={r['role']} lock={r.get('crossing_lock')} flag={r.get('flagged_give_way')} event={r.get('event')}")

# Rudder saturation events and counts
rud_sat = [r for r in rows if r.get('event') and 'RUD_SAT' in r.get('event')]
print(f"RUD_SAT events: {len(rud_sat)}")
for r in rud_sat[:50]:
    print(f"t={r['t']} agent={r['agent']} event={r['event']} psi={r.get('psi_deg')} hd_cmd={r.get('hd_cmd_deg')} raw={r.get('raw_deg')} rud={r.get('rud_deg')}")

# Show timeline summary per agent for locks/flags
summary = defaultdict(lambda: {'locks':0,'flags':0,'rud_sat':0,'non_neutral':0})
for r in rows:
    a = int(r['agent'])
    if r.get('crossing_lock') and r.get('crossing_lock') != '-1':
        summary[a]['locks'] += 1
    if r.get('flagged_give_way') and r.get('flagged_give_way') not in ('0',''):
        summary[a]['flags'] += 1
    if r.get('event') and 'RUD_SAT' in r.get('event'):
        summary[a]['rud_sat'] += 1
    if r.get('role') and r.get('role') not in ('neutral',''):
        summary[a]['non_neutral'] += 1

print('Per-agent summary:')
for a in agents:
    print(f" agent={a} locks={summary[a]['locks']} flags={summary[a]['flags']} rud_sat={summary[a]['rud_sat']} non_neutral={summary[a]['non_neutral']}")

# If no direct clue, print first 20 rows for visual inspection
if len(events)==0 and len(locks)==0 and len(flags)==0 and len(non_neutral)==0 and len(rud_sat)==0:
    print('\nNo clear events found — printing first 20 rows:')
    for r in rows[:20]:
        print(r)
