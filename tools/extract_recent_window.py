"""
Extract recent time window rows for agent 1 from the simulator CSV logs.
Usage: python tools/extract_recent_window.py [seconds]
Default seconds=50
"""
import csv
import sys
import os

WINDOW_S = float(sys.argv[1]) if len(sys.argv) > 1 else 50.0
LOG_DIR = os.path.join(os.path.dirname(__file__), '..', 'logs')
LOG_DIR = os.path.normpath(LOG_DIR)

files = [
    ('logs/pid_mismatch_debug.csv', 't', 'pid_mismatch_debug.csv'),
    ('logs/pid_deep_debug.csv', 't', 'pid_deep_debug.csv'),
    ('logs/pid_runtime_debug.csv', 't', 'pid_runtime_debug.csv'),
    ('logs/colregs_runtime_debug.csv', 'sim_time', 'colregs_runtime_debug.csv'),
]

def read_csv(path):
    try:
        with open(path, 'r', newline='') as f:
            reader = csv.reader(f)
            rows = list(reader)
            return rows
    except FileNotFoundError:
        return None

all_outputs = []
for relpath, time_col, label in files:
    path = os.path.join(os.path.dirname(__file__), '..', relpath)
    path = os.path.normpath(path)
    rows = read_csv(path)
    if not rows:
        all_outputs.append((label, None, f'File not found: {path}'))
        continue
    header = rows[0]
    # find time column index
    try:
        t_idx = header.index(time_col)
    except ValueError:
        # try common alternatives
        alt = None
        for cand in ['t', 'sim_time', 'time']:
            if cand in header:
                alt = cand
                break
        if alt:
            t_idx = header.index(alt)
        else:
            all_outputs.append((label, None, f'No time column ({time_col}) in {path}'))
            continue
    # find agent column index
    agent_idx = None
    for cand in ['agent', 'Agent', 'agent_id']:
        if cand in header:
            agent_idx = header.index(cand)
            break
    if agent_idx is None:
        # fallback to column 1 if present
        agent_idx = 1 if len(header) > 1 else None
    # parse numeric times and find max
    times = []
    parsed_rows = []
    for r in rows[1:]:
        if len(r) <= t_idx:
            continue
        t_raw = r[t_idx].strip()
        try:
            t = float(t_raw)
        except Exception:
            continue
        times.append(t)
        parsed_rows.append((t, r))
    if not times:
        all_outputs.append((label, header, 'No numeric time rows'))
        continue
    max_t = max(times)
    threshold = max_t - WINDOW_S
    # filter rows for agent '1' where t >= threshold
    filtered = [r for (t,r) in parsed_rows if t >= threshold and (agent_idx is None or (len(r) > agent_idx and r[agent_idx].strip() in ('1', '1.0')))]
    # sort by time
    filtered_sorted = sorted(filtered, key=lambda r: float(r[t_idx]))
    all_outputs.append((label, header, filtered_sorted))

# Print outputs
for label, header, payload in all_outputs:
    print('\n---', label, '---')
    if payload is None:
        print('MISSING')
        continue
    if isinstance(payload, str):
        print(payload)
        continue
    print('HEADER:', header)
    for row in payload[:1000]:
        print(row)
    if len(payload) > 1000:
        print(f'...truncated after {len(payload)} rows')

print('\nEXTRACTION_DONE')
