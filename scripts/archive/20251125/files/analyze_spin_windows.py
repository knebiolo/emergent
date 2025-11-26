"""Quick log scanner to find large heading jumps and sustained spins.

Usage: run in the repository root; reads logs/run_ship_Seattle_chicken.out
"""
import re
from collections import deque

log_path = r"logs/run_ship_Seattle_chicken.out"
# regex to match ctrl lines with hd_cur, hd_cmd
ctrl_re = re.compile(r"\[CTRL\] t=(?P<t>[0-9\.]+)s .*hd_cmd=(?P<hd_cmd>[-0-9\.]+)°.*, hd_cur=(?P<hd_cur>[-0-9\.]+)°.*rud=(?P<rud>[-0-9\.]+)")
# fallback looser regex
ctrl_re2 = re.compile(r"\[CTRL\].*hd_cmd=(?P<hd_cmd>[-0-9\.]+)°.*, hd_cur=(?P<hd_cur>[-0-9\.]+)°.*rud=(?P<rud>[-0-9\.]+)")

headings = []
with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
    for i, line in enumerate(f):
        m = ctrl_re.search(line)
        if not m:
            m = ctrl_re2.search(line)
        if m:
            t = float(m.group('t')) if 't' in m.groupdict() else None
            hd_cmd = float(m.group('hd_cmd'))
            hd_cur = float(m.group('hd_cur'))
            rud = float(m.group('rud'))
            headings.append((i+1, t, hd_cmd, hd_cur, rud, line.rstrip('\n')))

print(f"Found {len(headings)} ctrl lines")

# Scan for large deltas in hd_cur between consecutive ctrl lines (>100 deg within 1s)
candidates = []
for a, b in zip(headings, headings[1:]):
    i1, t1, _, hd1, _ = a[0], a[1], a[2], a[3], a[4]
    i2, t2, _, hd2, _ = b[0], b[1], b[2], b[3], b[4]
    if t1 is None or t2 is None:
        continue
    dt = t2 - t1
    d = abs(((hd2 - hd1 + 180) % 360) - 180)  # wrap delta
    if d > 90 and dt < 5.0:
        candidates.append((i1, t1, i2, t2, d, dt))

print(f"Found {len(candidates)} large-heading-delta candidates")
# print contexts for candidates
LINES_BEFORE = 10
LINES_AFTER = 10
if candidates:
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        all_lines = f.readlines()
    for idx, cand in enumerate(candidates[:20]):
        i1, t1, i2, t2, d, dt = cand
        start = max(1, i1 - LINES_BEFORE)
        end = min(len(all_lines), i2 + LINES_AFTER)
        print("\n--- Candidate %d: lines %d-%d dt=%.2fs d=%.1f° ---" % (idx+1, start, end, dt, d))
        for ln in range(start, end+1):
            print(f"{ln:6d}: {all_lines[ln-1].rstrip()}")
else:
    print("No large heading jumps detected")
