"""Detect cumulative heading rotation (spins) by unwrapping hd_cur in [CTRL] lines.
Print candidate windows where absolute cumulative change over a rolling window exceeds 360°.
"""
import re
from math import fmod

log_path = r"logs/run_ship_Seattle_chicken.out"
ctrl_re = re.compile(r"\[CTRL\].*hd_cur=(?P<hd_cur>[-0-9\.]+)°.*t=(?P<t>[0-9\.]+)s")
ctrl_re2 = re.compile(r"\[CTRL\].*t=(?P<t>[0-9\.]+)s .*hd_cur=(?P<hd_cur>[-0-9\.]+)°")
# fallback
ctrl_re3 = re.compile(r"\[CTRL\].*hd_cur=(?P<hd_cur>[-0-9\.]+)°")

headings = []
with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
    for i, line in enumerate(f):
        m = ctrl_re.search(line) or ctrl_re2.search(line) or ctrl_re3.search(line)
        if m:
            try:
                hd_cur = float(m.group('hd_cur'))
            except Exception:
                continue
            t = float(m.group('t')) if 't' in m.groupdict() and m.group('t') is not None else None
            headings.append((i+1, t, hd_cur, line.rstrip('\n')))

print(f"Read {len(headings)} heading samples")

# unwrap headings
unwrap = []
last = None
offset = 0.0
for idx, t, h, raw in headings:
    if last is None:
        last = h
        unwrap.append((idx, t, h))
        continue
    dh = h - last
    # adjust dh to be between -180,180
    while dh <= -180:
        dh += 360
    while dh > 180:
        dh -= 360
    offset += dh
    unwrapped = headings[0][2] + offset
    unwrap.append((idx, t, unwrapped))
    last = h

# sliding window cumulative absolute change
WINDOW_S = 30.0  # seconds window to check
threshold = 360.0
candidates = []
for i, (idx, t, u) in enumerate(unwrap):
    if t is None:
        continue
    # find j where t - t_j <= WINDOW_S
    j = i
    while j > 0 and unwrap[i][1] - unwrap[j-1][1] <= WINDOW_S:
        j -= 1
    if j < i:
        delta = abs(u - unwrap[j][2])
        if delta >= threshold:
            candidates.append((unwrap[j][0], unwrap[i][0], unwrap[j][1], unwrap[i][1], delta))

print(f"Found {len(candidates)} cumulative-spin candidates (window {WINDOW_S}s, threshold {threshold}°)")

if candidates:
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    for k, (start_ln, end_ln, t0, t1, d) in enumerate(candidates[:20]):
        s = max(1, start_ln - 20)
        e = min(len(lines), end_ln + 20)
        print(f"\n=== Candidate {k+1}: lines {s}-{e} t={t0}-{t1} d={d:.1f}° ===")
        for ln in range(s, e+1):
            print(f"{ln:6d}: {lines[ln-1].rstrip()}")
else:
    print("No cumulative spins detected")
