#!/usr/bin/env python3
"""Extract rows from a PID CSV between start_t and end_t and write to an output CSV.

Usage:
  python extract_time_window.py START END [IN_CSV] [OUT_CSV]

Defaults:
  IN_CSV = logs/pid_trace_forced.csv
  OUT_CSV = logs/pid_trace_window.csv
"""
import csv
import sys

IN_DEFAULT = 'logs/pid_trace_forced.csv'
OUT_DEFAULT = 'logs/pid_trace_window.csv'

if len(sys.argv) >= 3:
    try:
        start_t = float(sys.argv[1])
        end_t = float(sys.argv[2])
    except Exception:
        print('Usage: python extract_time_window.py START END [IN_CSV] [OUT_CSV]')
        sys.exit(2)
else:
    # fallback to previous hardcoded window if no args
    start_t = 446.30
    end_t = 462.30

IN = sys.argv[3] if len(sys.argv) >= 4 else IN_DEFAULT
OUT = sys.argv[4] if len(sys.argv) >= 5 else OUT_DEFAULT

count = 0
with open(IN, 'r', newline='') as inf, open(OUT, 'w', newline='') as outf:
    r = csv.reader(inf)
    try:
        header = next(r)
    except StopIteration:
        print('Input file is empty:', IN)
        sys.exit(1)
    w = csv.writer(outf)
    w.writerow(header)
    # find index of t
    try:
        t_idx = header.index('t')
    except ValueError:
        t_idx = 0
    for row in r:
        try:
            t = float(row[t_idx])
        except Exception:
            continue
        if start_t <= t <= end_t:
            w.writerow(row)
            count += 1

print(f'Wrote {count} rows to {OUT} (t in {start_t}..{end_t})')
