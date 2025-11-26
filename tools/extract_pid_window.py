#!/usr/bin/env python3
"""tools/extract_pid_window.py
Extract a time window from logs/pid_trace_forced.csv and write logs/pid_trace_window.csv.
Usage: python tools/extract_pid_window.py [start_time] [end_time]
If no args provided defaults to 497.6 -> 505.6 (±40 ticks around 501.6s collision).
"""
import csv
import sys

def main():
    inpath = 'logs/pid_trace_forced.csv'
    outpath = 'logs/pid_trace_window.csv'
    if len(sys.argv) >= 3:
        lo = float(sys.argv[1])
        hi = float(sys.argv[2])
    else:
        lo = 497.6
        hi = 505.6
    count = 0
    with open(inpath, 'r', newline='') as inf, open(outpath, 'w', newline='') as outf:
        r = csv.reader(inf)
        w = csv.writer(outf)
        header = next(r)
        w.writerow(header)
        for row in r:
            try:
                t = float(row[0])
            except Exception:
                continue
            if t >= lo and t <= hi:
                w.writerow(row)
                count += 1
    print(f'Wrote {count} rows to {outpath} (time window {lo} - {hi} s)')

if __name__ == '__main__':
    main()
