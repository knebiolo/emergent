#!/usr/bin/env python3
"""Filter correlated PID events CSV to only en-route (t <= arrival_time).

Usage: python scripts/filter_correlated_enroute.py <in.csv> <arrival_time> [out.csv]
"""
import sys, csv
if len(sys.argv) < 3:
    print(__doc__)
    sys.exit(2)
inpath = sys.argv[1]
arrival = float(sys.argv[2])
outpath = sys.argv[3] if len(sys.argv) > 3 else inpath.replace('.csv', '_enroute.csv')
count_total = 0
count_kept = 0
with open(inpath, 'r', encoding='utf-8') as fh:
    rdr = csv.reader(fh)
    try:
        header = next(rdr)
    except StopIteration:
        print('Empty input file')
        sys.exit(2)
    rows = []
    for row in rdr:
        count_total += 1
        if not row:
            continue
        try:
            t = float(row[0])
        except Exception:
            continue
        if t <= arrival:
            rows.append(row)
            count_kept += 1
with open(outpath, 'w', encoding='utf-8', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(header)
    w.writerows(rows)
print(f'Filtered {count_kept}/{count_total} rows (t<={arrival}) -> {outpath}')
