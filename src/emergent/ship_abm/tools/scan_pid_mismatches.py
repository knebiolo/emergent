"""
Scan logs/pid_mismatch_debug.csv for mismatch rows (match==0) and print them.
"""
import csv, os, sys
path = os.path.join(os.path.dirname(__file__), '..', 'logs', 'pid_mismatch_debug.csv')
path = os.path.normpath(path)
if not os.path.exists(path):
    print('MISMATCH_FILE_MISSING')
    sys.exit(0)
count = 0
with open(path, 'r', newline='') as fh:
    reader = csv.reader(fh)
    hdr = next(reader, None)
    print('HEADER:', hdr)
    for r in reader:
        # expect match column name 'match' in header
        try:
            idx = hdr.index('match')
        except ValueError:
            # fallback: assume column 5 (0-based)
            idx = 5 if len(r) > 5 else None
        if idx is None:
            break
        if len(r) <= idx:
            continue
        try:
            m = int(float(r[idx]))
        except Exception:
            # non-numeric, skip
            continue
        if m == 0:
            print(r)
            count += 1
            if count >= 200:
                print('...truncated after 200 rows')
                break
print('FOUND', count, 'mismatches')
