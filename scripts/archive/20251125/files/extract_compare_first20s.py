import csv
from pathlib import Path

root = Path(__file__).resolve().parents[1]
input_dir = root / 'sweep_results'
files = {
    'baseline': input_dir / 'pid_trace_baseline_straight.csv',
    'candidate': input_dir / 'pid_trace_validate_straight_nd1p25.csv',
    'post': input_dir / 'pid_trace_quick_short.csv',
}

# columns we want to extract from each input
cols = ['t', 'psi_deg', 'hd_cmd_deg', 'err_deg', 'rud_deg']

# read each file into dict keyed by t (string representation)
data = {}
for key, path in files.items():
    with path.open('r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                t = float(row['t'])
            except Exception:
                continue
            if t > 20.0:
                break
            k = f"{t:.3f}"
            if k not in data:
                data[k] = {'t': f"{t:.3f}"}
            # copy selected cols, prefix with source key
            for c in cols[1:]:
                data[k][f"{key}_{c}"] = row.get(c, '')

# sort by time and write combined CSV
out = input_dir / 'compare_first20s_straight.csv'
fieldnames = ['t']
for key in ['baseline', 'candidate', 'post']:
    for c in cols[1:]:
        fieldnames.append(f"{key}_{c}")

with out.open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for k in sorted(data.keys(), key=lambda x: float(x)):
        row = {fn: data[k].get(fn, '') for fn in fieldnames}
        writer.writerow(row)

print('wrote', out)
