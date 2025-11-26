"""
Scan all CSV traces in traces/ and report max |rud_deg| and max psi delta/sec per agent per file.
"""
import glob
import csv
from collections import defaultdict


def wrap_deg(x):
    return ((x + 180) % 360) - 180


def heading_diff_deg(a, b):
    return wrap_deg(a - b)


def analyze_file(path):
    agents = defaultdict(list)
    with open(path, 'r', newline='') as fh:
        reader = csv.reader(fh)
        header = next(reader)
        idx = {k: i for i, k in enumerate(header)}
        for row in reader:
            try:
                t = float(row[idx['t']])
                agent = int(row[idx['agent']])
                rud_deg = float(row[idx['rud_deg']])
                psi_deg = float(row[idx['psi_deg']])
                agents[agent].append({'t': t, 'rud_deg': rud_deg, 'psi_deg': psi_deg})
            except Exception:
                continue
    results = {}
    for agent, rows in agents.items():
        max_rud = 0.0
        max_dps = 0.0
        prev = None
        for r in rows:
            max_rud = max(max_rud, abs(r['rud_deg']))
            if prev is not None:
                dt = r['t'] - prev['t']
                if dt > 0:
                    dpsi = heading_diff_deg(r['psi_deg'], prev['psi_deg'])
                    rate = abs(dpsi) / dt
                    max_dps = max(max_dps, rate)
            prev = r
        results[agent] = (max_rud, max_dps)
    return results


def main():
    files = glob.glob('traces/*.csv')
    for f in files:
        res = analyze_file(f)
        print(f)
        for agent, stats in res.items():
            print(' agent', agent, 'max_rud_deg', stats[0], 'max_psi_deg_per_s', stats[1])

if __name__ == '__main__':
    main()
