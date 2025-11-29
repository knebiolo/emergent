import pandas as pd
from pathlib import Path

indir = Path('outputs/benchmarks')
out_md = indir / 'summary.md'
rows = []
for p in sorted(indir.glob('bench_*.csv')):
    df = pd.read_csv(p)
    mean = df['duration_s'].mean()
    median = df['duration_s'].median()
    std = df['duration_s'].std()
    # parse k and n from filename
    name = p.stem
    parts = name.split('_')
    k = parts[1][1:]
    n = parts[2][1:]
    rows.append((p.name, int(k), int(n), mean, median, std))

rows = sorted(rows, key=lambda x: (x[1], x[2]))
with out_md.open('w') as fh:
    fh.write('# HECRAS Benchmark Summary\n\n')
    fh.write('| file | k | num_agents | mean_s | median_s | std_s |\n')
    fh.write('|---|---:|---:|---:|---:|---:|\n')
    for name, k, n, mean, median, std in rows:
        fh.write(f'| {name} | {k} | {n} | {mean:.6f} | {median:.6f} | {std:.6f} |\n')

print('Wrote', out_md)
