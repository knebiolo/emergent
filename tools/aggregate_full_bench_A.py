import csv
import statistics as s
from pathlib import Path

files = [
    'outputs/full_bench_strict_A_n10.csv',
    'outputs/full_bench_strict_A_n50.csv',
    'outputs/full_bench_strict_A_n100.csv',
    'outputs/full_bench_strict_A_n1000.csv',
]

out = []
for f in files:
    p = Path(f)
    if not p.exists():
        out.append((f, None))
        continue
    with p.open() as fh:
        rdr = list(csv.DictReader(fh))
    vals = [float(r['duration_s']) for r in rdr]
    out.append((f, (s.mean(vals), s.median(vals), s.pstdev(vals), len(vals))))

print('Per-agent-only strict benchmark summary:')
for f, stats in out:
    if stats is None:
        print(f, ': MISSING')
    else:
        mean, med, pstdev, n = stats
        print(f, ': n=', n, ' mean={:.6f}s median={:.6f}s pstdev={:.6f}s'.format(mean, med, pstdev))

# write markdown summary
md = ['**Per-agent-only strict benchmark (20 timesteps)**\n']
for f, stats in out:
    if stats is None:
        md.append(f'- **{f}**: MISSING')
    else:
        mean, med, pstdev, n = stats
        md.append(f'- **{f}**: n={n}, mean={mean:.6f}s, median={med:.6f}s, pstdev={pstdev:.6f}s')

pout = Path('outputs/full_bench_summary_A.md')
pout.write_text('\n'.join(md))
print('\nWrote', pout)
