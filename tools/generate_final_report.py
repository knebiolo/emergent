import pstats
import sys

files = [
    ('premerge','tools/profile_2000_premerge.pstats'),
    ('postmerge','tools/profile_2000_postmerge.pstats'),
    ('final','tools/profile_2000_final.pstats')
]
results = {}
for label, path in files:
    try:
        p = pstats.Stats(path)
        # total time
        total = p.total_tt
        # find timestep cumulative
        # p.stats is a dict keyed by (filename, lineno, funcname) -> stat tuple
        cum = None
        for key, val in p.stats.items():
            # key is (filename, lineno, funcname)
            fn = key[2]
            if fn == 'timestep':
                # val format: (cc, ncalls, tot_time, cum_time, callers)
                try:
                    cum = float(val[3])
                except Exception:
                    cum = None
                break
        results[label] = {'path': path, 'total': total, 'timestep_cum': cum}
    except Exception as e:
        results[label] = {'path': path, 'error': str(e)}

# write report
with open('tools/perf_report_final.md','w') as f:
    f.write('# Final Performance Report\n\n')
    f.write('Benchmark: 2000 agents × 50 timesteps\n\n')
    for label in ['premerge','postmerge','final']:
        r = results.get(label, {})
        f.write(f'## {label}\n')
        if 'error' in r:
            f.write(f'- error loading: {r["error"]}\n\n')
            continue
        f.write(f'- pstats file: `{r["path"]}`\n')
        f.write(f'- total time (pstats total_tt): {r["total"]}\n')
        f.write(f'- `timestep` cumulative time: {r["timestep_cum"]}\n\n')
    # compute percent changes if available
    try:
        p0 = results['premerge']['timestep_cum']
        p1 = results['postmerge']['timestep_cum']
        pf = results['final']['timestep_cum']
        if p0 and p1:
            change1 = (p1 - p0)/p0*100.0
            f.write(f'Post-merge vs premerge: {change1:.2f}% change in `timestep`\n')
        if p0 and pf:
            changef = (pf - p0)/p0*100.0
            f.write(f'Final (battery merge) vs premerge: {changef:.2f}% change in `timestep`\n')
        if p1 and pf:
            change2 = (pf - p1)/p1*100.0
            f.write(f'Final vs postmerge: {change2:.2f}% change in `timestep`\n')
    except Exception:
        pass

print('wrote tools/perf_report_final.md')
