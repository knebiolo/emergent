"""Aggregate per-batch logs written by behavior into a single CSV and Markdown summary.

Writes:
 - outputs/profiling/batch_collated.csv
 - outputs/profiling/batch_collated_summary.md

"""
import csv
from pathlib import Path
import glob
import statistics
import json

OUT = Path('outputs/profiling')
OUT.mkdir(parents=True, exist_ok=True)

log_files = sorted(glob.glob('outputs/profiling/batch_log_*.csv'))
rows = []

# read sweep monitor to map tags -> run params
tag_map = {}
try:
    with open(OUT / 'sweep_monitor.csv', 'r', encoding='utf-8') as fh:
        r = csv.DictReader(fh)
        for row in r:
            tag = row.get('tag')
            if tag:
                tag_map[tag] = row
except Exception:
    tag_map = {}

for f in log_files:
    with open(f, 'r', encoding='utf-8') as fh:
        reader = csv.DictReader(fh)
        data = list(reader)
        if not data:
            continue
        times = [float(r['time']) for r in data if r.get('time')]
        rss_before = [int(r['rss_before']) for r in data if r.get('rss_before')]
        rss_after = [int(r['rss_after']) for r in data if r.get('rss_after')]
        # attempt to extract tag from rows (run_tag field) if present
        run_tag = data[0].get('run_tag') if data and 'run_tag' in data[0] else None
        sweep_row = tag_map.get(run_tag, {}) if run_tag else {}

        def pctile(xs, p):
            xs_sorted = sorted(xs)
            if not xs_sorted:
                return ''
            k = max(0, min(len(xs_sorted)-1, int(round((p/100.0) * (len(xs_sorted)-1)))))
            return xs_sorted[k]

        d = {
            'file': Path(f).name,
            'run_tag': run_tag or '',
            'count_batches': len(data),
            'time_mean_s': statistics.mean(times) if times else '',
            'time_median_s': statistics.median(times) if times else '',
            'time_std_s': statistics.pstdev(times) if len(times) > 1 else '',
            'time_p25_s': pctile(times, 25) if times else '',
            'time_p75_s': pctile(times, 75) if times else '',
            'rss_before_mean': statistics.mean(rss_before) if rss_before else '',
            'rss_after_mean': statistics.mean(rss_after) if rss_after else '',
            'rss_delta_mean': (statistics.mean(rss_after) - statistics.mean(rss_before)) if rss_before and rss_after else '',
            'sweep_wallclock': sweep_row.get('wallclock', ''),
            'sweep_rss': sweep_row.get('rss', ''),
        }
        rows.append(d)
        
# post-process rows: coerce numeric-like fields and compute rss delta safely
def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None

for r in rows:
    r['time_mean_s'] = safe_float(r.get('time_mean_s', ''))
    r['time_median_s'] = safe_float(r.get('time_median_s', ''))
    r['time_std_s'] = safe_float(r.get('time_std_s', ''))
    r['rss_before_mean'] = safe_float(r.get('rss_before_mean', ''))
    r['rss_after_mean'] = safe_float(r.get('rss_after_mean', ''))
    r['rss_delta_mean'] = safe_float(r.get('rss_delta_mean', ''))

# report top outliers by absolute RSS delta and by mean time
def top_outliers(rows, key, topn=5):
    keyed = [(r.get(key) if r.get(key) is not None else 0.0, r) for r in rows]
    keyed = sorted(keyed, key=lambda x: abs(x[0]), reverse=True)
    return [r for _, r in keyed[:topn]]

top_rss = top_outliers(rows, 'rss_delta_mean', topn=5)
top_time = top_outliers(rows, 'time_mean_s', topn=5)

print('\nTop 5 RSS-delta outliers:')
for r in top_rss:
    print(f" {r['file']}: rss_delta_mean={r.get('rss_delta_mean')}")
print('\nTop 5 time-mean outliers:')
for r in top_time:
    print(f" {r['file']}: time_mean_s={r.get('time_mean_s')}")

out_csv = OUT / 'batch_collated_detailed.csv'
with out_csv.open('w', newline='', encoding='utf-8') as fh:
    fieldnames = ['file','run_tag','count_batches','time_mean_s','time_median_s','time_std_s','time_p25_s','time_p75_s','rss_before_mean','rss_after_mean','rss_delta_mean','sweep_wallclock','sweep_rss']
    w = csv.DictWriter(fh, fieldnames=fieldnames)
    w.writeheader()
    for r in rows:
        w.writerow(r)

# write a short markdown summary and append to the session journal
md = OUT / 'batch_collated_summary.md'
with md.open('w', encoding='utf-8') as fh:
    fh.write('# Batch Collated Summary\n\n')
    if not rows:
        fh.write('No batch log files with data were found.\n')
    else:
        fh.write(f'Total runs aggregated: {len(rows)}\n\n')
        fh.write('|file|run_tag|count_batches|time_mean(s)|time_median(s)|time_std(s)|p25(s)|p75(s)|rss_delta_mean(bytes)|sweep_wallclock(s)|sweep_rss(bytes)|\n')
        fh.write('|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n')
        for r in rows:
            fh.write(f"{r['file']}|{r['run_tag']}|{r['count_batches']}|{r['time_mean_s']}|{r['time_median_s']}|{r['time_std_s']}|{r['time_p25_s']}|{r['time_p75_s']}|{r['rss_delta_mean']}|{r['sweep_wallclock']}|{r['sweep_rss']}\n")

# Append a short note to session journal
try:
    session_journal = Path('src/emergent/salmon_abm/.ai_journal/session/session-2025-12-28-131200.md')
    with session_journal.open('a', encoding='utf-8') as sj:
        sj.write('\n\n## Batch Collation Summary\n')
        sj.write(f'Wrote {out_csv} and {md}. Aggregated {len(rows)} runs.\n')
except Exception:
    pass

print('Wrote', out_csv, md)
