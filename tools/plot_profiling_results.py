import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path('outputs/profiling')
CSV = OUT / 'batch_collated_detailed.csv'
if not CSV.exists():
    raise SystemExit(f'Missing {CSV}')

df = pd.read_csv(CSV)
# ensure numeric
for col in ['time_mean_s','time_median_s','time_std_s','rss_delta_mean','count_batches']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Try to parse batch size from file or run_tag
if 'batch' in df.columns:
    df['batch_size'] = pd.to_numeric(df['batch'], errors='coerce')
else:
    # extract batch from file names like batch_log_n10000_b256_...
    def extract_batch(fn):
        import re
        m = re.search(r'_b(\d+)_', fn)
        if m:
            return int(m.group(1))
        m = re.search(r'_b(\d+)\.', fn)
        if m:
            return int(m.group(1))
        return None
    df['batch_size'] = df['file'].apply(extract_batch)

# Extract agent count
if 'n_agents' in df.columns:
    df['n_agents'] = pd.to_numeric(df['n_agents'], errors='coerce')
else:
    def extract_n(fn):
        import re
        m = re.search(r'_n(\d+)_', fn)
        if m:
            return int(m.group(1))
        return None
    df['n_agents'] = df['file'].apply(extract_n)

# Drop rows with missing batch_size
df = df.dropna(subset=['batch_size'])

# Pivot time_mean by batch_size and n_agents
fig1 = plt.figure(figsize=(8,5))
for n in sorted(df['n_agents'].unique()):
    sub = df[df['n_agents']==n]
    sub = sub.sort_values('batch_size')
    plt.plot(sub['batch_size'], sub['time_mean_s'], marker='o', label=f'n={int(n)}')
plt.xscale('log', base=2)
plt.xlabel('Batch size')
plt.ylabel('Per-batch mean time (s)')
plt.title('Per-batch mean time vs batch size')
plt.legend()
plt.grid(True, which='both', ls='--', alpha=0.4)
fig1_path = OUT / 'time_vs_batch.png'
fig1.savefig(fig1_path, dpi=150, bbox_inches='tight')
plt.close(fig1)

# RSS delta mean vs batch size
fig2 = plt.figure(figsize=(8,5))
for n in sorted(df['n_agents'].unique()):
    sub = df[df['n_agents']==n]
    sub = sub.sort_values('batch_size')
    plt.plot(sub['batch_size'], sub['rss_delta_mean'], marker='o', label=f'n={int(n)}')
plt.xscale('log', base=2)
plt.xlabel('Batch size')
plt.ylabel('Mean RSS delta (bytes)')
plt.title('Mean RSS delta vs batch size')
plt.legend()
plt.grid(True, which='both', ls='--', alpha=0.4)
fig2_path = OUT / 'rss_delta_vs_batch.png'
fig2.savefig(fig2_path, dpi=150, bbox_inches='tight')
plt.close(fig2)

print('Wrote', fig1_path, fig2_path)

# write outliers summary
outliers = df[['file','run_tag','n_agents','batch_size','time_mean_s','rss_delta_mean']].copy()
outliers['abs_rss_delta'] = outliers['rss_delta_mean'].abs()
top_rss = outliers.sort_values('abs_rss_delta', ascending=False).head(10)
top_time = outliers.sort_values('time_mean_s', ascending=False).head(10)
outliers_csv = OUT / 'outliers_summary.csv'
top_rss.to_csv(OUT / 'top_rss_outliers.csv', index=False)
top_time.to_csv(OUT / 'top_time_outliers.csv', index=False)
top_rss.to_csv(outliers_csv, index=False)
print('Wrote outlier summaries:', outliers_csv, 'top_rss_outliers.csv', 'top_time_outliers.csv')
