import pandas as pd
from pathlib import Path

in_csv = Path('figs') / 'prbs_id' / 'prbs_id_summary.csv'
out_csv = Path('figs') / 'prbs_id' / 'prbs_best_per_speed.csv'

df = pd.read_csv(in_csv)
# infer U token from filename (id_long_U3.0_...)

def get_U(file):
    name = Path(file).name
    parts = name.split('_')
    for p in parts:
        if p.startswith('U'):
            try:
                return float(p[1:])
            except:
                return p
    return 'unknown'

DF = df.copy()
DF['U'] = DF['file'].apply(get_U)

best_rows = []
for U, g in DF.groupby('U'):
    # pick highest r2_r, tie-breaker lower rmse
    g_sorted = g.sort_values(['r2_r', 'rmse_r_deg_s'], ascending=[False, True])
    best_rows.append(g_sorted.iloc[0])

best_df = pd.DataFrame(best_rows)
best_df.to_csv(out_csv, index=False)
print('Wrote', out_csv)
