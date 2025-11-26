import pandas as pd
import glob

files = glob.glob('**/*.csv', recursive=True)
results = []
mx = -1.0
for f in files:
    try:
        df = pd.read_csv(f)
        m = None
        if 'max_rud_deg' in df.columns:
            try:
                m = float(df['max_rud_deg'].max())
            except Exception:
                m = None
        if m is None and 'rud_deg' in df.columns:
            try:
                m = float(df['rud_deg'].abs().max())
            except Exception:
                m = None
        if m is not None:
            results.append((f, m))
            if m > mx:
                mx = m
    except Exception:
        # ignore files we can't parse
        continue

if mx < 0:
    print('No rudder data found in CSV files.')
else:
    print('overall_max_rud_deg_deg:', mx)
    exceed = [(f,m) for f,m in results if m >= 14.0]
    if exceed:
        print('\nFiles exceeding or equal to 14.0°:')
        for f,m in exceed:
            print(f, m)
    else:
        print('\nNo files reached 14.0° or above (all below cap).')

    print('\nPer-file max_rud_deg:')
    for f,m in sorted(results, key=lambda x: -x[1])[:50]:
        print(f, m)
