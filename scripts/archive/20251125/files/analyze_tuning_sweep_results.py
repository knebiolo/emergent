import csv
from collections import defaultdict
import math

path = 'c:\\Users\\Kevin.Nebiolo\\OneDrive - Kleinschmidt Associates\\Software\\emergent\\scripts\\tuning_sweep_summary.csv'
rows = []
with open(path, newline='') as f:
    reader = csv.DictReader(f)
    for r in reader:
        try:
            rows.append({
                'run_id': r['run_id'],
                'Kp': float(r['Kp_factor']),
                'Ki': float(r['Ki_factor']),
                'max_rudder': float(r['max_rudder_deg']),
                'enroute': int(float(r['enroute_event_count'])),
                'mean_abs_err': float(r['mean_abs_err_deg']) if r['mean_abs_err_deg'] else math.nan,
            })
        except Exception as e:
            print('skip row', r, 'err', e)

print('total rows:', len(rows))
# unique parameter combos
combos = defaultdict(list)
for r in rows:
    key = (r['Kp'], r['Ki'], r['max_rudder'])
    combos[key].append(r)

print('unique parameter combos:', len(combos))

# aggregate per combo
aggs = []
for k,v in combos.items():
    en = [x['enroute'] for x in v]
    me = [x['mean_abs_err'] for x in v if not math.isnan(x['mean_abs_err'])]
    aggs.append({'Kp':k[0],'Ki':k[1],'max_rudder':k[2],'count_rows':len(v),'enroute_mean': sum(en)/len(en),'enroute_min':min(en),'enroute_max':max(en),'mean_abs_err_mean': sum(me)/len(me) if me else math.nan})

# sort by enroute_mean desc
tag = sorted(aggs, key=lambda x: x['enroute_mean'], reverse=True)
print('\nTop combos by mean enroute_event_count:')
for t in tag[:10]:
    print(t)

# top individual rows by enroute
rows_sorted = sorted(rows, key=lambda x: x['enroute'], reverse=True)
print('\nTop individual rows by enroute_event_count:')
for r in rows_sorted[:10]:
    print(r)

# by mean_abs_err
rows_sorted_err = sorted(rows, key=lambda x: x['mean_abs_err'], reverse=True)
print('\nTop individual rows by mean_abs_err (desc):')
for r in rows_sorted_err[:10]:
    print(r)

# group stats by Kp
by_kp = defaultdict(list)
for r in rows:
    by_kp[r['Kp']].append(r['enroute'])
print('\nMean enroute_event_count by Kp:')
for k in sorted(by_kp.keys()):
    vals = by_kp[k]
    print(k, sum(vals)/len(vals), 'n=',len(vals))

# counts of distinct enroute values
vals = defaultdict(int)
for r in rows:
    vals[r['enroute']]+=1
print('\ndistinct enroute_event_count values and frequencies:')
for k in sorted(vals.keys()):
    print(k, vals[k])

print('\nFinished analysis')
