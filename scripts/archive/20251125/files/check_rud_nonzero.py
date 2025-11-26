import csv

p='logs/ctrl_window_extract.csv'
count=0
minr=1e9
maxr=0.0
with open(p,'r',encoding='utf-8') as f:
    r=csv.DictReader(f)
    total=0
    for row in r:
        total+=1
        rud=float(row['rud'])
        minr=min(minr, abs(rud))
        maxr=max(maxr, abs(rud))
        if abs(rud) > 1e-6:
            count+=1
print(f"Total samples={total}, nonzero_rud_count={count}, min_abs_rud={minr}, max_abs_rud={maxr}")
