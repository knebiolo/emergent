import h5py
import csv
import numpy as np
from emergent.salmon_abm.utils import geo_to_pixel
import sys

pre = sys.argv[1]
trace = sys.argv[2]

with h5py.File(pre,'r') as f:
    mental = f.attrs.get('mental_map_transform', None)
    print('mental_map_transform:', mental)
    mem = f['memory']['0'][:]
    print('memory[0] shape', mem.shape)

with open(trace,'r') as fh:
    rdr = csv.DictReader(fh)
    # find the first row for agent 0
    row0 = None
    for r in rdr:
        if int(r.get('agent', r.get('Agent', 0))) == 0:
            row0 = r
            break
    if row0 is None:
        print('No agent 0 row found in trace')
        sys.exit(2)
    ax = float(row0.get('x', row0.get('X', 0)))
    ay = float(row0.get('y', row0.get('Y', 0)))
    print('agent0 world pos:', ax, ay)

# compute pixel indices
prow, pcol = geo_to_pixel(ax, ay, mental)
print('agent0 mental pixel (row,col)=', prow, pcol)

# print neighborhood and t_since
buff = 10
rmin = max(0, prow - buff)
rmax = min(mem.shape[0], prow + buff + 1)
cmin = max(0, pcol - buff)
cmax = min(mem.shape[1], pcol + buff + 1)
print('slice rows', rmin, rmax, 'cols', cmin, cmax)
section = mem[rmin:rmax, cmin:cmax]
print('mmap_section shape', section.shape)
print('mmap_section sample (min,max,unique_count):', float(np.nanmin(section)), float(np.nanmax(section)), int(np.unique(section).size))

# assume t=0 (single-step run) — find t from trace if present
# trace has timestep column
with open(trace,'r') as fh:
    rdr = csv.DictReader(fh)
    first = next(iter(rdr))
# default t=0
t = 0
print('assumed t =', t)

t_since = section - t
mult = np.where((t_since > 10) & (t_since < 7200), 1 - (t_since - 5) / (7195), 0)
print('t_since stats min,max:', float(np.nanmin(t_since)), float(np.nanmax(t_since)))
print('multiplier stats min,max,sum,nonzero:', float(np.nanmin(mult)), float(np.nanmax(mult)), float(np.nansum(mult)), int((mult!=0).sum()))
print('multiplier sample (flattened):', mult.ravel()[:20])
