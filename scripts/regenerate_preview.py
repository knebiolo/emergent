import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

outdir = Path('outputs')
if not outdir.exists():
    print('outputs directory missing')
    sys.exit(1)

# find mesh npz
mesh_files = sorted(outdir.glob('*_mesh.npz'))
if not mesh_files:
    print('No mesh npz files found in outputs/')
    sys.exit(1)
mesh = mesh_files[-1]
print('Using mesh file:', mesh.resolve())

data = np.load(str(mesh))
verts = data.get('verts')
colors = data.get('colors')
if verts is None:
    print('mesh has no verts')
    sys.exit(1)

# build preview image
x = verts[:,0]
y = verts[:,1]

png = outdir / (mesh.stem + '_preview_regen.png')
plt.figure(figsize=(10,8))
if colors is not None and colors.shape[0] == verts.shape[0]:
    # colors may be RGBA float 0-1 or 0-255
    c = colors[:,:3]
    if c.max() > 2:
        c = c/255.0
    plt.scatter(x,y,c=c,s=1)
else:
    plt.scatter(x,y,s=1)
plt.axis('equal')
plt.tight_layout()
plt.savefig(png, dpi=150)
plt.close()
print('Wrote preview to', png.resolve())
