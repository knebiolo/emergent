import glob, os, numpy as np
files = glob.glob(os.path.join(os.getcwd(),'outputs','*_mesh.npz'))
if not files:
    print('No mesh files found')
    raise SystemExit(0)
f = max(files, key=os.path.getmtime)
print('Using mesh file:', f)
d = np.load(f)
verts = d['verts']
faces = d['faces']
colors = d['colors']
print('verts.shape', verts.shape)
print('faces.shape', faces.shape)
print('colors.shape', colors.shape)
print('Z min/max:', float(verts[:,2].min()), float(verts[:,2].max()))
print('colors min/max per channel:', colors.min(axis=0).tolist(), colors.max(axis=0).tolist())
print('first 5 verts z:', verts[:5,2].tolist())
print('first 5 colors:', colors[:5].tolist())
print('faces[0:5]:', faces[:5].tolist())
