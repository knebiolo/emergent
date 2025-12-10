"""Inspect .npz TIN payload bounds without importing the viewer.
Usage: run with the project conda Python to print verts/faces shapes and XY bounds.
"""
import os
import numpy as np

def inspect(path):
    print('Inspecting', path)
    data = np.load(path)
    keys = list(data.keys())
    print('keys:', keys)
    verts = data.get('verts')
    faces = data.get('faces')
    colors = data.get('colors')
    print('verts shape:', None if verts is None else verts.shape)
    print('faces shape:', None if faces is None else faces.shape)
    print('colors shape:', None if colors is None else getattr(colors, 'shape', None))
    if verts is not None:
        minxy = np.min(verts[:, :2], axis=0)
        maxxy = np.max(verts[:, :2], axis=0)
        print('xy min:', minxy)
        print('xy max:', maxxy)

if __name__ == '__main__':
    p = os.path.join('outputs', 'tin_experiment.npz')
    if os.path.exists(p):
        inspect(p)
    else:
        print('no real payload found at', p)
