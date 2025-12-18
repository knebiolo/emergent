"""Simple launcher to start SalmonViewer (v3 shim) and load the latest outputs mesh.

Usage: python scripts/launch_viewer.py

This creates a minimal `sim` placeholder with no heavy dependencies and uses
`SalmonViewer.load_hecras_mesh` to load a prebuilt mesh `.npz` from `outputs/`.
"""
import sys, os, glob
import numpy as np
from PyQt5 import QtWidgets
from emergent.salmon_abm.viewer_v3.viewer_shim import SalmonViewer


def find_latest_mesh():
    outdir = os.path.join(os.getcwd(), 'outputs')
    if not os.path.isdir(outdir):
        return None
    files = glob.glob(os.path.join(outdir, '*_mesh.npz'))
    if not files:
        return None
    return max(files, key=os.path.getmtime)


class DummySim:
    def __init__(self):
        self.hdf5 = None
        self.perimeter_points = None
        self.vert_exag = 1.0


def main():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    sim = DummySim()
    viewer = SalmonViewer(sim)
    # try to load the latest mesh file directly
    mesh = find_latest_mesh()
    if mesh is not None:
        try:
            d = np.load(mesh)
            verts = d['verts']
            faces = d['faces']
            colors = d['colors']
            viewer.gl_widget.set_mesh(verts, faces, colors)
            print('Loaded mesh', mesh)
        except Exception:
            print('Failed to load mesh file:', mesh)
    else:
        print('No mesh file found in outputs/')
    try:
        # build full UI (this calls QApplication.exec_ internally)
        viewer.run()
        return
    except Exception as e:
        print('Failed to start Qt/OpenGL viewer:', e)
        # fallback: open the most recent preview PNG if available
        outdir = os.path.join(os.getcwd(), 'outputs')
        candidates = []
        if os.path.isdir(outdir):
            candidates = glob.glob(os.path.join(outdir, '*_preview*.png'))
        if candidates:
            latest_preview = max(candidates, key=os.path.getmtime)
            print('Opening preview image instead:', latest_preview)
            try:
                # Windows: launch default image viewer
                os.startfile(latest_preview)
            except Exception:
                try:
                    # fallback: show via matplotlib
                    import matplotlib.pyplot as plt
                    img = plt.imread(latest_preview)
                    plt.figure(figsize=(10, 8))
                    plt.imshow(img)
                    plt.axis('off')
                    plt.show()
                except Exception as e2:
                    print('Failed to open preview image:', e2)
        else:
            print('No preview image found in outputs/')


if __name__ == '__main__':
    main()
