# Minimal reproducer for SalmonViewer.load_tin_payload
import numpy as np
import os
from emergent.salmon_abm.salmon_viewer import SalmonViewer

# Create a synthetic payload: simple triangle mesh
verts = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 1.0], [5.0, 8.0, 0.5]])
faces = np.array([[0,1,2]])
colors = np.array([[1.0,0.0,0.0,1.0], [0.0,1.0,0.0,1.0], [0.0,0.0,1.0,1.0]])

payload = {'verts': verts, 'faces': faces, 'colors': colors}

sv = None
# We only need the load_tin_payload function; create a dummy sim placeholder
class DummySim:
    def __init__(self):
        self.num_agents = 3
        self.X = np.array([1.0, 5.0, 8.0])
        self.Y = np.array([1.0, 2.0, 3.0])
        self.dead = np.zeros(self.num_agents)
        self.heading = np.zeros(self.num_agents)

sim = DummySim()
from PyQt5 import QtWidgets
app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
sv = SalmonViewer(sim)
verts_out, faces_out, colors_out = sv.load_tin_payload(payload)
print('synthetic verts_out shape:', verts_out.shape)
print('synthetic faces_out shape:', faces_out.shape)
print('synthetic colors_out shape:', colors_out.shape)

# Try a real payload if available
real_path = os.path.join('outputs', 'tin_experiment.npz')
if os.path.exists(real_path):
    print('\nLoading real payload:', real_path)
    r_verts, r_faces, r_colors = sv.load_tin_payload(real_path)
    print('real verts shape:', getattr(r_verts, 'shape', None))
    print('real faces shape:', getattr(r_faces, 'shape', None))
    print('real colors shape:', getattr(r_colors, 'shape', None))
    minxy = np.min(r_verts[:, :2], axis=0)
    maxxy = np.max(r_verts[:, :2], axis=0)
    print('real mesh bounds:', minxy, maxxy)
    print('agent positions:', sim.X, sim.Y)
    in_x = (sim.X >= minxy[0]) & (sim.X <= maxxy[0])
    in_y = (sim.Y >= minxy[1]) & (sim.Y <= maxxy[1])
    print('agents_in_real_bbox:', in_x & in_y)

# Sanity check: agent coordinates relative to mesh bounds
minxy = np.min(verts_out[:,:2], axis=0)
maxxy = np.max(verts_out[:,:2], axis=0)
print('mesh bounds:', minxy, maxxy)
print('agent positions:', sim.X, sim.Y)

# Validate that agents lie within or near mesh bounds (basic test)
in_x = (sim.X >= minxy[0]) & (sim.X <= maxxy[0])
in_y = (sim.Y >= minxy[1]) & (sim.Y <= maxxy[1])
print('agents_in_bbox:', in_x & in_y)
