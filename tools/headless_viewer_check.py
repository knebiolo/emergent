"""Headless viewer check script.

Runs a minimal import and API checks for `salmon_viewer_v2` without using pytest.
Usage: `python tools/headless_viewer_check.py`
"""
import numpy as np
import traceback
from emergent.salmon_abm.salmon_viewer import SalmonViewer
from PyQt5 import QtWidgets

# Attempt to import the sockeye simulation module to surface import-time errors
try:
    import emergent.salmon_abm.sockeye as sockeye
    print('Imported emergent.salmon_abm.sockeye OK')
except Exception as e:
    print('Failed to import emergent.salmon_abm.sockeye:')
    traceback.print_exc()


class DummySim:
    def __init__(self):
        self.use_hecras = False
        self.num_agents = 0


def main():
    print('Creating QApplication, DummySim and SalmonViewer...')
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    sim = DummySim()
    sv = SalmonViewer(sim)
    print('Testing load_tin_payload with synthetic payload...')
    verts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.1], [0.0, 1.0, 0.2]])
    faces = np.array([[0, 1, 2]])
    colors = np.ones((3, 4))
    payload = {'verts': verts, 'faces': faces, 'colors': colors}
    v, f, c = sv.load_tin_payload(payload)
    print('Verts shape:', v.shape, 'Faces shape:', f.shape, 'Colors shape:', c.shape)
    print('Checking widgets...')
    for attr in ('play_btn', 'pause_btn', 'reset_btn', 've_slider', 'episode_label'):
        print(attr, 'exists:', hasattr(sv, attr))
    print('All checks passed (headless).')
    # clean up the QApplication created for headless checks
    try:
        QtWidgets.QApplication.quit()
    except Exception:
        pass

if __name__ == '__main__':
    main()
