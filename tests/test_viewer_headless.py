import numpy as np
from emergent.salmon_abm.salmon_viewer import SalmonViewer


class DummySim:
    def __init__(self):
        self.use_hecras = False
        self.num_agents = 0


def test_load_tin_payload_dict():
    sv = SalmonViewer(DummySim())
    # create a tiny payload
    verts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.1], [0.0, 1.0, 0.2]])
    faces = np.array([[0, 1, 2]])
    colors = np.ones((3, 4))
    payload = {'verts': verts, 'faces': faces, 'colors': colors}
    v, f, c = sv.load_tin_payload(payload)
    assert v.shape == (3, 3)
    assert f.shape == (1, 3)
    assert c.shape == (3, 4)


def test_viewer_widgets_exist():
    sv = SalmonViewer(DummySim())
    # check for expected widgets from reference
    assert hasattr(sv, 'play_btn')
    assert hasattr(sv, 'pause_btn')
    assert hasattr(sv, 'reset_btn')
    assert hasattr(sv, 've_slider')
    assert hasattr(sv, 'episode_label')
