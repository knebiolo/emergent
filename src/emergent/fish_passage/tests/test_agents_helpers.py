import numpy as np

from emergent.fish_passage.utils import calculate_front_masks, determine_slices_from_vectors, determine_slices_from_headings


def test_determine_slices_from_headings_and_vectors():
    headings = np.array([0.0, np.pi/2, np.pi, 3*np.pi/2])
    slices = determine_slices_from_headings(headings, num_slices=4)
    assert np.array_equal(slices, np.array([0, 1, 2, 3]))

    vectors = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    vs = determine_slices_from_vectors(vectors, num_slices=4)
    assert np.array_equal(vs, np.array([0, 1, 2, 3]))


def test_calculate_front_masks():
    # simple 3x3 grid centered at origin
    xs, ys = np.meshgrid(np.linspace(-1, 1, 3), np.linspace(-1, 1, 3))
    headings = np.array([0.0])  # facing +x
    agent_x = np.array([0.0])
    agent_y = np.array([0.0])
    masks = calculate_front_masks(headings, xs, ys, agent_x, agent_y, behind_value=0)
    # cells with x > 0 should be front (columns index 2 in meshgrid ordering)
    assert masks.shape == (1, 3, 3)
    assert masks[0, :, 2].all() == True
