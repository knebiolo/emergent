import numpy as np
from emergent.fish_passage.perception import calculate_front_masks, determine_slices_from_vectors, determine_slices_from_headings


def test_determine_slices_from_headings():
    headings = np.array([0.0, np.pi/2, np.pi, 3*np.pi/2])
    idx = determine_slices_from_headings(headings, num_slices=4)
    assert np.array_equal(idx, np.array([0, 1, 2, 3]))


def test_determine_slices_from_vectors():
    vectors = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    idx = determine_slices_from_vectors(vectors, num_slices=4)
    assert np.array_equal(idx, np.array([0, 1, 2, 3]))


def test_calculate_front_masks_simple():
    # 1 agent at origin facing +x
    headings = np.array([0.0])
    x_coords = np.array([[ -1.0, 0.0, 1.0 ], [ -1.0, 0.0, 1.0 ]])
    y_coords = np.array([[ -1.0, -1.0, -1.0 ], [ 1.0, 1.0, 1.0 ]])
    agent_x = np.array([0.0])
    agent_y = np.array([0.0])
    masks = calculate_front_masks(headings, x_coords, y_coords, agent_x, agent_y, behind_value=0)
    assert masks.shape == (1, 2, 3)
    # cells with x>0 are in front
    assert masks[0, :, 2].all() == 1
