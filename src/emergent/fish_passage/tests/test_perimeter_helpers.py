import numpy as np

from emergent.fish_passage.geometry import (
	order_points_nearest_neighbor,
	thin_perimeter_uniform,
)


def test_order_points_nearest_neighbor_simple_square():
	pts = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
	order = order_points_nearest_neighbor(pts, start_idx=0)
	assert set(order) == set(range(4))
	# sequence should start at 0
	assert int(order[0]) == 0


def test_thin_perimeter_uniform_triangle():
	pts = np.array([[0.0, 0.0], [2.0, 0.0], [1.0, 1.732], [0.0, 0.0]])
	# perimeter ~6.0; target_spacing 2 -> expect ~3 points
	thinned = thin_perimeter_uniform(pts, target_spacing=2.0)
	assert thinned.shape[0] >= 2
	# ensure returned points are subset of original points coordinates
	for p in thinned:
		assert any(np.allclose(p, q) for q in pts)
