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


def test_compute_distance_to_bank_hecras_wrapper_consistency():
	# Create a small synthetic grid of points
	xs, ys = np.meshgrid(np.arange(5), np.arange(5))
	coords = np.column_stack((xs.flatten().astype(float), ys.flatten().astype(float)))
	# wetted mask: central cross
	wetted_mask = np.zeros(len(coords), dtype=bool)
	wetted_mask[(coords[:,0]==2) | (coords[:,1]==2)] = True
	perimeter_indices = np.where(wetted_mask & ((coords[:,0]==0)|(coords[:,0]==4)|(coords[:,1]==0)|(coords[:,1]==4)))[0]
	# compute both ways
	from emergent.fish_passage.geometry import compute_distance_to_bank
	d1 = compute_distance_to_bank(coords, wetted_mask, perimeter_indices)
	wetted_info = {'wetted_mask': wetted_mask, 'perimeter_cells': perimeter_indices}
	from emergent.fish_passage.geometry import compute_distance_to_bank_hecras
	d2 = compute_distance_to_bank_hecras(wetted_info, coords)
	assert np.allclose(np.nan_to_num(d1, nan=-1.0), np.nan_to_num(d2, nan=-1.0))
