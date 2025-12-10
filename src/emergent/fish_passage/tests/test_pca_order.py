import numpy as np
from emergent.fish_passage.geometry import pca_primary_axis, order_by_projection


def make_elongated_cloud(n=100, angle=0.2, length=50.0, noise=0.5):
    # generate points along a line at given angle plus Gaussian noise
    ts = np.linspace(0, length, n)
    xs = ts * np.cos(angle)
    ys = ts * np.sin(angle)
    xs += np.random.normal(scale=noise, size=n)
    ys += np.random.normal(scale=noise, size=n)
    return np.column_stack([xs, ys])


def test_pca_primary_axis_direction():
    coords = make_elongated_cloud(n=200, angle=0.5)
    (vx, vy), proj = pca_primary_axis(coords)
    # direction should align with angle ~0.5 (cos positive)
    assert abs(vx) > 0.5
    assert proj.shape[0] == coords.shape[0]


def test_order_by_projection_monotonic():
    coords = make_elongated_cloud(n=200, angle=0.3)
    order = order_by_projection(coords)
    proj_vals = np.dot(coords - coords.mean(axis=0), np.array([1.0, 0.0]))
    # ensure order indices are a permutation
    assert set(order.tolist()) == set(range(len(coords)))
    # Check monotonicity along order using projection on primary axis
    (vx, vy), proj = pca_primary_axis(coords)
    ordered_proj = proj[order]
    assert np.all(np.diff(ordered_proj) >= -1e-6)
