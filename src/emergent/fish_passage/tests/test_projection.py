import numpy as np
from emergent.fish_passage import projection


def _legacy_project(xs_line, ys_line, px, py):
    # Try to use legacy implementation if available
    try:
        from emergent.salmon_abm.sockeye import _project_points_onto_line_numba as legacy
        return legacy(np.asarray(xs_line), np.asarray(ys_line), np.asarray(px), np.asarray(py))
    except Exception:
        # Fallback: compute using the same numpy algorithm the legacy code used
        seg_x0 = xs_line[:-1]
        seg_y0 = ys_line[:-1]
        seg_x1 = xs_line[1:]
        seg_y1 = ys_line[1:]
        vx = seg_x1 - seg_x0
        vy = seg_y1 - seg_y0
        seg_len = np.hypot(vx, vy)
        cumlen = np.concatenate([[0.0], np.cumsum(seg_len)])
        M = np.asarray(px).size
        px_e = np.asarray(px)[:, None]
        py_e = np.asarray(py)[:, None]
        x0_e = seg_x0[None, :]
        y0_e = seg_y0[None, :]
        vx_e = vx[None, :]
        vy_e = vy[None, :]
        wx = px_e - x0_e
        wy = py_e - y0_e
        denom = vx_e * vx_e + vy_e * vy_e
        denom = np.where(denom == 0, 1e-12, denom)
        t = (wx * vx_e + wy * vy_e) / denom
        t_clamped = np.clip(t, 0.0, 1.0)
        cx = x0_e + t_clamped * vx_e
        cy = y0_e + t_clamped * vy_e
        d2 = (px_e - cx) ** 2 + (py_e - cy) ** 2
        idx = np.argmin(d2, axis=1)
        chosen_t = t_clamped[np.arange(M), idx]
        chosen_seg = idx
        distances_along = cumlen[chosen_seg] + chosen_t * seg_len[chosen_seg]
        return distances_along


def test_projection_parity_simple():
    xs = np.array([0.0, 5.0, 10.0])
    ys = np.array([0.0, 0.0, 0.0])
    px = np.array([1.0, 6.0, 9.0, -1.0, 12.0])
    py = np.array([1.0, 0.0, -1.0, 0.0, 0.0])

    expected = _legacy_project(xs, ys, px, py)
    got = projection.project_points_onto_line(xs, ys, px, py)

    assert np.allclose(expected, got)


def test_projection_degenerate_segment():
    # include a zero-length segment in the middle
    xs = np.array([0.0, 5.0, 5.0, 10.0])
    ys = np.array([0.0, 0.0, 0.0, 0.0])
    px = np.array([2.5, 5.0, 7.5])
    py = np.array([1.0, 0.0, -1.0])

    expected = _legacy_project(xs, ys, px, py)
    got = projection.project_points_onto_line(xs, ys, px, py)
    assert np.allclose(expected, got)
