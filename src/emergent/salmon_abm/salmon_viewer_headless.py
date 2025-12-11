"""Headless SalmonViewer shim for tests.

This module provides a minimal `SalmonViewer` class that exposes
`load_tin_payload(payload)` and other small helpers needed by headless
unit tests. It intentionally avoids importing PyQt/OpenGL so tests can
import it in environments without GUI dependencies.

Use-case: tests that need to exercise mesh loading and payload handling
without launching the real viewer should import this shim instead of the
full GUI module.
"""
from __future__ import annotations
from typing import Any, Dict, Tuple
import numpy as np


class SalmonViewer:
    def __init__(self, simulation: Any = None, dt: float = 0.1, T: float = 1.0, **kwargs):
        self.sim = simulation
        self.dt = dt
        self.T = T
        # Minimal dummy attributes expected by some tests (headless placeholders)
        self.play_btn = None
        self.pause_btn = None
        self.reset_btn = None
        self.ve_slider = None
        self.episode_label = None

    def load_tin_payload(self, payload: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Normalize a payload dict into (verts, faces, colors).

        - `verts` will be returned as Nx3 float array (adds z=0 for 2D inputs)
        - `faces` will be returned as Mx3 int array
        - `colors` will be returned as Nx4 float array
        """
        verts = payload.get('verts')
        faces = payload.get('faces')
        colors = payload.get('colors')

        verts = np.asarray(verts, dtype=float)
        if verts.ndim == 1:
            # try to reshape if possible
            if verts.size % 3 == 0:
                verts = verts.reshape((-1, 3))
            elif verts.size % 2 == 0:
                verts = verts.reshape((-1, 2))
        if verts.ndim == 2 and verts.shape[1] == 2:
            z = np.zeros((verts.shape[0], 1), dtype=float)
            verts = np.concatenate([verts, z], axis=1)

        faces = np.asarray(faces, dtype=int)
        colors = np.asarray(colors, dtype=float)
        return verts, faces, colors
