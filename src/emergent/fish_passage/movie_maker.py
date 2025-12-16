"""Thin movie_maker wrapper for portability.

Attempts to recreate legacy `movie_maker` behavior using matplotlib
animation/ffmpeg when available. If heavy deps are missing, falls back to
writing per-frame PNGs to a directory so callers can assemble a movie later.
"""
from datetime import datetime
import os
from typing import Any

import numpy as np

try:
    import h5py
except Exception:
    h5py = None

try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as manimation
except Exception:
    plt = None
    manimation = None


def movie_maker(directory: str, model_name: str, crs: Any, dt: float, depth_rast_transform: Any) -> str:
    """Create a movie or per-frame images from model HDF5 data.

    Returns path to the created mp4 file if successful, or the path to the
    directory containing PNG frames when falling back.
    """
    model_directory = os.path.join(directory, f"{model_name}.h5")

    # Prefer HDF5 if available and the model file exists, otherwise fall back
    # to X.npy/Y.npy which tests and some workflows may provide.
    x_path = os.path.join(directory, 'X.npy')
    y_path = os.path.join(directory, 'Y.npy')
    X_arr = None
    Y_arr = None
    if h5py is not None and os.path.exists(model_directory):
        try:
            with h5py.File(model_directory, 'r') as hdf5:
                X_arr = hdf5['agent_data/X'][:]
                Y_arr = hdf5['agent_data/Y'][:]
        except Exception:
            # Fall back to numpy arrays below
            X_arr = None
            Y_arr = None

    if X_arr is None or Y_arr is None:
        if os.path.exists(x_path) and os.path.exists(y_path):
            X_arr = np.load(x_path)
            Y_arr = np.load(y_path)
        else:
            raise RuntimeError('No agent data available: missing HDF5 and X.npy/Y.npy')

    num_columns = int(X_arr.shape[1])

    out_mp4 = os.path.join(directory, f"{model_name}.mp4")

    if plt is not None and manimation is not None:
        try:
            FFMpegWriter = manimation.writers['ffmpeg']
            metadata = dict(title=model_name, artist='Matplotlib', comment=f'emergent model run {datetime.now()}')
            writer = FFMpegWriter(fps=int(round(30.0 / dt)), metadata=metadata)

            fig, ax = plt.subplots(figsize=(10, 5))
            agent_pts, = plt.plot([], [], marker='o', ms=1, ls='', color='red')

            with writer.saving(fig, out_mp4, dpi=150):
                for i in range(num_columns):
                    agent_pts.set_data(X_arr[:, i], Y_arr[:, i])
                    writer.grab_frame()
            return out_mp4
        except Exception:
            # fall through to PNG fallback
            pass

    # Fallback: write PNG frames to a directory
    frames_dir = os.path.join(directory, f"{model_name}_frames")
    os.makedirs(frames_dir, exist_ok=True)
    for i in range(num_columns):
        # simple scatter plot per frame
        fig, ax = plt.subplots() if plt is not None else (None, None)
        if plt is not None:
            ax.scatter(X_arr[:, i], Y_arr[:, i], s=1, c='red')
            ax.set_xlabel('Easting')
            ax.set_ylabel('Northing')
            out_png = os.path.join(frames_dir, f'frame_{i:04d}.png')
            fig.savefig(out_png)
            plt.close(fig)
        else:
            # Without matplotlib, write simple text placeholder files
            out_png = os.path.join(frames_dir, f'frame_{i:04d}.txt')
            with open(out_png, 'w') as fh:
                fh.write(f'frame {i}\n')

    return frames_dir
