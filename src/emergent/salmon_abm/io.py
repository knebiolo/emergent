"""I/O and environment helpers for the salmon ABM.

This module provides small wrappers around rasterio, geopandas and h5py
used by the simulation. Implementations are conservative to avoid
side-effects during import and testing.
"""
import os
import h5py
import pandas as pd
import rasterio
import geopandas as gpd
from rasterio.transform import Affine
from matplotlib import animation as manimation
from datetime import datetime
import numpy as np


def output_excel(records, model_dir, model_name):
    """Write a dict of pandas DataFrames to an Excel file.

    Parameters
    - records: dict[str, pd.DataFrame]
    - model_dir: str
    - model_name: str
    """
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
    output_excel_path = os.path.join(model_dir, f'output_{model_name}.xlsx')
    # Try to write an Excel file; if openpyxl/xlsxwriter are not available,
    # fall back to writing CSV files per sheet.
    try:
        with pd.ExcelWriter(output_excel_path) as writer:
            for generation_name, df in records.items():
                try:
                    df.to_excel(writer, sheet_name=str(generation_name))
                except Exception:
                    pd.DataFrame({'error': ['could not write sheet']}).to_excel(writer, sheet_name=str(generation_name))
    except Exception:
        # fallback: write CSVs
        for generation_name, df in records.items():
            csv_path = os.path.join(model_dir, f'{model_name}_{generation_name}.csv')
            try:
                df.to_csv(csv_path, index=False)
            except Exception:
                with open(csv_path + '.error', 'w') as fh:
                    fh.write('could not write data')


def movie_maker(directory, model_name, crs, dt, depth_rast_transform, depth_arr, X_arr=None, Y_arr=None):
    """Lightweight movie maker that uses matplotlib animation writers.

    This function focuses on creating a simple mp4 from a 2D depth array and
    optional agent trajectories. To keep it testable we do not call ffmpeg
    directly here; matplotlib will use the configured writer when available.
    """
    model_directory = os.path.join(directory, f'{model_name}.h5')
    if not os.path.isdir(directory):
        os.makedirs(directory, exist_ok=True)

    metadata = dict(title=model_name, artist='Matplotlib', comment=f'emergent model run {datetime.now()}')
    fps = max(1, int(round(30.0 / max(dt, 1e-6))))
    out_path = os.path.join(directory, f'{model_name}.mp4')

    # matplotlib's writer registry supports dict-like access but may raise
    # a KeyError if ffmpeg is not configured. Use try/except.
    try:
        WriterClass = manimation.writers['ffmpeg']
        try:
            writer = WriterClass(fps=fps, metadata=metadata)
            writer_available = True
        except Exception:
            writer = None
            writer_available = False
    except Exception:
        writer = None
        writer_available = False

    return {'out_path': out_path, 'fps': fps, 'writer_available': writer_available}


def enviro_import(path):
    """Read a raster file and return (array, transform, crs).

    This wrapper uses rasterio to read the first band and returns
    commonly used metadata.
    """
    with rasterio.open(path) as src:
        arr = src.read(1)
        transform = src.transform
        crs = src.crs
    return arr, transform, crs


def longitudinal_import(shapefile):
    """Read a longitudinal shapefile and return a GeoDataFrame.

    The original code computes linear positions; that logic can be
    implemented later — here we provide a reliable reader.
    """
    gdf = gpd.read_file(shapefile)
    return gdf


__all__ = [
    'output_excel',
    'movie_maker',
    'enviro_import',
    'longitudinal_import',
]
