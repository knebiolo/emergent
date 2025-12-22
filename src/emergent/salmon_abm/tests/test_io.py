import os
import tempfile
import numpy as np
from pathlib import Path
import importlib.util

# load io module by path to avoid package __init__ side effects
tests_dir = Path(__file__).resolve().parent
pkg_dir = tests_dir.parent
io_path = pkg_dir / 'io.py'
spec = importlib.util.spec_from_file_location('salmon_abm.io', str(io_path))
io = importlib.util.module_from_spec(spec)
spec.loader.exec_module(io)


def test_output_excel_creates_file():
    # use a repo-local temp dir to avoid permission issues with system temp
    base = Path(__file__).resolve().parent
    model_dir = base / '_tmp_io'
    model_dir.mkdir(exist_ok=True)
    records = {'gen1': __import__('pandas').DataFrame({'a': [1, 2, 3]})}
    io.output_excel(records, str(model_dir), 'testmodel')
    out_xlsx = model_dir / 'output_testmodel.xlsx'
    out_csv = model_dir / 'testmodel_gen1.csv'
    # Accept either Excel output or CSV fallback
    assert out_xlsx.exists() or out_csv.exists()


def test_movie_maker_returns_report():
    base = Path(__file__).resolve().parent
    outdir = base / '_tmp_io'
    outdir.mkdir(exist_ok=True)
    report = io.movie_maker(str(outdir), 'mymodel', crs=None, dt=0.5, depth_rast_transform=None, depth_arr=np.zeros((2,2)))
    assert 'out_path' in report
    assert report['fps'] >= 1


def test_enviro_import_tmpfile():
    # create a small GeoTIFF using rasterio
    import rasterio
    from rasterio.transform import from_origin

    data = np.arange(9, dtype=np.float32).reshape(3, 3)
    base = Path(__file__).resolve().parent
    tmpdir = base / '_tmp_io'
    tmpdir.mkdir(exist_ok=True)
    out_file = tmpdir / 'test.tif'
    transform = from_origin(0, 3, 1, 1)
    with rasterio.open(
        out_file,
        'w',
        driver='GTiff',
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=data.dtype,
        crs='+proj=latlong',
        transform=transform,
    ) as dst:
        dst.write(data, 1)

    arr, tr, crs = io.enviro_import(str(out_file))
    assert arr.shape == data.shape
    assert tr == transform
