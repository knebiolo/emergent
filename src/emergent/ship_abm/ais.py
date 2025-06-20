# -*- coding: utf-8 -*-

"""# AIS Data Automation Module

This document outlines the design and implementation of a Python module to automate the download, filtering, and rasterization of AIS (Automatic Identification System) data from NOAA’s AIS archive. The goal is to:

1. Download daily AIS ZIP files from NOAA for a specified date range without permanently storing them on disk.
2. Filter the AIS CSV data by a user-defined geographic bounding box (latitude/longitude).
3. Generate a 2D heatmap (raster) representing AIS ping density within that bounding box.
4. Discard each daily ZIP immediately after processing to minimize storage usage.

This module can be integrated into the existing Emergent project code base, typically under a new file `ais_noaa.py` in the `emergent/ship_abm/` directory.

---

## 1. Module Overview

* **Module Name**: `ais_noaa.py`
* **Dependencies**:

  * `requests` (for HTTP downloads)
  * `zipfile` (for in-memory ZIP handling)
  * `io` (for treating downloaded bytes as a file)
  * `datetime` (for date arithmetic)
  * `numpy` (for array manipulation and histogram generation)
  * `pandas` (for reading CSV data efficiently)
  * `logging` (optional, for progress/debug statements)

### 1.1 Core Function


from datetime import date
from typing import Tuple
import numpy as np

def compute_ais_heatmap(
    bbox: Tuple[float, float, float, float],  # (min_lon, min_lat, max_lon, max_lat)
    start_date: date,
    end_date: date,
    grid_size: Tuple[int, int] = (500, 500),   # (nx, ny)
    year: int = None                          # If None, infer from start_date
) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:

    Download daily AIS data from NOAA, filter to a bounding box, accumulate a 2D histogram heatmap, and return it.

    Parameters
    ----------
    bbox : (min_lon, min_lat, max_lon, max_lat)
    start_date : datetime.date, inclusive
    end_date : datetime.date, inclusive
    grid_size : (nx, ny) = resolution of histogram bins
    year : int = archive year (if None, will be set to start_date.year)

    Returns
    -------
    heatmap : np.ndarray, shape (ny, nx)
        The cumulative AIS ping count per cell.
    extent : (min_lon, max_lon, min_lat, max_lat)
        To use in `matplotlib.pyplot.imshow(..., extent=extent)`

    Notes
    -----
    - The function streams each daily ZIP from NOAA, reads the CSV content in-memory, filters to `bbox`,
      computes a daily 2D histogram, accumulates it, and discards the ZIP.
    - If a daily file is missing (HTTP 404), it is logged and skipped.
    - Memory footprint is kept minimal by only reading needed columns ("LON", "LAT").

    # Implementation below...

### 1.2 Assumptions & Notes

* NOAA’s AIS data archive URL pattern:

  ```
  https://coast.noaa.gov/htdata/CMSP/AISDataHandler/{year}/AIS_{year}_{MM:02d}_{DD:02d}.zip
  ```
* Each ZIP contains a CSV with at least `LAT` and `LON` columns (WGS84).
* Very large daily ZIPs (\~100–300 MB) will be streamed into memory via `requests`.
* We rely on `pandas.read_csv()` to read columns in-memory, without writing to disk.
* If memory use becomes problematic, we could fallback to writing the ZIP to a temporary file and streaming from there,
  but this implementation attempts fully in-memory handling first.


"""

# ais_noaa.py
import io
import zipfile
import requests
from datetime import timedelta, date
import numpy as np
import pandas as pd
import logging
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configure module-level logging (optional)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)


def compute_ais_heatmap(
    bbox: tuple,
    start_date: date,
    end_date: date,
    grid_size: tuple = (5000, 5000),
    year = None
) -> tuple:
    """
    Download daily AIS ZIP files from NOAA for the specified date range, filter by bbox,
    accumulate a 2D histogram (heatmap), and return that heatmap plus its geographic extent.

    Parameters
    ----------
    bbox : tuple of floats
        (min_lon, min_lat, max_lon, max_lat) in WGS84.
    start_date : datetime.date
        Start date (inclusive).
    end_date : datetime.date
        End date (inclusive).
    grid_size : tuple of int, optional
        (nx, ny) number of histogram bins (columns by rows). Default = (500, 500).
    year : int, optional
        If provided, use this year for URL construction. Otherwise, defaults to start_date.year.

    Returns
    -------
    heatmap : np.ndarray, shape = (ny, nx)
        Cumulative AIS ping counts per grid cell.
    extent : tuple of floats
        (min_lon, max_lon, min_lat, max_lat) for plotting.
    """
    # Unpack bounding box
    min_lon, min_lat, max_lon, max_lat = bbox
    nx, ny = grid_size

    if not isinstance(start_date, date):
        start_date = pd.to_datetime(start_date).date()
    if not isinstance(end_date, date):
        end_date = pd.to_datetime(end_date).date()
    # Determine the year if not explicitly provided
    if year is None:
        year = start_date.year

    # Initialize accumulator (shape: rows=ny, cols=nx)
    accum = np.zeros((ny, nx), dtype=float)

    # Iterate over each date
    n_days = (end_date - start_date).days + 1
    for day_offset in range(n_days):
        current = start_date + timedelta(days=day_offset)
        mm = current.month
        dd = current.day

        # Construct NOAA AIS URL for this date
        url = (
            f"https://coast.noaa.gov/htdata/CMSP/"
            f"AISDataHandler/{year}/AIS_{year}_{mm:02d}_{dd:02d}.zip"
        )
        log.info(f"Downloading AIS data for {current.isoformat()}: {url}")

        try:
            response = requests.get(url, stream=True, timeout=60, verify = False)
            response.raise_for_status()
        except requests.HTTPError as e:
            log.warning(f"HTTPError for {url} - skipping date: {e}")
            continue
        except requests.RequestException as e:
            log.warning(f"Connection error for {url} - skipping date: {e}")
            continue

        # Open ZIP in-memory
        try:
            zf = zipfile.ZipFile(io.BytesIO(response.content))
        except zipfile.BadZipFile:
            log.error(f"Bad ZIP file on {current.isoformat()} - skipping")
            continue

        # Find CSV within ZIP
        csv_candidates = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not csv_candidates:
            log.error(f"No CSV found in ZIP for {current.isoformat()} - skipping")
            zf.close()
            continue

        csv_name = csv_candidates[0]
        log.debug(f"Found CSV in ZIP: {csv_name}")

        # Read CSV columns 'LON' and 'LAT' only, directly from ZIP
        try:
            with zf.open(csv_name) as csvfile:
                df = pd.read_csv(
                    csvfile,
                    usecols=["LON", "LAT"],
                    dtype={"LON": float, "LAT": float},
                )
        except Exception as e:
            log.error(f"Error reading CSV for {current.isoformat()}: {e}")
            zf.close()
            continue

        zf.close()

        # Filter to bounding box
        mask = (
            (df["LON"] >= min_lon)
            & (df["LON"] <= max_lon)
            & (df["LAT"] >= min_lat)
            & (df["LAT"] <= max_lat)
        )
        df = df.loc[mask]

        if df.empty:
            log.info(f"No AIS pings in bbox for {current.isoformat()}")
            continue

        # Compute histogram for this day's points
        lons = df["LON"].values
        lats = df["LAT"].values
        hm, xedges, yedges = np.histogram2d(
            lons,
            lats,
            bins=[nx, ny],
            range=[[min_lon, max_lon], [min_lat, max_lat]],
        )
        # Note: histogram2d returns shape (nx, ny); transpose to (ny, nx)
        accum += hm.T
        log.info(f"Processed {df.shape[0]} pings; updated heatmap.")

        # The ZIP was read in-memory; no on-disk cleanup needed.

    extent = (min_lon, max_lon, min_lat, max_lat)
    return accum, extent
