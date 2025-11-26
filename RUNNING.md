Run instructions for emergent (headless / GUI)

Minimal headless run (no ENC, no AIS, no OFS):

- Recommended: create a virtualenv/conda env with Python 3.10+.
- Install minimal dependencies for headless runs:

PowerShell commands:

```powershell
python -m pip install -r requirements.txt
# If you prefer a minimal set for headless tests:
python -m pip install numpy scipy pyproj
```

- Run the minimal smoke test:

```powershell
python scripts/run_ship_min.py
```

- Run a short headless simulation (spawns simple straight-line waypoints):

```powershell
python scripts/run_sim_min.py
python scripts/run_sim_min.py  # will pick a valid port from config and run
```

GUI run (requires heavy deps):

```powershell
python -m pip install -r requirements.txt
# plus GUI libs
python -m pip install pyqt5 pyqtgraph geopandas shapely fiona rtree
```

Notes:
- `load_enc` can block startup if GDAL/Fiona are not available or ENC files are large. Use `load_enc=False` for fast headless testing.
- If you hit errors in `simulation` or `ship` code, paste the full traceback here and I'll triage.
