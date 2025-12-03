import numpy as np
import h5py
import os
from pathlib import Path
from scipy.ndimage import distance_transform_edt

# Minimal demo to create a tiny environment HDF and run the centerline extraction
REPO_ROOT = Path(__file__).resolve().parents[1]
HDF_PATH = REPO_ROOT / 'tmp_demo_env.h5'

# Create a simple wetted mask with a meandering channel
h = 200
w = 300
wetted = np.zeros((h, w), dtype=np.float32)
for i in range(h):
    center = int(w/2 + 40.0 * np.sin(i / 20.0))
    left = max(0, center - 6)
    right = min(w, center + 6)
    wetted[i, left:right] = 1.0

# x_coords/y_coords as simple affine grid
x_coords = np.zeros((h, w), dtype=np.float32)
y_coords = np.zeros((h, w), dtype=np.float32)
for i in range(h):
    for j in range(w):
        x_coords[i, j] = j
        y_coords[i, j] = i

# write HDF
with h5py.File(str(HDF_PATH), 'w') as f:
    env = f.create_group('environment')
    env.create_dataset('wetted', data=wetted)
    env.create_dataset('x_coords', data=x_coords)
    env.create_dataset('y_coords', data=y_coords)

print('Wrote demo HDF:', HDF_PATH)

# import the simulation class and run init
import sys
sys.path.insert(0, str(REPO_ROOT))
from src.emergent.salmon_abm.sockeye_SoA_OpenGL_RL import simulation

# The simulation class constructor in this module expects many args; provide minimal required
sim = simulation(
    model_dir=str(REPO_ROOT),
    model_name='tmp_demo',
    crs='EPSG:26904',
    basin='demo',
    water_temp=10.0,
    start_polygon=str(REPO_ROOT / 'data' / 'salmon_abm' / 'starting_location' / 'start_loc_river_right.shp'),
    longitudinal_profile=None,
    env_files=None,
    num_timesteps=10,
    num_agents=10,
    use_hecras=True,
    hecras_plan_path=str(HDF_PATH)
)
print('Sim initialized; longitude:', getattr(sim, 'longitude', None))

# Attach the demo HDF as sim.hdf5 so extraction code can read it
try:
    sim.hdf5 = h5py.File(str(HDF_PATH), 'a')
except Exception:
    print('Could not attach demo HDF to sim.hdf5')

# Run a raster->skeleton extraction using the sim's HDF environment and set sim.longitude
try:
    env = sim.hdf5.get('environment') if sim.hdf5 is not None else None
    distance_to_rast = None
    wetted = None
    x_coords = None
    y_coords = None
    if env is not None:
        if 'distance_to' in env:
            distance_to_rast = np.array(env['distance_to'])
        if 'wetted' in env:
            wetted = np.array(env['wetted'])
        if 'x_coords' in env:
            x_coords = np.array(env['x_coords'])
        if 'y_coords' in env:
            y_coords = np.array(env['y_coords'])

    if distance_to_rast is None and wetted is not None:
        mask = (wetted != -9999) & (wetted > 0)
        distance_to_rast = distance_transform_edt(mask)

    main_centerline = None
    if distance_to_rast is not None:
        from scipy.ndimage import maximum_filter
        from skimage.morphology import skeletonize
        from skimage.measure import label
        from shapely.geometry import LineString, MultiLineString
        from shapely.ops import linemerge

        footprint_size = 5
        local_max = maximum_filter(distance_to_rast, size=footprint_size)
        is_ridge = (distance_to_rast == local_max) & (distance_to_rast > 0.5)
        skeleton = skeletonize(is_ridge)
        labeled = label(skeleton, connectivity=2)
        centerlines = []
        for region_id in range(1, int(labeled.max()) + 1):
            region_mask = (labeled == region_id)
            ys, xs = np.where(region_mask)
            if len(xs) < 5:
                continue
            world_coords = []
            if x_coords is not None and y_coords is not None:
                for i in range(len(xs)):
                    try:
                        if getattr(x_coords, 'ndim', 0) == 2 and getattr(y_coords, 'ndim', 0) == 2:
                            xw = x_coords[ys[i], xs[i]]
                            yw = y_coords[ys[i], xs[i]]
                        else:
                            xw = x_coords[xs[i]] if getattr(x_coords, 'ndim', 0) >= 1 else float(xs[i])
                            yw = y_coords[ys[i]] if getattr(y_coords, 'ndim', 0) >= 1 else float(ys[i])
                    except Exception:
                        xw = float(xs[i])
                        yw = float(ys[i])
                    world_coords.append((xw, yw))
            else:
                for i in range(len(xs)):
                    world_coords.append((float(xs[i]), float(ys[i])))
            if len(world_coords) >= 2:
                centerlines.append(LineString(world_coords))

        if centerlines:
            merged = linemerge(centerlines)
            if isinstance(merged, LineString):
                main_centerline = merged
            elif isinstance(merged, MultiLineString):
                main_centerline = max(merged.geoms, key=lambda g: g.length)

    if main_centerline is not None and main_centerline.length > 1.0:
        # set the canonical `centerline` attribute using the simulation's importer
        sim.centerline = sim.centerline_import(main_centerline)
        # compute per-agent distances along the new centerline if possible
        try:
            sim.current_longitudes = sim.compute_linear_positions(sim.centerline)
        except Exception:
            pass
        print('Extraction set sim.centerline; length:', sim.centerline.length)
    else:
        print('No valid centerline extracted (main_centerline is None or too short)')

    # Save outputs
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 8))
        if env is not None and 'wetted' in env:
            ax.imshow(np.array(env['wetted']), cmap='Blues', origin='lower')
        if getattr(sim, 'centerline', None) is not None:
            xs, ys = sim.centerline.xy
            ax.plot(xs, ys, '-y')
        ax.set_title('Demo centerline extraction')
        outpng = REPO_ROOT / 'outputs' / 'centerline_demo.png'
        fig.savefig(outpng, dpi=150)
        print('Wrote PNG to', outpng)
    except Exception as e:
        print('Could not write PNG:', e)

    try:
        import geopandas as gpd
        if getattr(sim, 'centerline', None) is not None:
            g = gpd.GeoSeries([sim.centerline])
            out = REPO_ROOT / 'outputs' / 'centerline_demo.shp'
            g.to_file(out)
            print('Wrote centerline shapefile to', out)
    except Exception as e:
        print('Could not write shapefile:', e)
except Exception as e:
    print('Centerline extraction failed:', e)

print('Done')
