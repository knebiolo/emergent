"""Run a short simulation loop under cProfile and report hotspots.

Usage:
    python tools/profile_timestep_cprofile.py --agents 100 --timesteps 10

This script creates a small simulation configured to use HECRAS per-agent sampling
but without writing full-grid rasters. It profiles the `sim.timestep()` calls and
prints a top-50 report by cumulative time and writes a pstats file `profile.pstats`.
"""
import argparse
import cProfile
import pstats
import io
import os
import sys
import numpy as np
import h5py

# Allow running from repo root when executed from tools/
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from emergent.salmon_abm.sockeye_SoA import simulation


def build_sim(num_agents, use_hecras=True, hecras_write_rasters=False):
    # Build minimal env_files mapping similar to other profiling tools
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    model_dir = os.path.join(repo_root, 'outputs', 'dali_scenario')
    env_files = {
        'x_vel': 'vel_x.tif',
        'y_vel': 'vel_y.tif',
        'depth': 'depth.tif',
        'wsel': 'wsel.tif',
        'elev': 'elev.tif',
        'vel_dir': 'vel_dir.tif',
        'vel_mag': 'vel_mag.tif',
        'wetted': 'wetted.tif'
    }
    env_files = {k: os.path.join(model_dir, v) for k, v in env_files.items()}

    # find or create a tiny start polygon
    start_polygon = None
    for p in (os.path.join(repo_root, 'data')).replace('\\', '/') and []:
        pass

    # use user's longitudinal shapefile if present
    user_long_dir = r"C:\Users\Kevin.Nebiolo\OneDrive - Kleinschmidt Associates\Software\emergent\data\salmon_abm\Longitudinal"
    tmp = os.path.join(repo_root, 'tools', 'temp_start_poly.geojson')
    line_tmp = os.path.join(repo_root, 'tools', 'temp_longitudinal_line.geojson')
    longitudinal_profile = None
    try:
        if os.path.isdir(user_long_dir):
            # pick the first shapefile in the folder
            for f in os.listdir(user_long_dir):
                if f.lower().endswith('.shp'):
                    longitudinal_profile = os.path.join(user_long_dir, f)
                    break
    except Exception:
        longitudinal_profile = None
    if not os.path.exists(tmp):
        import json
        geom = {
            "type": "FeatureCollection",
            "features": [
                {"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [[[0,0],[0,1],[1,1],[1,0],[0,0]]]}}
            ]
        }
        os.makedirs(os.path.dirname(tmp), exist_ok=True)
        with open(tmp, 'w') as f:
            json.dump(geom, f)
    start_polygon = tmp
    # create a simple LineString geojson for longitudinal_profile if no user file found
    if longitudinal_profile is None:
        if not os.path.exists(line_tmp):
            import json
            line = {
                "type": "FeatureCollection",
                "features": [
                    {"type": "Feature", "properties": {}, "geometry": {"type": "LineString", "coordinates": [[0,0],[1,1],[2,2]]}}
                ]
            }
            with open(line_tmp, 'w') as f:
                json.dump(line, f)
        longitudinal_profile = line_tmp

    # Instantiate simulation with required args
    sim = simulation(model_dir=str(model_dir), model_name='prof_test', crs='EPSG:4326', basin='Nushagak River',
                     water_temp=10.0, start_polygon=start_polygon, env_files=env_files,
                     longitudinal_profile=longitudinal_profile, fish_length=500, num_timesteps=100, num_agents=num_agents, use_gpu=False,
                     hecras_plan_path=None, hecras_fields=None, hecras_k=8, use_hecras=use_hecras, hecras_write_rasters=hecras_write_rasters,
                     defer_hdf=True, defer_log_dir=os.path.join(repo_root, 'tools', 'profile_log'))
    # Ensure mental_map_transform exists to satisfy geo_to_pixel
    try:
        import rasterio
        from rasterio.transform import Affine
        sim.mental_map_transform = Affine.translation(0, 0)
    except Exception:
        class _AffStub:
            def __invert__(self):
                return self
        sim.mental_map_transform = _AffStub()

    # Provide minimal raster transform placeholders expected by environment sampling
    class _AffStub:
        def __init__(self):
            # coefficients roughly matching an identity transform
            self.a = 1.0
            self.b = 0.0
            self.c = 0.0
            self.d = 0.0
            self.e = 1.0
            self.f = 0.0
        def __invert__(self):
            return self
        def __mul__(self, other):
            # emulate Affine * (cols, rows) -> (x, y)
            try:
                cols, rows = other
                x = self.c + self.a * (cols + 0.5) + self.b * (rows + 0.5)
                y = self.f + self.d * (cols + 0.5) + self.e * (rows + 0.5)
                return (x, y)
            except Exception:
                return other
    for name in ('vel_dir_rast_transform','vel_mag_rast_transform','depth_rast_transform','x_coords_transform','y_coords_transform'):
        if not hasattr(sim, name):
            setattr(sim, name, _AffStub())

    # Provide minimal raster dimensions expected by environment sampling
    if not hasattr(sim, 'width'):
        sim.width = 200
    if not hasattr(sim, 'height'):
        sim.height = 200

    # Minimal agents_within_buffers to satisfy alignment/avoid cues
    if not hasattr(sim, 'agents_within_buffers'):
        sim.agents_within_buffers = [np.array([], dtype=np.int32)]

    # Create a small temporary HDF5 file with minimal environment datasets
    temp_h5 = os.path.join(repo_root, 'tools', 'temp_profile_env.h5')
    try:
        with h5py.File(temp_h5, 'w') as f:
            env = f.create_group('environment')
            shape = (sim.height, sim.width)
            data_zero = np.zeros(shape, dtype='f4')
            data_one = np.ones(shape, dtype='f4')
            # create datasets under /environment
            env.create_dataset('vel_mag', shape, dtype='f4', data=data_zero)
            env.create_dataset('vel_dir', shape, dtype='f4', data=data_zero)
            env.create_dataset('vel_x', shape, dtype='f4', data=data_zero)
            env.create_dataset('vel_y', shape, dtype='f4', data=data_zero)
            env.create_dataset('depth', shape, dtype='f4', data=data_zero)
            env.create_dataset('wetted', shape, dtype='f4', data=data_one)
            env.create_dataset('distance_to', shape, dtype='f4', data=data_zero)
            # Also provide x_coords and y_coords arrays matching raster centers
            xs = np.tile(np.arange(sim.width, dtype='f4'), (sim.height,1))
            ys = np.tile(np.arange(sim.height, dtype='f4')[:,None], (1,sim.width))
            env.create_dataset('x_coords', shape, dtype='f4', data=xs)
            env.create_dataset('y_coords', shape, dtype='f4', data=ys)
            # Duplicate top-level datasets for code paths expecting root-level names
            for name in ('vel_mag','vel_dir','vel_x','vel_y','depth','wetted','distance_to','x_coords','y_coords'):
                f.create_dataset(name, data=env[name])
            # provide an empty refugia group with dataset '0' to satisfy lookups
            refug = f.create_group('refugia')
            refug.create_dataset('0', shape, dtype='f4', data=np.zeros(shape, dtype='f4'))
            # minimal memory and migrate groups
            mem = f.create_group('memory')
            mem.create_dataset('0', shape, dtype='f4', data=np.zeros(shape, dtype='f4'))
            mig = f.create_group('migrate')
            mig.create_dataset('0', shape, dtype='f4', data=np.zeros(shape, dtype='f4'))
    except Exception:
        # best-effort; if file creation fails, continue without it
        temp_h5 = None

    # attach the hdf5 file to the simulation (open for read)
    if temp_h5 is not None and os.path.exists(temp_h5):
        fobj = h5py.File(temp_h5, 'r')
        # some code paths access sim.simulation.hdf5 - ensure both references exist
        sim.hdf5 = fobj
        try:
            sim.simulation = type('S', (), {})()
            sim.simulation.hdf5 = fobj
        except Exception:
            pass

    return sim


def profile_run(num_agents, timesteps, out_stats='profile.pstats'):
    sim = build_sim(num_agents)

    pr = cProfile.Profile()
    pr.enable()

    # timestep signature requires (t, dt, g, pid_controller) — pass simple placeholders
    # Create a minimal PID stub if code expects PID controller
    # instantiate the real PID controller from the module
    from emergent.salmon_abm.sockeye_SoA import PID_controller
    pid = PID_controller(num_agents, k_p=1.0, k_i=0.0, k_d=0.0)
    try:
        pid.interp_PID()
    except Exception:
        pass
    # simplify behavior arbitration to avoid needing full environment cues
    try:
        if hasattr(sim, 'behavior') and sim.behavior is not None:
            sim.behavior.arbitrate = (lambda self_obj, tt: getattr(sim, 'heading', np.zeros(getattr(sim, 'num_agents', 1))))
    except Exception:
        pass
    for t in range(timesteps):
        sim.timestep(t, 1.0, 9.81, pid)

    pr.disable()
    ps = pstats.Stats(pr).sort_stats('cumulative')
    ps.dump_stats(out_stats)

    s = io.StringIO()
    ps.stream = s
    ps.print_stats(50)
    print(s.getvalue())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--agents', type=int, default=100)
    parser.add_argument('--timesteps', type=int, default=10)
    parser.add_argument('--out', default='profile.pstats')
    args = parser.parse_args()

    profile_run(args.agents, args.timesteps, args.out)


if __name__ == '__main__':
    main()
