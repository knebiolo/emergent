"""Smoke test for HECRAS centerline derivation using OpenGL_RL module.

This creates a minimal simulation-like object and invokes the centerline
initialization path in `sockeye_SoA_OpenGL_RL.py` to confirm a centerline
is derived from `hecras_run.h5`.
"""
import os
import h5py
import traceback

# adjust import path
import sys
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.emergent.salmon_abm import sockeye_SoA_OpenGL_RL as rlmod

class DummySim:
    def __init__(self, hdf_path):
        self.hdf5 = h5py.File(hdf_path, 'r+')
        self.num_agents = 10
        self.flush_interval = 1
        self.num_timesteps = 1
        self._hdf5_buffers = {}
        self._buffer_pos = 0
        # minimal placeholders
        self.hecras_plan_path = hdf_path
        self.hecras_fields = ['Cells Minimum Elevation']

    def close(self):
        try:
            self.hdf5.close()
        except Exception:
            pass


def run_smoke(hdf_path):
    sim = DummySim(hdf_path)
    try:
        # Instantiate the RL simulation class if present, otherwise call the
        # standalone centerline init helper paths. Many modules expect a class
        # named 'Simulation' - check for it.
        Simulation = None
        for name in dir(rlmod):
            obj = getattr(rlmod, name)
            if isinstance(obj, type) and name.lower().startswith('simulation'):
                Simulation = obj
                break
        if Simulation is not None:
            print('Found Simulation class in module; instantiating with HECRAS plan...')
            # Attempt a best-effort instantiation with minimal args
            try:
                s = Simulation(centerline=None, hecras_plan_path=hdf_path, use_hecras=True)
                # check centerline
                cl = getattr(s, 'centerline', None)
                print('centerline from Simulation:', bool(cl), type(cl))
                return s
            except Exception as e:
                print('Failed to instantiate Simulation class:', e)
                traceback.print_exc()
        # fallback: call functions that operate on a sim object
        print('Falling back to calling helper functions directly...')
        with h5py.File(hdf_path, 'r') as hf:
            env = hf.get('environment')
            if env is not None and 'distance_to' in env:
                distance = None
                try:
                    distance = np.array(env['distance_to'])
                except Exception:
                    pass
                if distance is not None:
                    # call derive_centerline_from_distance_raster directly
                    try:
                        from src.emergent.salmon_abm.sockeye_SoA_OpenGL_RL import derive_centerline_from_distance_raster
                        cl, all_lines = derive_centerline_from_distance_raster(distance)
                        print('Derived centerline via helper:', bool(cl), 'num_lines=', len(all_lines))
                        return cl
                    except Exception as e:
                        print('Helper failed:', e)
                        traceback.print_exc()
        print('No distance raster found in HDF; smoke test cannot proceed.')
    finally:
        sim.close()

if __name__ == '__main__':
    import numpy as np
    hdf_path = os.path.abspath(os.path.join(repo_root, 'hecras_run.h5'))
    if not os.path.exists(hdf_path):
        print('HECRAS HDF not found at', hdf_path)
        sys.exit(2)
    run_smoke(hdf_path)
