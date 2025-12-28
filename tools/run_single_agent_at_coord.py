"""Run a single-agent headless sim with agent placed at specified world coords.

Usage:
  python tools/run_single_agent_at_coord.py --x 548100.0 --y 6641800.0 --preload outputs/diagnostics/preseed_memory_at_falls.h5 --out outputs/diagnostics --model-name single_at_seed

This script creates a 1-agent simulation, preloads memory into the sim DB, sets agent 0 position to the requested coords,
runs one step with --debug-behavior and writes diagnostics.
"""
import os
import argparse
import time

import numpy as np

from emergent.salmon_abm.simulation import simulation
from emergent.salmon_abm import hdf5_io
from emergent.salmon_abm.diagnostics import HDF5DiagnosticsWriter


def run_single(x, y, preload, outdir, model_name):
    outdir = os.path.abspath(outdir)
    os.makedirs(outdir, exist_ok=True)

    sim = simulation(
        model_dir=outdir,
        model_name=model_name,
        crs=None,
        basin='nuyakuk',
        water_temp=10.0,
        start_polygon=None,
        env_files=None,
        longitudinal_profile=None,
        num_timesteps=1,
        num_agents=1,
        db_path=os.path.join(outdir, f'{model_name}_headless.h5')
    )

    # preload memory into the sim DB (if provided)
    if preload and os.path.exists(preload):
        try:
            import h5py
            with h5py.File(preload, 'r') as ph5:
                target = hdf5_io.get_hdf5_obj(sim) or getattr(sim, 'hdf5', None) or getattr(sim, 'db', None)
                if target is None:
                    print('No HDF5 target to preload into')
                else:
                    if 'memory' in ph5:
                        for k in ph5['memory'].keys():
                            try:
                                data = np.array(ph5['memory'][k])
                                hdf5_io.write_dataset(target, f'memory/{k}', data, dtype='f4')
                            except Exception as e:
                                print('Failed to copy memory key', k, e)
                    try:
                        if hasattr(target, 'flush'):
                            target.flush()
                    except Exception:
                        pass
        except Exception as e:
            print('Preload failed:', e)

    # open diagnostics writer and attach
    diag_path = os.path.join(outdir, f'{model_name}_diagnostics.h5')
    dw = HDF5DiagnosticsWriter(diag_path)
    dw.open(mode='a')
    sim.diagnostics_writer = dw

    # set agent 0 position to provided coords
    sim.X[0] = float(x)
    sim.Y[0] = float(y)
    sim.prev_X[0] = float(x)
    sim.prev_Y[0] = float(y)

    # ensure headers and coordinates written to DB so behavior sampling can find env transforms
    try:
        h = hdf5_io.get_hdf5_obj(sim)
        if h is not None:
            try:
                hdf5_io.write_dataset(h, 'X', sim.X)
                hdf5_io.write_dataset(h, 'Y', sim.Y)
            except Exception:
                pass
    except Exception:
        pass

    # enable debug flags so behavior writes NPZ/HDF5
    setattr(sim, 'debug_behavior', True)
    setattr(sim, 'debug_movement', True)

    # run single timestep
    sim.current_step = 0
    sim.timestep(0, 1.0)

    # After stepping, compute authoritative avoid cue using behavior helper
    try:
        dw = getattr(sim, 'diagnostics_writer', None)
        b = getattr(sim, '_behavior', None)
        print('DBG single-run: diagnostics_writer=', type(dw), 'behavior=', type(b))
        if b is not None and dw is not None:
            default_weights = {'avoid': 25000}
            try:
                print('DBG single-run: calling already_been_here')
                avoid = b.already_been_here(default_weights['avoid'], 0)
                print('DBG single-run: avoid returned type=', type(avoid), 'shape=', getattr(avoid, 'shape', None))
            except Exception as ex:
                avoid = None
                print('DBG single-run: already_been_here raised:', ex)
            if avoid is not None:
                try:
                    payload = {}
                    payload['avoid_vec'] = np.asarray(avoid).astype(float)
                    try:
                        payload['avoid_mag'] = np.linalg.norm(payload['avoid_vec'], axis=1)
                    except Exception:
                        payload['avoid_mag'] = np.asarray([np.linalg.norm(payload['avoid_vec'])])
                    try:
                        print('DBG single-run: writing step to HDF5, dw.file=', getattr(dw, 'file', None))
                    except Exception:
                        pass
                    dw.write_step(0, payload)
                    print('Wrote avoid diagnostics to HDF5 for step 0')
                except Exception as e:
                    print('Failed to write avoid diagnostics:', e)
        else:
            print('DBG single-run: missing behavior or diagnostics writer; cannot write avoid')
    except Exception as e:
        print('DBG single-run: unexpected error computing/writing avoid:', e)

    # flush and close
    try:
        if hasattr(dw, 'close'):
            dw.close()
    except Exception:
        pass
    sim.close()
    print('Run complete. Diagnostics written to', diag_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--x', type=float, required=True)
    parser.add_argument('--y', type=float, required=True)
    parser.add_argument('--preload', type=str, default='outputs/diagnostics/preseed_memory_at_falls.h5')
    parser.add_argument('--out', type=str, default='outputs/diagnostics')
    parser.add_argument('--model-name', type=str, default='single_at_seed')
    args = parser.parse_args()
    run_single(args.x, args.y, args.preload, args.out, args.model_name)
