"""Probe behavior outputs without running the full simulation loop.

Creates a short `simulation` instance, imports environment rasters into HDF5,
and calls `behavior.arbitrate(0)` to capture `head_vec` and cue magnitudes.
Writes `outputs/diagnostics/behavior_probe.json` with the results.
"""
import os
import json
import numpy as np

from emergent.salmon_abm.simulation import simulation
from emergent.salmon_abm import io, hdf5_io
from emergent.salmon_abm import behavior as behavior_mod


def discover_env_files(base_dir):
    keys = ['depth.tif', 'vel_x.tif', 'vel_y.tif', 'vel_mag.tif', 'vel_dir.tif']
    out = []
    for fn in keys:
        p = os.path.join(base_dir, fn)
        if os.path.exists(p):
            out.append(p)
    return out


def import_env_to_h5(sim, env_files):
    try:
        h5 = hdf5_io.get_hdf5_obj(sim)
        for ef in env_files:
            try:
                arr, transform, crs = io.enviro_import(ef)
                key = 'environment/' + os.path.splitext(os.path.basename(ef))[0]
                hdf5_io.write_dataset(h5, key, np.array(arr))
                print('Imported raster into HDF5:', key)
                if os.path.basename(ef).startswith('depth'):
                    try:
                        t = transform
                        sim.depth_rast_transform = (t.a, t.b, t.c, t.d, t.e, t.f)
                    except Exception:
                        pass
            except Exception as e:
                print('Failed to import raster', ef, e)
    except Exception as e:
        print('Raster import into HDF5 failed:', e)


def run_probe(nagents=12):
    base = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'data', 'salmon_abm'))
    base = os.path.abspath(base)
    env_files = discover_env_files(base)

    outdir = os.path.abspath(os.path.join('outputs', 'diagnostics'))
    os.makedirs(outdir, exist_ok=True)

    sim = simulation(
        model_dir=outdir,
        model_name='behavior_probe',
        crs=None,
        basin='nuyakuk',
        water_temp=10.0,
        start_polygon=None,
        env_files=env_files,
        longitudinal_profile=None,
        num_timesteps=1,
        num_agents=nagents,
        db_path=os.path.join(outdir, 'behavior_probe.h5')
    )

    # import environment rasters for sampling
    import_env_to_h5(sim, env_files)

    # ensure behavior helper present
    beh = behavior_mod.behavior(1.0, sim)
    # enable debug flag so behavior stores last_head_vec
    sim.debug_behavior = True

    # call arbitrate once
    # Inspect individual cue shapes to find broadcasting offenders
    cue_shape_report = {}
    # record simulation heading shape for debugging
    try:
        cue_shape_report['simulation_heading'] = {'ndim': np.asarray(sim.heading).ndim, 'shape': np.asarray(sim.heading).shape, 'dtype': str(np.asarray(sim.heading).dtype)}
    except Exception as e:
        cue_shape_report['simulation_heading'] = f'error: {e}'
    try:
        try:
            cue_shape_report['rheotaxis'] = np.asarray(beh.rheo_cue(25000)).shape
        except Exception as e:
            cue_shape_report['rheotaxis'] = f'error: {e}'
        import traceback
        for fn_name, fn, args in [
            ('alignment', beh.alignment_cue, (20500,)),
            ('cohesion', beh.cohesion_cue, (11000,)),
            ('low_speed', beh.vel_cue, (1500,)),
            ('wave_drag', beh.wave_drag_cue, (0,)),
            ('refugia', beh.find_nearest_refuge, (50000,)),
            ('border', beh.border_cue, (50000, 0)),
            ('shallow', beh.shallow_cue, (100000,)),
            ('avoid', beh.already_been_here, (25000, 0)),
            ('collision', beh.collision_cue, (50000,)),
        ]:
            try:
                arr = fn(*args)
                cue_shape_report[fn_name] = {'ndim': np.asarray(arr).ndim, 'shape': np.asarray(arr).shape}
            except Exception as e:
                cue_shape_report[fn_name] = {'error': str(e), 'traceback': traceback.format_exc()}
    except Exception:
        pass

    try:
        head = beh.arbitrate(0)
    except Exception as e:
        print('behavior.arbitrate failed:', e)
        head = None

    out = {'head_shape': None, 'head_preview': None, 'cue_magnitudes_keys': [], 'cue_shape_report': cue_shape_report}
    try:
        if hasattr(sim, 'last_head_vec'):
            out['head_shape'] = np.asarray(sim.last_head_vec).shape
            out['head_preview'] = np.asarray(sim.last_head_vec).tolist()[:5]
        if hasattr(sim, 'last_cue_magnitudes'):
            out['cue_magnitudes_keys'] = list(sim.last_cue_magnitudes.keys())
            # include small sample values for a few cues
            samples = {}
            for k, v in sim.last_cue_magnitudes.items():
                try:
                    samples[k] = np.asarray(v).tolist()[:5]
                except Exception:
                    samples[k] = None
            out['cue_magnitudes_sample'] = samples
    except Exception as e:
        out['error'] = str(e)

    outfn = os.path.join(outdir, 'behavior_probe.json')
    with open(outfn, 'w', encoding='utf-8') as fh:
        json.dump(out, fh, indent=2)
    print('Wrote probe summary to', outfn)
    sim.close()


if __name__ == '__main__':
    run_probe()
