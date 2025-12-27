"""Headless runner for Nuyakuk diagnostic traces.

Writes per-timestep per-agent CSV with positions, velocities and sampled env values.

Usage:
  python tools/run_nuyakuk_headless.py --nagents 200 --nsteps 200 --out outputs/diagnostics
"""
import os
import argparse
import time
import csv

import numpy as np
import h5py

from emergent.salmon_abm.simulation import simulation
from emergent.salmon_abm import io, hdf5_io
from emergent.salmon_abm.diagnostics import HDF5DiagnosticsWriter


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
        # Use centralized helper which writes raster into HDF5 and attaches
        # a plain (a,b,c,d,e,f) transform tuple onto `sim` when provided.
        h5 = hdf5_io.get_hdf5_obj(sim)
        for ef in env_files:
            try:
                arr, tr_tup, crs = io.write_raster_to_hdf5(h5, ef, dataset_name=None, sim=sim)
                key = 'environment/' + os.path.splitext(os.path.basename(ef))[0]
                print('Imported raster into HDF5:', key)
            except Exception as e:
                print('Failed to import raster', ef, e)
        # write x/y coords if depth exists (helper may have set sim.depth_rast_transform)
        try:
            depth_ds = hdf5_io.read_dataset(h5, 'environment/depth')
            if depth_ds is not None:
                depth_arr = np.array(depth_ds)
                nrows, ncols = depth_arr.shape
                a, b, c, d, e, f = getattr(sim, 'depth_rast_transform', (1.0, 0.0, 0.0, 0.0, 1.0, 0.0))
                cols = np.arange(ncols, dtype=float)
                rows = np.arange(nrows, dtype=float)
                col_indices, row_indices = np.meshgrid(cols, rows)
                x_coords = a * col_indices + b * row_indices + c
                y_coords = d * col_indices + e * row_indices + f
                hdf5_io.write_dataset(h5, 'environment/x_coords', x_coords)
                hdf5_io.write_dataset(h5, 'environment/y_coords', y_coords)
        except Exception:
            pass
    except Exception as e:
        print('Raster import into HDF5 failed:', e)


def run_headless(args):
    base = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'data', 'salmon_abm'))
    base = os.path.abspath(base)
    env_files = discover_env_files(base)
    start_poly_default = os.path.join(base, 'start_loc_river_right.shp')
    # allow overriding start polygon from CLI
    start_poly = args.start_polygon if getattr(args, 'start_polygon', None) else start_poly_default

    outdir = os.path.abspath(args.out)
    os.makedirs(outdir, exist_ok=True)

    sim = simulation(
        model_dir=outdir,
        model_name=args.model_name,
        crs=None,
        basin='nuyakuk',
        water_temp=10.0,
        start_polygon=start_poly if (start_poly and os.path.exists(start_poly)) else None,
        env_files=env_files,
        longitudinal_profile=os.path.join(base, 'longitudinal.shp') if os.path.exists(os.path.join(base, 'longitudinal.shp')) else None,
        num_timesteps=args.nsteps,
        num_agents=args.nagents,
        db_path=os.path.join(outdir, f'{args.model_name}_headless.h5')
    )

    # If provided, preload memory HDF5 into sim.hdf5['memory/*'] before starting
    if getattr(args, 'preload_memory', None):
        try:
            preload_path = os.path.abspath(args.preload_memory)
            if os.path.exists(preload_path):
                print('Preloading memory from', preload_path)
                with h5py.File(preload_path, 'r') as ph5:
                    # ensure sim has memory group
                    if 'memory' in ph5:
                        for key in ph5['memory'].keys():
                            dst = f'memory/{key}'
                            try:
                                data = np.array(ph5['memory'][key])
                                # write into sim.hdf5, create dataset if missing
                                try:
                                    if dst in sim.hdf5:
                                        sim.hdf5[dst][:] = data
                                    else:
                                        sim.hdf5.create_dataset(dst, data=data, dtype='f4')
                                except Exception:
                                    try:
                                        sim.hdf5[dst] = data
                                    except Exception:
                                        pass
                            except Exception:
                                pass
                try:
                    sim.hdf5.flush()
                except Exception:
                    pass
        except Exception as e:
            print('Failed to preload memory:', e)

    # Open HDF5 diagnostics writer for this run and attach to sim
    try:
        diag_path = os.path.join(outdir, f'{args.model_name}_diagnostics.h5')
        diag_writer = HDF5DiagnosticsWriter(diag_path)
        diag_writer.open(mode='a')
        sim.diagnostics_writer = diag_writer
        print('Opened HDF5 diagnostics:', diag_path)
    except Exception:
        sim.diagnostics_writer = None

    # apply optional deterministic seed
    if getattr(args, 'seed', None) is not None:
        try:
            seed = int(args.seed)
            # numpy global seed for deterministic numpy operations
            np.random.seed(seed)
            # attach a numpy Generator to sim to be used by components
            try:
                sim.rng = np.random.default_rng(seed)
            except Exception:
                sim.rng = None
            print('Applied deterministic seed:', seed)
        except Exception:
            pass

    # load optional test weights JSON and attach to sim
    if getattr(args, 'test_weights_file', None):
        try:
            import json
            with open(args.test_weights_file, 'r', encoding='utf-8') as fh:
                tw = json.load(fh)
            sim.test_weights = tw
            print('Loaded test weights from', args.test_weights_file)
        except Exception as e:
            print('Failed to load test weights file:', e)

    # enable optional movement debugging
    if getattr(args, 'debug_movement', False):
        setattr(sim, 'debug_movement', True)
    if getattr(args, 'debug_behavior', False):
        setattr(sim, 'debug_behavior', True)

    # import rasters into HDF5 for sampling
    import_env_to_h5(sim, env_files)

    # After rasters are imported into the HDF5 DB, re-run heading initialization
    # so agents sample the velocity rasters and start with realistic headings
    # and component velocities. This fixes headless runs that constructed the
    # simulation before rasters existed (which left headings at zero).
    try:
        sim.initialize_headings_from_db()
        # recompute initial fish velocity from newly-initialized heading/ideal_sog
        try:
            fv_x = sim.ideal_sog * np.cos(sim.heading)
            fv_y = sim.ideal_sog * np.sin(sim.heading)
            sim.initial_fish_vel = np.stack((fv_x, fv_y), axis=1)
        except Exception:
            pass
        print('Reinitialized headings from imported rasters')
    except Exception as e:
        print('initialize_headings_from_db after import failed:', e)

    # report sampling validity for initial positions
    try:
        depth_vals0 = sim.sample_environment(getattr(sim, 'depth_rast_transform', None), 'depth')
        valid_depth = int(np.sum(np.isfinite(depth_vals0) & (depth_vals0 != -9999.0)))
        valid_report = os.path.join(outdir, f'{args.model_name}_initial_sampling.json')
        try:
            import json
            with open(valid_report, 'w', encoding='utf-8') as fh:
                json.dump({'num_agents': int(sim.num_agents), 'valid_depth_samples': int(valid_depth)}, fh, indent=2)
            print('Wrote initial sampling report to', valid_report)
        except Exception:
            pass
    except Exception:
        pass

    csv_path = os.path.join(outdir, f'{args.model_name}_trace.csv')
    print('Writing trace to', csv_path)
    header = ['timestep', 'agent', 'x', 'y', 'x_vel', 'y_vel', 'depth', 'vel_x_sample', 'vel_y_sample', 'vel_mag_sample']
    with open(csv_path, 'w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(header)

        # run short deterministic simulation
        dt = 1.0
        for t in range(args.nsteps):
            # expose current step so movement debug filenames are meaningful
            setattr(sim, 'current_step', int(t))
            sim.timestep(t, dt)
            # debug: report whether behavior populated last_cue_vecs for this step
            try:
                if getattr(sim, 'debug_behavior', False):
                    has_raw = hasattr(sim, 'last_cue_vecs') and isinstance(getattr(sim, 'last_cue_vecs', None), dict)
                    if has_raw:
                        try:
                            keys = list(sim.last_cue_vecs.keys())
                        except Exception:
                            keys = ['<err>']
                    else:
                        keys = []
                    print('RUNNER DBG: step=', t, 'has_last_cue_vecs=', has_raw, 'keys=', keys)
            except Exception:
                pass
            # sample environment values at agent positions
            depth_vals = sim.sample_environment(getattr(sim, 'depth_rast_transform', None), 'depth')
            velx_vals = sim.sample_environment(getattr(sim, 'vel_x_rast_transform', getattr(sim, 'depth_rast_transform', None)), 'vel_x')
            vely_vals = sim.sample_environment(getattr(sim, 'vel_y_rast_transform', getattr(sim, 'depth_rast_transform', None)), 'vel_y')
            mag_vals = sim.sample_environment(getattr(sim, 'vel_mag_rast_transform', getattr(sim, 'depth_rast_transform', None)), 'vel_mag')

            # ensure x_vel / y_vel available (sim updates after movement)
            xvel = getattr(sim, 'x_vel', None)
            yvel = getattr(sim, 'y_vel', None)
            if xvel is None:
                xvel = (sim.X - sim.prev_X) / dt
            if yvel is None:
                yvel = (sim.Y - sim.prev_Y) / dt

            # write per-agent rows
            for a in range(sim.num_agents):
                row = [t, a, float(sim.X[a]), float(sim.Y[a]), float(xvel[a]), float(yvel[a]),
                       float(depth_vals[a]) if np.isfinite(depth_vals[a]) else '',
                       float(velx_vals[a]) if np.isfinite(velx_vals[a]) else '',
                       float(vely_vals[a]) if np.isfinite(vely_vals[a]) else '',
                       float(mag_vals[a]) if np.isfinite(mag_vals[a]) else '']
                writer.writerow(row)

            # small flush to keep file consistent
            fh.flush()
            if (t + 1) % max(1, int(args.nsteps / 10)) == 0:
                print(f'Progress: {t+1}/{args.nsteps}')

            # Force per-step NPZ dumps when debug_behavior is enabled.
            # This makes per-step diagnostics deterministic and available
            # for offline analysis (cohesion/alignment/etc.). We re-use
            # the same payload shape used inside behavior.arbitrate()
            # to ensure consumers can parse the outputs.
            try:
                if getattr(sim, 'debug_behavior', False):
                    outdir = getattr(sim, 'model_dir', None) or os.path.join('outputs', 'diagnostics')
                    os.makedirs(outdir, exist_ok=True)
                    import time
                    fname_npz = os.path.join(outdir, f'behavior_debug_step_{int(t)}_{int(time.time())}.npz')
                    # collect best-effort diagnostics from sim
                    safe_payload = {}
                    try:
                        if hasattr(sim, 'last_head_vec'):
                            safe_payload['head_vec'] = np.asarray(sim.last_head_vec).astype(float)
                        if hasattr(sim, 'last_cue_magnitudes'):
                            for k, v in sim.last_cue_magnitudes.items():
                                safe_payload[f'{k}_mag'] = np.asarray(v).astype(float)
                        # neighbor lists
                        if hasattr(sim, 'agents_within_buffers'):
                            neighbor_counts = np.array([len(x) for x in sim.agents_within_buffers], dtype=np.int32)
                            neighbors_concat = np.concatenate(sim.agents_within_buffers).astype(np.int32) if neighbor_counts.sum() > 0 else np.array([], dtype=np.int32)
                            safe_payload['neighbor_counts'] = neighbor_counts
                            safe_payload['neighbors_concat'] = neighbors_concat
                        # include per-cue raw vectors if behavior stored them
                        if hasattr(sim, 'last_cue_vecs'):
                            try:
                                for ck, cv in sim.last_cue_vecs.items():
                                    safe_payload[f'{ck}_vec'] = np.asarray(cv).astype(float)
                            except Exception:
                                pass
                        # include alignment diagnostics if present
                        if hasattr(sim, '_alignment_diag'):
                            ad = sim._alignment_diag
                            for k in ('raw_headings_neighbors', 'headings_neighbors_used', 'used_velocity_heading', 'neighbor_indices', 'agent_indices'):
                                if k in ad:
                                    safe_payload[k] = np.asarray(ad[k])
                    except Exception:
                        pass
                    try:
                        np.savez_compressed(fname_npz, **safe_payload)
                        try:
                            print('Wrote per-step behavior NPZ:', fname_npz)
                        except Exception:
                            pass
                    except Exception as e:
                        try:
                            print('Failed writing per-step NPZ:', e)
                        except Exception:
                            pass
                    # Also write to HDF5 diagnostics writer when available (atomic per-step storage)
                    try:
                        dw = getattr(sim, 'diagnostics_writer', None)
                        if dw is not None:
                            # use step index t and include safe_payload arrays
                            dw.write_step(t, safe_payload)
                            try:
                                print('Wrote per-step diagnostics to HDF5 for step', t)
                            except Exception:
                                pass
                    except Exception:
                        pass
                    # Additionally write an authoritative per-step NPZ into a dedicated folder
                    try:
                        forced_dir = os.path.join(outdir, 'forced_rawvecs')
                        os.makedirs(forced_dir, exist_ok=True)
                        # Build authoritative payload from simulation attributes set by behavior
                        auth_payload = {}
                        # brief polling to allow behavior to populate last_head_vec/last_cue_vecs
                        try:
                            import time as _time
                            waited = 0.0
                            interval = 0.01
                            maxwait = 0.1
                            while waited < maxwait and not (hasattr(sim, 'last_head_vec') and getattr(sim, 'last_head_vec') is not None):
                                _time.sleep(interval)
                                waited += interval
                        except Exception:
                            pass
                        try:
                            if hasattr(sim, 'last_head_vec') and getattr(sim, 'last_head_vec') is not None:
                                auth_payload['head_vec'] = np.asarray(sim.last_head_vec).astype(float)
                            if hasattr(sim, 'last_cue_vecs') and isinstance(sim.last_cue_vecs, dict):
                                for ck, cv in sim.last_cue_vecs.items():
                                    try:
                                        auth_payload[f'{ck}_vec'] = np.asarray(cv).astype(float)
                                    except Exception:
                                        pass
                            if hasattr(sim, 'last_cue_magnitudes') and isinstance(sim.last_cue_magnitudes, dict):
                                for ck, cv in sim.last_cue_magnitudes.items():
                                    try:
                                        auth_payload[f'{ck}_mag'] = np.asarray(cv).astype(float)
                                    except Exception:
                                        pass
                            # neighbor diagnostics
                            if hasattr(sim, 'agents_within_buffers'):
                                try:
                                    neighbor_counts = np.array([len(x) for x in sim.agents_within_buffers], dtype=np.int32)
                                    auth_payload['neighbor_counts'] = neighbor_counts
                                    if neighbor_counts.sum() > 0:
                                        auth_payload['neighbors_concat'] = np.concatenate(sim.agents_within_buffers).astype(np.int32)
                                except Exception:
                                    pass
                        except Exception:
                            pass

                        # atomic write: write to temp file then replace
                        import tempfile
                        ts = int(time.time())
                        tmp_fd, tmp_path = tempfile.mkstemp(prefix=f'auth_step_{t}_', suffix='.npz', dir=forced_dir)
                        os.close(tmp_fd)
                        try:
                            # Attempt to augment auth_payload with the behavior-produced NPZ for this step
                            try:
                                import glob
                                beh_files = sorted(glob.glob(os.path.join(outdir, f'behavior_debug_step_{int(t)}_*')))
                                if beh_files:
                                    # pick the latest
                                    bf = beh_files[-1]
                                    try:
                                        bdata = np.load(bf)
                                        for k in bdata.files:
                                            if k not in auth_payload:
                                                try:
                                                    auth_payload[k] = np.asarray(bdata[k])
                                                except Exception:
                                                    pass
                                    except Exception:
                                        pass
                            except Exception:
                                pass
                            # use numpy to save to tmp_path
                            np.savez_compressed(tmp_path, **auth_payload)
                            final_path = os.path.join(forced_dir, f'auth_behavior_step_{t}_{ts}.npz')
                            # atomic replace
                            os.replace(tmp_path, final_path)
                            try:
                                # ensure file is flushed to disk (best-effort)
                                with open(final_path, 'rb') as f:
                                    try:
                                        os.fsync(f.fileno())
                                    except Exception:
                                        pass
                                print('Wrote authoritative NPZ:', final_path)
                            except Exception:
                                print('Wrote authoritative NPZ (no fsync):', final_path)
                        except Exception as e:
                            try:
                                if os.path.exists(tmp_path):
                                    os.remove(tmp_path)
                            except Exception:
                                pass
                            try:
                                print('Failed writing authoritative NPZ:', e)
                            except Exception:
                                pass
                    except Exception:
                        pass
                    # Additionally, write an authoritative HDF5 diagnostics step.
                    # Compute per-cue vectors here using sim._behavior to avoid race timing.
                    try:
                        dw = getattr(sim, 'diagnostics_writer', None)
                        if dw is not None and hasattr(sim, '_behavior'):
                            # decide weights (matching behavior.arbitrate logic)
                            tw = getattr(sim, 'test_weights', None)
                            known_keys = ['rheotaxis', 'alignment', 'cohesion', 'low_speed', 'wave_drag', 'refugia', 'border', 'shallow', 'avoid', 'collision']
                            if tw:
                                default_weights = {k: 0.0 for k in known_keys}
                                for k, v in tw.items():
                                    try:
                                        if k in default_weights:
                                            default_weights[k] = float(v)
                                    except Exception:
                                        pass
                            else:
                                default_weights = {
                                    'rheotaxis': 25000,
                                    'alignment': 20500,
                                    'cohesion': 11000,
                                    'low_speed': 1500,
                                    'wave_drag': 0,
                                    'refugia': 50000,
                                    'border': 50000,
                                    'shallow': 100000,
                                    'avoid': 25000,
                                    'collision': 50000,
                                }
                            # compute cues via behavior helper
                            b = sim._behavior
                            try:
                                rheo = b.rheo_cue(default_weights['rheotaxis'])
                            except Exception:
                                rheo = np.zeros((sim.num_agents, 2))
                            try:
                                alignment = b.alignment_cue(default_weights['alignment'])
                            except Exception:
                                alignment = np.zeros((sim.num_agents, 2))
                            try:
                                cohesion = b.cohesion_cue(default_weights['cohesion'])
                            except Exception:
                                cohesion = np.zeros((sim.num_agents, 2))
                            try:
                                low_speed = b.vel_cue(default_weights['low_speed'])
                            except Exception:
                                low_speed = np.zeros((sim.num_agents, 2))
                            try:
                                wave_drag = b.wave_drag_cue(default_weights['wave_drag'])
                            except Exception:
                                wave_drag = np.zeros((sim.num_agents, 2))
                            try:
                                refugia = b.find_nearest_refuge(default_weights['refugia'])
                            except Exception:
                                refugia = np.zeros((sim.num_agents, 2))
                            try:
                                border = b.border_cue(default_weights['border'], t)
                            except Exception:
                                border = np.zeros((sim.num_agents, 2))
                            try:
                                shallow = b.shallow_cue(default_weights['shallow'])
                            except Exception:
                                shallow = np.zeros((sim.num_agents, 2))
                            try:
                                avoid = b.already_been_here(default_weights['avoid'], t)
                            except Exception:
                                avoid = np.zeros((sim.num_agents, 2))
                            try:
                                collision = b.collision_cue(default_weights['collision'])
                            except Exception:
                                collision = np.zeros((sim.num_agents, 2))

                            cue_map = {
                                'rheo': rheo,
                                'alignment': alignment,
                                'cohesion': cohesion,
                                'low_speed': low_speed,
                                'wave_drag': wave_drag,
                                'refugia': refugia,
                                'border': border,
                                'shallow': shallow,
                                'avoid': avoid,
                                'collision': collision,
                            }
                            auth_payload = {}
                            # include per-cue vecs and magnitudes
                            for ck, cv in cue_map.items():
                                try:
                                    arr = np.asarray(cv).astype(float)
                                    auth_payload[f'{ck}_vec'] = arr
                                    try:
                                        auth_payload[f'{ck}_mag'] = np.linalg.norm(arr, axis=1)
                                    except Exception:
                                        pass
                                except Exception:
                                    pass
                            # resultant head vector (sum of migratory cues excluding refugia?)
                            try:
                                # sum all cues (simple sum for diagnostics)
                                total = np.zeros((sim.num_agents, 2), dtype=float)
                                for v in cue_map.values():
                                    try:
                                        total += np.asarray(v, dtype=float)
                                    except Exception:
                                        pass
                                auth_payload['head_vec'] = total
                            except Exception:
                                pass
                            try:
                                dw.write_step(t, auth_payload)
                                print('Wrote authoritative diagnostics to HDF5 for step', t)
                            except Exception:
                                pass
                    except Exception:
                        pass
            except Exception:
                pass

    print('Headless run complete. Trace saved to', csv_path)
    # fallback behavior debug dump: write last_head_vec and last_cue_magnitudes if enabled
    try:
        if getattr(sim, 'debug_behavior', False):
            out_debug = os.path.join(outdir, f'{args.model_name}_behavior_last.json')
            d = {}
            if hasattr(sim, 'last_head_vec'):
                d['last_head_vec'] = np.asarray(sim.last_head_vec).astype(float).tolist()
            if hasattr(sim, 'last_cue_magnitudes'):
                d['last_cue_magnitudes'] = {k: np.asarray(v).astype(float).tolist() for k, v in sim.last_cue_magnitudes.items()}
            if d:
                import json
                with open(out_debug, 'w', encoding='utf-8') as fh:
                    json.dump(d, fh)
                print('Wrote behavior fallback debug to', out_debug)
    except Exception:
        pass

    # Persist agent headings into the HDF5 so off-line analysis can access them
    try:
        h5 = hdf5_io.get_hdf5_obj(sim)
        if h5 is not None:
            # ensure agent_data/heading dataset exists (shape: num_agents x num_timesteps)
            try:
                if 'agent_data/heading' not in h5:
                    import numpy as _np
                    hdf5_io.write_dataset(h5, 'agent_data/heading', _np.zeros((sim.num_agents, int(sim.num_timesteps)), dtype=_np.float32))
                # write current heading into column 0
                try:
                    arr = h5['agent_data/heading']
                    arr[:, 0] = np.array(sim.heading)
                    hdf5_io.write_dataset(h5, 'agent_data/heading', arr[:])
                except Exception:
                    try:
                        hdf5_io.write_dataset(h5, 'agent_data/heading', np.array(sim.heading)[:, None])
                    except Exception:
                        pass
                # also write top-level heading for quick access
                try:
                    hdf5_io.write_dataset(h5, 'heading', np.array(sim.heading))
                except Exception:
                    pass
            except Exception:
                pass
    except Exception:
        pass

    sim.close()

    # If preseed export requested, write out sim.hdf5['memory/*'] to a separate HDF5
    if getattr(args, 'preseed_memory_out', None):
        try:
            out_path = os.path.abspath(args.preseed_memory_out)
            print('Exporting memory to', out_path)
            with h5py.File(out_path, 'w') as oh5:
                # create memory group
                mg = oh5.create_group('memory')
                # copy datasets from sim.hdf5 if present
                try:
                    if 'memory' in sim.hdf5:
                        for k in sim.hdf5['memory'].keys():
                            try:
                                data = np.array(sim.hdf5[f'memory/{k}'])
                                mg.create_dataset(str(k), data=data, dtype='f4')
                            except Exception:
                                pass
                except Exception:
                    pass
            print('Memory export complete')
        except Exception as e:
            print('Failed to export memory:', e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nagents', type=int, default=200)
    parser.add_argument('--nsteps', type=int, default=200)
    parser.add_argument('--seed', type=int, default=None, help='Optional RNG seed for deterministic runs')
    parser.add_argument('--debug-movement', action='store_true', help='Enable movement debug dumps')
    parser.add_argument('--debug-behavior', action='store_true', help='Enable behavior debug dumps')
    parser.add_argument('--model-name', dest='model_name', default='nuyakuk_headless')
    parser.add_argument('--out', default=os.path.join('outputs', 'diagnostics'))
    parser.add_argument('--start-polygon', dest='start_polygon', default=None, help='Optional path to start location shapefile')
    parser.add_argument('--test-weights-file', dest='test_weights_file', default=None, help='Optional JSON file with per-cue test weight overrides')
    parser.add_argument('--preload-memory', dest='preload_memory', default=None, help='Optional HDF5 file to preload per-agent memory (memory/* datasets)')
    parser.add_argument('--preseed-memory-out', dest='preseed_memory_out', default=None, help='Optional path to export sim.hdf5 memory/* datasets after run')
    args = parser.parse_args()
    run_headless(args)


if __name__ == '__main__':
    main()
