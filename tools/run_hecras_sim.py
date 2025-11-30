"""Run a short HECRAS-driven simulation and save a PNG of agent positions.

This script adapts the raster-based example to the repository's simulation API.
It will try to use HECRAS mapping if a HECRAS HDF/plan path is provided; otherwise
it will fall back to a raster file path (example). The script runs a short
simulation (default 10 timesteps) and writes `tools/hecras_sim.png`.

Usage examples:
  python tools/run_hecras_sim.py
  python tools/run_hecras_sim.py --timesteps 20 --agents 500

The script performs a Numba warmup if `src.emergent` exposes `_numba_warmup_for_sim`.
"""
from __future__ import annotations

import argparse
import os
import sys
import math
import numpy as np

# Ensure repository root is on path so imports work when run from tools/
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    # Import simulation API from the package
    from src.emergent.salmon_abm.sockeye_SoA import simulation
    from src.emergent.salmon_abm.sockeye_dynamic_environment import HECRAS
except Exception as exc:  # pragma: no cover - best-effort import
    print('Could not import simulation modules:', exc)
    raise


def try_warmup(n_agents: int = 2000):
    """Call warmup helper if present to reduce JIT overhead.

    This creates a small dummy simulation-like object and calls the
    repository `_numba_warmup_for_sim(sim)` helper with it. Using a
    dummy shaped like production arrays primes Numba specializations.
    """
    class _DummySim:
        pass

    d = _DummySim()
    d.num_agents = int(max(128, int(n_agents)))
    import numpy as _np
    # typical swim buffer width used in warmup scripts
    d.swim_speeds = _np.zeros((d.num_agents, 8), dtype=_np.float64)

    # Try package-level re-export first
    try:
        from src.emergent import _numba_warmup_for_sim
        try:
            print('Calling repository numba warmup...')
            _numba_warmup_for_sim(d)
            return
        except Exception as e:
            print('Repository warmup helper raised:', e)
    except Exception:
        pass

    # Try module-level helper as a fallback
    try:
        from src.emergent.salmon_abm import _numba_warmup_for_sim as _warm
        try:
            print('Calling salmon_abm numba warmup...')
            _warm._numba_warmup_for_sim(d)
            return
        except Exception as e:
            print('salmon_abm warmup helper raised:', e)
    except Exception:
        pass

    print('No numba warmup helper found; proceeding without warmup')


def run_sim(out_png: str, agents: int = 200, timesteps: int = 10, hecras_plan: str | None = None, resolution: float = 1.0, depth_only: bool = False):
    # Use HECRAS mapping if plan path is provided and exists; otherwise fall back
    env_files = None
    # auto-discover HECRAS .p05 HDF in the project data folder if not explicitly provided
    hecras_folder = os.path.join(REPO_ROOT, 'data', 'salmon_abm', '20240506')
    if hecras_plan is None:
        # prefer file that contains '.p05' in the name
        if os.path.isdir(hecras_folder):
            for f in os.listdir(hecras_folder):
                if '.p05' in f.lower() and f.lower().endswith('.hdf'):
                    hecras_plan = os.path.join(hecras_folder, f)
                    print('Auto-discovered HECRAS plan:', hecras_plan)
                    break

    if hecras_plan and os.path.exists(hecras_plan):
        print('Preparing HECRAS rasters from plan:', hecras_plan)
        # HECRAS(...) expects model_dir (output folder) and HECRAS_dir (path without '.hdf')
        hecras_dir_noext = os.path.splitext(hecras_plan)[0]
        hecras_out_dir = os.path.join(REPO_ROOT, 'outputs', 'hecras_run')
        os.makedirs(hecras_out_dir, exist_ok=True)
        try:
            if depth_only:
                # Fast depth-only extraction: read minimal HDF datasets and rasterize depth and elevation.
                import h5py
                import rasterio
                from rasterio.transform import Affine

                print('Running depth-only extraction from HECRAS HDF (fast path)')
                hdf = h5py.File(hecras_dir_noext + '.hdf', 'r')
                pts = np.array(hdf.get('Geometry/2D Flow Areas/2D area/Cells Center Coordinate'))
                wsel = hdf['Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/2D area/Water Surface'][-1]
                elev = np.array(hdf.get('Geometry/2D Flow Areas/2D area/Cells Minimum Elevation'))
                hdf.close()

                # build grid extents
                xmin = np.min(pts[:,0]); xmax = np.max(pts[:,0])
                ymin = np.min(pts[:,1]); ymax = np.max(pts[:,1])
                xint = np.arange(xmin, xmax, resolution)
                yint = np.arange(ymax, ymin, resolution * -1.)
                xnew, ynew = np.meshgrid(xint, yint)

                # Simple nearest-neighbor rasterization via KDTree lookup (fast)
                from scipy.spatial import cKDTree
                tree = cKDTree(pts)
                qx = np.column_stack((xnew.ravel(), ynew.ravel()))
                dists, inds = tree.query(qx, k=1)
                elev_rast = np.asarray(elev)[inds].reshape((len(yint), len(xint)))
                wsel_rast = np.asarray(wsel)[inds].reshape((len(yint), len(xint)))
                depth_rast = wsel_rast - elev_rast

                transform = Affine.translation(xnew[0][0] - 0.5 * resolution, ynew[0][0] - 0.5 * resolution) * Affine.scale(resolution, -1 * resolution)
                # write rasters
                def _write(name, arr):
                    outp = os.path.join(hecras_out_dir, name)
                    with rasterio.open(outp, mode='w', driver='GTiff', width=arr.shape[1], height=arr.shape[0], count=1, dtype='float64', crs='EPSG:4326', transform=transform) as dst:
                        dst.write(arr, 1)
                _write('elev.tif', elev_rast)
                _write('depth.tif', depth_rast)
                env_files = {'elev': os.path.join(hecras_out_dir, 'elev.tif'), 'depth': os.path.join(hecras_out_dir, 'depth.tif')}
                print('Depth-only extraction complete; rasters written to', hecras_out_dir)
            else:
                env_files = HECRAS(model_dir=hecras_out_dir, HECRAS_dir=hecras_dir_noext, resolution=resolution, crs='EPSG:4326')
                print('HECRAS mapping returned:', env_files)
        except Exception as exc:
            # don't fall back silently — fail so user can see mapping error
            print('HECRAS mapping failed; aborting. Error:')
            raise
    else:
        raise FileNotFoundError('No HECRAS plan found; place a .hdf in data/salmon_abm/20240506 or pass --hecras-plan')

    # Ensure warmup is executed in this process to avoid JIT overhead during timed run
    try:
        # prefer the centralized warmup API
        try:
            from src.emergent.warmup import run_global_warmup
            print('Calling centralized warmup...')
            run_global_warmup(2000)
        except Exception:
            # fall back to local helper
            try_warmup()
    except Exception:
        # fallback: run the tools/numba_warmup.py script if present
        warmup_script = os.path.join(REPO_ROOT, 'tools', 'numba_warmup.py')
        if os.path.exists(warmup_script):
            try:
                print('Running fallback warmup script:', warmup_script)
                import runpy
                runpy.run_path(warmup_script, run_name='__main__')
            except Exception as exc:
                print('Fallback warmup script failed:', exc)

    # discover start polygon (prefer river right)
    start_folder = os.path.join(REPO_ROOT, 'data', 'salmon_abm', 'starting_location')
    start_polygon = None
    if os.path.isdir(start_folder):
        cand = os.path.join(start_folder, 'start_loc_river_right.shp')
        if os.path.exists(cand):
            start_polygon = cand
            print('Using start polygon (river right):', cand)

    # Build the simulation object — prefer the `build_sim` helper from tools if available
    sim = None
    try:
        from tools.profile_timestep_cprofile import build_sim
        print('Using tools.profile_timestep_cprofile.build_sim to construct simulation')
        sim = build_sim(num_agents=agents, use_hecras=bool(env_files), hecras_write_rasters=False)
    except Exception:
        # Fall back to direct constructor; adapt to the required signature
        sim_kwargs = dict(
            model_dir=str(REPO_ROOT),
            model_name='hecras_run',
            crs='EPSG:4326',
            basin='unknown',
            water_temp=10.0,
            start_polygon=None,
            env_files=env_files,
            longitudinal_profile=None,
            fish_length=500,
            num_timesteps=timesteps,
            num_agents=agents,
            use_gpu=False,
            hecras_plan_path=hecras_plan,
            hecras_fields=None,
            hecras_k=8,
            use_hecras=bool(env_files),
            hecras_write_rasters=False,
            defer_hdf=False,
            defer_log_dir=None,
            defer_log_fmt='npz'
        )
        print('Constructing simulation with:', sim_kwargs)
        # pass discovered start polygon if available
        if start_polygon is not None:
            sim_kwargs['start_polygon'] = start_polygon
        sim = simulation(**sim_kwargs)

    print('Running simulation for %d timesteps' % timesteps)

    # If we have a HECRAS HDF, enable node-based mapping on the simulation
    try:
        if hecras_plan and os.path.exists(hecras_plan):
            import h5py
            print('Loading HECRAS nodes for node-based mapping (fast)')
            hdf = h5py.File(os.path.splitext(hecras_plan)[0] + '.hdf', 'r')
            try:
                pts = np.array(hdf.get('Geometry/2D Flow Areas/2D area/Cells Center Coordinate'))
            except Exception:
                pts = None
            node_fields = {}
            # depth and vel components if present
            try:
                node_fields['depth'] = hdf['Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/2D area/Water Surface'][-1] - np.array(hdf.get('Geometry/2D Flow Areas/2D area/Cells Minimum Elevation'))
            except Exception:
                pass
            try:
                node_fields['vel_x'] = hdf['Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/2D area/Cell Velocity - Velocity X'][-1]
                node_fields['vel_y'] = hdf['Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/2D area/Cell Velocity - Velocity Y'][-1]
            except Exception:
                pass
            hdf.close()

            if pts is not None and node_fields:
                try:
                    # Ensure simulation has required sampled attributes expected by enable_hecras
                    n = getattr(sim, 'num_agents', getattr(sim, 'pop_size', None) or agents)
                    if not hasattr(sim, 'depth'):
                        sim.depth = np.zeros(n, dtype=float)
                    if not hasattr(sim, 'x_vel'):
                        sim.x_vel = np.zeros(n, dtype=float)
                    if not hasattr(sim, 'y_vel'):
                        sim.y_vel = np.zeros(n, dtype=float)
                    if not hasattr(sim, 'vel_mag'):
                        sim.vel_mag = np.zeros(n, dtype=float)
                    if not hasattr(sim, 'wet'):
                        sim.wet = np.ones(n, dtype=float)
                    if not hasattr(sim, 'distance_to'):
                        sim.distance_to = np.zeros(n, dtype=float)

                    print('Enabling HECRAS node mapping on simulation (nearest neighbor k=1)')
                    sim.enable_hecras(pts, node_fields, k=1)
                except Exception as exc:
                    print('Failed to enable node-based HECRAS mapping:', exc)
    except Exception:
        pass
    # Use the timestep API used across profiling tools: sim.timestep(t, dt, g, pid)
    positions = []
    try:
        from src.emergent.salmon_abm.sockeye_SoA import PID_controller
        pid = PID_controller(getattr(sim, 'num_agents', agents), k_p=1.0, k_i=0.0, k_d=0.0)
        try:
            pid.interp_PID()
        except Exception:
            pass
    except Exception:
        pid = None

    for t in range(timesteps):
        try:
            sim.timestep(t, 1.0, 9.81, pid)
        except Exception:
            # if timestep fails, try a generic run() if available
            try:
                sim.run(timesteps)
                break
            except Exception:
                raise
        # try to grab agent XY arrays; many sims expose `X` and `Y`
        if hasattr(sim, 'X') and hasattr(sim, 'Y'):
            positions.append(np.column_stack((np.asarray(sim.X), np.asarray(sim.Y))))
        else:
            # attempt other getters
            try:
                pos = sim.get_agent_positions()
                positions.append(np.asarray(pos))
            except Exception:
                # give up on recording positions for this timestep
                pass

    # Plot the final positions
    if positions:
        final = positions[-1]
        if final.size == 0:
            print('No agent positions recorded; saving empty figure')
            plt.figure(figsize=(6, 6))
            plt.text(0.5, 0.5, 'No agents', ha='center')
            plt.savefig(out_png)
            return

        xs = final[:, 0]
        ys = final[:, 1]
        plt.figure(figsize=(6, 6))
        plt.scatter(xs, ys, s=8, c='tab:blue', alpha=0.75)
        plt.title(f'HECRAS sim: agents={agents} timesteps={timesteps}')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.tight_layout()
        plt.savefig(out_png)
        print('Saved', out_png)
    else:
        print('No positions captured; nothing to plot')


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--timesteps', '-t', type=int, default=10)
    parser.add_argument('--agents', '-a', type=int, default=200)
    parser.add_argument('--hecras-plan', '-p', type=str, default=None)
    parser.add_argument('--resolution', '-r', type=float, default=1.0,
                        help='HECRAS interpolation resolution in native units (coarser -> faster)')
    parser.add_argument('--depth-only', action='store_true', help='Only extract rasterized depth/elevation (skip velocity interpolation)')
    parser.add_argument('--out', '-o', type=str, default=os.path.join('tools', 'hecras_sim.png'))
    args = parser.parse_args(argv)

    out = args.out
    os.makedirs(os.path.dirname(out), exist_ok=True)
    # If user didn't explicitly request velocities and resolution is coarse, prefer depth-only fast path
    depth_only_flag = args.depth_only or (args.resolution > 1.0 and not args.depth_only)
    run_sim(out, agents=args.agents, timesteps=args.timesteps, hecras_plan=args.hecras_plan, resolution=args.resolution, depth_only=depth_only_flag)


if __name__ == '__main__':
    main()
