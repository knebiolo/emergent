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
# If the user requested `--interactive` on the command line, attempt to select
# a GUI-capable backend before importing `pyplot`. This helps when the default
# backend is non-interactive (e.g. 'Agg') and the user expects a window.
_interactive_requested = any(a in sys.argv for a in ('--interactive', '-i'))
if _interactive_requested:
    for _bk in ('Qt5Agg', 'TkAgg', 'WXAgg', 'Qt4Agg'):
        try:
            matplotlib.use(_bk, force=True)
            break
        except Exception:
            continue
import matplotlib.pyplot as plt
# Print backend for debugging why interactive window may not appear
print('Matplotlib backend in use:', matplotlib.get_backend())

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


def run_sim(out_png: str, agents: int = 200, timesteps: int = 10, hecras_plan: str | None = None, resolution: float = 1.0, fast_raster: bool = False, interactive: bool = False, node_only: bool = False, movie: bool = False, movie_out: str | None = None, fps: int = 10):
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
            if node_only and not depth_only:
                # User explicitly requested node-only: skip full HECRAS interpolation/rasterization.
                print('Node-only requested: skipping HECRAS raster interpolation (will read nodal arrays directly).')
                env_files = None
            elif fast_raster:
                # Fast KDTree rasterization: read HDF datasets and rasterize all fields using nearest-neighbor.
                import h5py
                import rasterio
                from rasterio.transform import Affine

                print('Running fast KDTree rasterization from HECRAS HDF')
                hdf = h5py.File(hecras_dir_noext + '.hdf', 'r')
                pts = np.array(hdf.get('Geometry/2D Flow Areas/2D area/Cells Center Coordinate'))
                wsel = hdf['Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/2D area/Water Surface'][-1]
                elev = np.array(hdf.get('Geometry/2D Flow Areas/2D area/Cells Minimum Elevation'))
                # Also extract velocities for behavioral cues
                try:
                    vel_x = hdf['Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/2D area/Cell Velocity - Velocity X'][-1]
                    vel_y = hdf['Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/2D area/Cell Velocity - Velocity Y'][-1]
                except Exception:
                    vel_x = None
                    vel_y = None
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
                
                # Create wetted raster: 1 where depth > 0, 0 elsewhere
                wetted_rast = (depth_rast > 0.0).astype(np.float64)
                
                # Rasterize velocities if available
                if vel_x is not None and vel_y is not None:
                    vel_x_rast = np.asarray(vel_x)[inds].reshape((len(yint), len(xint)))
                    vel_y_rast = np.asarray(vel_y)[inds].reshape((len(yint), len(xint)))
                    # Calculate velocity direction (radians) and magnitude
                    vel_dir_rast = np.arctan2(vel_y_rast, vel_x_rast)
                    vel_mag_rast = np.sqrt(vel_x_rast**2 + vel_y_rast**2)
                else:
                    vel_x_rast = np.zeros_like(depth_rast)
                    vel_y_rast = np.zeros_like(depth_rast)
                    vel_dir_rast = np.zeros_like(depth_rast)
                    vel_mag_rast = np.zeros_like(depth_rast)

                transform = Affine.translation(xnew[0][0] - 0.5 * resolution, ynew[0][0] - 0.5 * resolution) * Affine.scale(resolution, -1 * resolution)
                # write rasters
                def _write(name, arr):
                    outp = os.path.join(hecras_out_dir, name)
                    with rasterio.open(outp, mode='w', driver='GTiff', width=arr.shape[1], height=arr.shape[0], count=1, dtype='float64', crs='EPSG:26904', transform=transform) as dst:
                        dst.write(arr, 1)
                _write('elev.tif', elev_rast)
                _write('depth.tif', depth_rast)
                _write('wetted.tif', wetted_rast)
                _write('x_vel.tif', vel_x_rast)
                _write('y_vel.tif', vel_y_rast)
                _write('vel_dir.tif', vel_dir_rast)
                _write('vel_mag.tif', vel_mag_rast)
                env_files = {
                    'elev': os.path.join(hecras_out_dir, 'elev.tif'),
                    'depth': os.path.join(hecras_out_dir, 'depth.tif'),
                    'wetted': os.path.join(hecras_out_dir, 'wetted.tif'),
                    'x_vel': os.path.join(hecras_out_dir, 'x_vel.tif'),
                    'y_vel': os.path.join(hecras_out_dir, 'y_vel.tif'),
                    'vel_dir': os.path.join(hecras_out_dir, 'vel_dir.tif'),
                    'vel_mag': os.path.join(hecras_out_dir, 'vel_mag.tif')
                }
                print('Fast KDTree rasterization complete; rasters written to', hecras_out_dir)
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
    # Prefer directly constructing the authoritative `simulation` class so agents
    # are initialized from `start_polygon` (world coordinates). Avoid `build_sim`.
    try:
        sim_kwargs = dict(
            model_dir=str(REPO_ROOT),
            model_name='hecras_run',
            crs='EPSG:26904',
            basin='Nushagak River',
            water_temp=10.0,
            start_polygon=None,
            env_files=env_files,
            longitudinal_profile=None,
            fish_length=500,
            num_timesteps=timesteps,
            num_agents=agents,
            use_gpu=False,
            hecras_plan_path=None,
            hecras_fields=None,
            hecras_k=8,
            use_hecras=False,
            hecras_write_rasters=False,
            defer_hdf=False,
            defer_log_dir=None,
            defer_log_fmt='npz'
        )
        # pass discovered start polygon if available
        if start_polygon is not None:
            sim_kwargs['start_polygon'] = start_polygon
        print('Constructing simulation with:', sim_kwargs)
        sim = simulation(**sim_kwargs)
    except Exception as exc:
        print('Failed to construct simulation with constructor:', exc)
        raise

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
                    
                    # Perform initial sampling of HECRAS values at agent positions
                    # The mapping is already built by enable_hecras, now apply it to each field
                    if 'depth' in node_fields:
                        sim.depth = sim.apply_hecras_mapping(node_fields['depth'])
                    if 'vel_x' in node_fields:
                        sim.x_vel = sim.apply_hecras_mapping(node_fields['vel_x'])
                    if 'vel_y' in node_fields:
                        sim.y_vel = sim.apply_hecras_mapping(node_fields['vel_y'])
                        sim.vel_mag = np.sqrt(sim.x_vel**2 + sim.y_vel**2)
                    
                    # Re-initialize heading now that we have velocities from HECRAS
                    # The __init__ tried to set heading but velocities weren't available yet
                    # arctan2 gives flow direction; subtract π to point upstream (opposite direction)
                    flow_direction = np.arctan2(sim.y_vel, sim.x_vel)  # radians
                    sim.heading = flow_direction - np.pi  # Point upstream
                    sim.max_practical_sog = np.array([sim.sog * np.cos(sim.heading), 
                                                       sim.sog * np.sin(sim.heading)])
                    
                    print('Node-based HECRAS mapping enabled; simulation will sample HECRAS per-agent directly.')
                    print(f'Initial agent positions: X range [{sim.X.min():.1f}, {sim.X.max():.1f}], Y range [{sim.Y.min():.1f}, {sim.Y.max():.1f}]')
                    print(f'Initial velocities: x_vel range [{sim.x_vel.min():.3f}, {sim.x_vel.max():.3f}], y_vel range [{sim.y_vel.min():.3f}, {sim.y_vel.max():.3f}]')
                    print(f'Initial depth range: [{sim.depth.min():.3f}, {sim.depth.max():.3f}]')
                    print(f'Initial heading range: [{np.degrees(sim.heading).min():.1f}°, {np.degrees(sim.heading).max():.1f}°]')
                except Exception as exc:
                    print('Failed to enable node-based HECRAS mapping; raster fallback will be used. Error:', exc)
                    sim.use_hecras = False
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

    if interactive and matplotlib.get_backend().lower() != 'agg':
        # Attempt interactive plotting: show depth raster as background (if present) and update agent positions live.
        try:
            plt.ion()
            fig, ax = plt.subplots(figsize=(8, 6))
            # Try to detect a GUI toolkit to provide a better diagnostic if no window appears
            gui_ok = False
            try:
                import PyQt5  # type: ignore
                gui_ok = True
            except Exception:
                try:
                    import tkinter as _tk  # type: ignore
                    gui_ok = True
                except Exception:
                    gui_ok = False
            if not gui_ok:
                print('Interactive requested but no GUI toolkit found (PyQt5/Tk).')
                print('Install PyQt5 (pip install PyQt5) or run without --interactive to save PNG only.')
                plt.close(fig)
                raise RuntimeError('No GUI toolkit available')
            # ensure the figure manager asks the backend to create a window
            try:
                fig.show()
            except Exception:
                pass
            depth_path = os.path.join(REPO_ROOT, 'outputs', 'hecras_run', 'depth.tif')
            background = None
            try:
                if os.path.exists(depth_path):
                    import rasterio
                    with rasterio.open(depth_path) as src:
                        bg = src.read(1)
                        tr = src.transform
                        h, w = bg.shape
                        left = tr.c
                        top = tr.f
                        right = left + tr.a * w
                        bottom = top + tr.e * h
                        extent = (left, right, bottom, top)
                        background = (bg, extent)
                        ax.imshow(bg, extent=extent, origin='lower', cmap='viridis')
            except Exception:
                background = None

            # initial scatter (empty) — ensure high zorder so points draw above raster
            scat = ax.scatter([], [], s=8, c='orange', edgecolors='k', alpha=0.95, zorder=10)
            ax.set_title(f'HECRAS sim: agents={agents}')
            # If we have a background raster, fix the axis extent so autoscale doesn't hide points
            if background is not None:
                _, extent = background
                ax.set_xlim(extent[0], extent[1])
                ax.set_ylim(extent[2], extent[3])

            for t in range(timesteps):
                try:
                    sim.timestep(t, 1.0, 9.81, pid)
                except Exception:
                    try:
                        sim.run(timesteps)
                        break
                    except Exception:
                        raise

                if hasattr(sim, 'X') and hasattr(sim, 'Y'):
                    arr = np.column_stack((np.asarray(sim.X), np.asarray(sim.Y)))
                else:
                    try:
                        arr = np.asarray(sim.get_agent_positions())
                    except Exception:
                        arr = None

                if arr is not None and arr.size:
                    # Filter out NaN positions
                    mask = np.isfinite(arr).all(axis=1)
                    arr_f = arr[mask]
                    if arr_f.size:
                        # Agent coordinates are already in world/UTM coordinates, use them directly
                        scat.set_offsets(arr_f)
                        # diagnostic info
                        xmin, ymin = np.min(arr_f[:, 0]), np.min(arr_f[:, 1])
                        xmax, ymax = np.max(arr_f[:, 0]), np.max(arr_f[:, 1])
                        sample = arr_f[:5].tolist()
                        print(f't={t}: plotting {len(arr_f)} agents; X range [{xmin:.1f},{xmax:.1f}] Y range [{ymin:.1f},{ymax:.1f}] sample: {sample}')
                    else:
                        scat.set_offsets(np.empty((0, 2)))
                        print(f't={t}: no valid agent positions')
                fig.canvas.draw_idle()
                plt.pause(0.01)

            # leave interactive mode; keep the window open until user closes it
            try:
                plt.ioff()
                fig.tight_layout()
                print('Entering blocking show() - close the window to continue and save final PNG.')
                plt.show(block=True)
            except Exception:
                # If blocking show fails, save the final frame instead
                fig.tight_layout()
                fig.savefig(out_png)
                print('Saved', out_png)
            return
        except Exception as exc:
            print('Interactive plotting not available; falling back to PNG. Reason:', exc)

    # Non-interactive or fallback: run simulation and record positions for post-run plotting
    for t in range(timesteps):
        try:
            sim.timestep(t, 1.0, 9.81, pid)
        except Exception:
            # if timestep fails, try a generic run() if available
            try:
                sim.run(model_name='hecras_run', n=timesteps, dt=1.0, video=False, interactive=False)
                break
            except Exception:
                raise
        # try to grab agent XY arrays; many sims expose `X` and `Y`
        arr = None
        if hasattr(sim, 'X') and hasattr(sim, 'Y'):
            arr = np.column_stack((np.asarray(sim.X), np.asarray(sim.Y)))
        else:
            # attempt other getters
            try:
                pos = sim.get_agent_positions()
                arr = np.asarray(pos)
            except Exception:
                arr = None

        if arr is None:
            print(f't={t}: no agent position accessor returned positions')
            positions.append(np.empty((0, 2)))
            continue

        # filter and diagnostics
        if arr.size:
            mask = np.isfinite(arr).all(axis=1)
            arr_f = arr[mask]
            if arr_f.size:
                xmin, ymin = np.min(arr_f[:, 0]), np.min(arr_f[:, 1])
                xmax, ymax = np.max(arr_f[:, 0]), np.max(arr_f[:, 1])
                sample = arr_f[:10].tolist()
                print(f'nonint t={t}: captured {len(arr_f)} agents; X range [{xmin:.1f},{xmax:.1f}] Y range [{ymin:.1f},{ymax:.1f}] sample: {sample}')
            else:
                print(f'nonint t={t}: no finite agent positions')
        else:
            print(f'nonint t={t}: positions array empty')
        positions.append(arr)

    # Plot the final positions
    if positions:
        final = positions[-1]
        if final.size == 0:
            print('No agent positions recorded; saving empty figure')
            plt.figure(figsize=(6, 6))
            plt.text(0.5, 0.5, 'No agents', ha='center')
            plt.savefig(out_png)
            return

        # Save final agent positions for inspection
        try:
            csv_out = os.path.join('tools', 'agents_last.csv')
            np.savetxt(csv_out, final, delimiter=',', header='x,y', comments='')
            print('Saved final agent positions to', csv_out)
        except Exception:
            pass

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
    parser.add_argument('--fast-raster', action='store_true', help='Fast KDTree rasterization of HECRAS data (skip full interpolation)')
    parser.add_argument('--interactive', action='store_true', help='Show a live interactive plot of the simulation (fall back to PNG if no GUI)')
    parser.add_argument('--node-only', action='store_true', help='Skip HECRAS raster interpolation and use node-based mapping directly (fast)')
    parser.add_argument('--movie', action='store_true', help='Run headless and produce an MP4 (or PNG frames) of the simulation')
    parser.add_argument('--movie-out', type=str, default=os.path.join('tools', 'hecras_sim.mp4'), help='Output path for movie')
    parser.add_argument('--fps', type=int, default=10, help='Frames per second for movie')
    parser.add_argument('--out', '-o', type=str, default=os.path.join('tools', 'hecras_sim.png'))
    args = parser.parse_args(argv)

    out = args.out
    os.makedirs(os.path.dirname(out), exist_ok=True)
    # If user didn't explicitly request velocities and resolution is coarse, prefer fast raster path
    fast_raster_flag = args.fast_raster or (args.resolution > 1.0 and not args.fast_raster)
    run_sim(out, agents=args.agents, timesteps=args.timesteps, hecras_plan=args.hecras_plan, resolution=args.resolution, fast_raster=fast_raster_flag, interactive=args.interactive, node_only=args.node_only, movie=args.movie, movie_out=args.movie_out, fps=args.fps)


if __name__ == '__main__':
    main()
