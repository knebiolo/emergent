import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from emergent.ship_abm.simulation_core import simulation

def random_recent_datetime():
    now = datetime(2025, 11, 23, 0, 0, 0)
    days_back = np.random.randint(1, 30)
    hours = np.random.randint(0, 24)
    return now - timedelta(days=days_back, hours=hours)

def run_single_scenario(config, env_datetime, run_id, output_dir,
                        power_loss_lon=None, power_loss_lat=None, power_loss_radius_m=None,
                        skip_env: bool = False):
    cfg = config['simulation']
    dt, T = cfg['dt'], cfg['T']
    port = cfg['port_name']
    waypoints_lonlat = config['route']['waypoints_lonlat']
    speed_knots = config['vessel']['initial_conditions']['speed_knots']
    speed_ms = speed_knots * 0.514444
    power_loss_time = config['events']['power_loss_schedule'][0]['time_s']
    print(f"  Initializing simulation (port={port}, dt={dt}, T={T})")
    print(f"  Start position: Lon={waypoints_lonlat[0][0]:.6f}, Lat={waypoints_lonlat[0][1]:.6f}")
    print(f"  Initial speed: {speed_knots} knots ({speed_ms:.2f} m/s)")
    print(f"  Environmental datetime: {env_datetime}")
    sim = simulation(port_name=port, dt=dt, T=T, n_agents=1, load_enc=True, verbose=False)
    cbofs_loaded = False
    if not skip_env:
        try:
            print(f"  Loading CBOFS for {env_datetime}...")
            sim.load_environmental_forcing(start=env_datetime)
            cbofs_loaded = True
            print(f"  CBOFS loaded successfully")
        except Exception as e:
            print(f"  Warning: Could not load CBOFS: {e}")
    else:
        print('  Skipping environmental forcing (no-ENV mode)')
    sim.waypoints = [waypoints_lonlat]
    sim.spawn_speed = speed_ms
    sim.spawn()
    # Force the simulation to use the configured start waypoint (prevents previous runs from spawning elsewhere)
    try:
        from pyproj import Transformer
        latlon_to_utm = Transformer.from_crs('EPSG:4326', sim.crs_utm, always_xy=True)
        start_lon, start_lat = float(waypoints_lonlat[0][0]), float(waypoints_lonlat[0][1])
        sx, sy = latlon_to_utm.transform(start_lon, start_lat)
        # compute heading toward second waypoint if available
        if len(waypoints_lonlat) > 1:
            nxt_lon, nxt_lat = float(waypoints_lonlat[1][0]), float(waypoints_lonlat[1][1])
            nx, ny = latlon_to_utm.transform(nxt_lon, nxt_lat)
            psi0 = float(np.arctan2(ny - sy, nx - sx))
        else:
            psi0 = float(sim.psi[0])
        sim.pos[:, 0] = (sx, sy)
        sim.psi[0] = psi0
        try:
            gx, gy = latlon_to_utm.transform(float(waypoints_lonlat[-1][0]), float(waypoints_lonlat[-1][1]))
            sim.goals[:, 0] = (gx, gy)
        except Exception:
            pass
        # ensure ship internals reflect the enforced start
        try:
            if hasattr(sim, 'ship'):
                setattr(sim.ship, 'x', sim.pos[0, 0])
                setattr(sim.ship, 'y', sim.pos[1, 0])
        except Exception:
            pass
    except Exception:
        # best-effort only; if conversion fails keep original spawn
        pass
    if power_loss_lon is not None and power_loss_lat is not None and power_loss_radius_m is not None:
        print(f"  Running simulation (spatial power loss at lon={power_loss_lon}, lat={power_loss_lat}, radius={power_loss_radius_m} m)")
    else:
        print(f"  Running simulation (power loss at t={power_loss_time}s)")
    power_cut = False
    trajectory = []
    collision_detected = False
    collision_time = None
    # Prepare transformer to convert lon/lat -> sim UTM coordinates for spatial triggers
    try:
        from pyproj import Transformer
        latlon_to_utm = Transformer.from_crs("EPSG:4326", sim.crs_utm, always_xy=True)
        if power_loss_lon is not None and power_loss_lat is not None:
            px, py = latlon_to_utm.transform(power_loss_lon, power_loss_lat)
        else:
            px = py = None
    except Exception:
        px = py = None

    for step in range(sim.steps):
        t = step * dt
        # Time-based power-loss (from config)
        if (power_loss_lon is None or power_loss_lat is None or power_loss_radius_m is None) and t >= power_loss_time and not power_cut:
            try:
                sim.ship.cut_power(0)
                print(f"  Power loss triggered at t={t:.1f}s")
                power_cut = True
            except Exception:
                pass
        # Spatial power-loss: cut power when vessel comes within radius of provided lon/lat
        if (px is not None and py is not None and power_loss_radius_m is not None) and not power_cut:
            x_now, y_now = sim.pos[0, 0], sim.pos[1, 0]
            try:
                dist = np.sqrt((x_now - px)**2 + (y_now - py)**2)
                if dist <= float(power_loss_radius_m):
                    try:
                        sim.ship.cut_power(0)
                        print(f"  Spatial power loss triggered at t={t:.1f}s (dist={dist:.1f} m)")
                        power_cut = True
                    except Exception:
                        pass
            except Exception:
                pass
        nu = np.vstack([sim.state[0], sim.state[1], sim.state[3]])
        hd, sp, rud = sim._compute_controls_and_update(nu, t)
        sim.hd_cmds = hd
        sim._step_dynamics(hd, sp, rud)
        x, y = sim.pos[0, 0], sim.pos[1, 0]
        u = sim.state[0, 0]
        psi = sim.psi[0]
        collision_events = sim._check_collision(t)
        if collision_events and not collision_detected:
            collision_detected = True
            collision_time = t
            print(f"  COLLISION at t={t:.1f}s")
            break
        if step % max(1, int(5/cfg['dt'])) == 0:
            trajectory.append({'time_s': t, 'x_m': x, 'y_m': y, 'speed_ms': u, 'heading_rad': psi})
    final_x, final_y = sim.pos[0, 0], sim.pos[1, 0]
    bridge_lon = config['incident_data']['collision_coordinates']['lon']
    bridge_lat = config['incident_data']['collision_coordinates']['lat']
    from pyproj import Transformer
    latlon_to_utm = Transformer.from_crs("EPSG:4326", sim.crs_utm, always_xy=True)
    bridge_x, bridge_y = latlon_to_utm.transform(bridge_lon, bridge_lat)
    distance_to_bridge = np.sqrt((final_x - bridge_x)**2 + (final_y - bridge_y)**2)
    result = {'run_id': run_id, 'env_datetime': env_datetime.isoformat(), 'cbofs_loaded': cbofs_loaded, 'collision': collision_detected, 'collision_time_s': collision_time if collision_detected else None, 'final_distance_to_bridge_m': distance_to_bridge, 'simulation_completed': not collision_detected}
    print(f"  Result: {'COLLISION' if collision_detected else 'NO COLLISION'}")
    print(f"  Final distance to bridge: {distance_to_bridge:.1f}m")
    if trajectory:
        traj_df = pd.DataFrame(trajectory)
        traj_file = output_dir / f'run_{run_id:03d}_trajectory.csv'
        traj_df.to_csv(traj_file, index=False)
    return result

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--n-runs', type=int, default=10)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--no-env', action='store_true', dest='no_env', help='Skip loading environmental forcing (useful for local/speed runs)')
    parser.add_argument('--power-loss-lon', type=float, default=None, help='Longitude of spatial power loss trigger')
    parser.add_argument('--power-loss-lat', type=float, default=None, help='Latitude of spatial power loss trigger')
    parser.add_argument('--power-loss-radius-m', type=float, default=None, help='Radius (m) around lon/lat to trigger power loss')
    args = parser.parse_args()
    if args.seed:
        np.random.seed(args.seed)
    # Robust config load: read raw bytes, detect BOM/encoding, then json.loads
    from pathlib import Path as _Path
    import codecs as _codecs
    raw = _Path(args.config).read_bytes()
    cfg_text = None
    try:
        # UTF-16 LE BOM
        if raw.startswith(b"\xff\xfe") or raw.startswith(_codecs.BOM_UTF16_LE):
            cfg_text = raw.decode('utf-16')
        # UTF-16 BE BOM
        elif raw.startswith(b"\xfe\xff") or raw.startswith(_codecs.BOM_UTF16_BE):
            cfg_text = raw.decode('utf-16')
        # UTF-8 BOM
        elif raw.startswith(_codecs.BOM_UTF8) or raw.startswith(b"\xef\xbb\xbf"):
            cfg_text = raw.decode('utf-8-sig')
        else:
            # try utf-8, fall back to latin-1
            try:
                cfg_text = raw.decode('utf-8')
            except Exception:
                cfg_text = raw.decode('latin-1')
        config = json.loads(cfg_text)
    except Exception as e:
        raise RuntimeError(f"Failed to read config {args.config}: {e}")
    output_dir = Path(config['output']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    print("\nMV DALI SCENARIO - ENVIRONMENTAL SENSITIVITY ANALYSIS")
    print(f"Testing {args.n_runs} random environmental conditions")
    print("CBOFS data: Last 30 days (rolling retention)\n")
    results = []
    for i in range(args.n_runs):
        env_dt = random_recent_datetime()
        print("="*60)
        print(f"Run {i+1}: Environmental date = {env_dt}")
        print("="*60)
        result = run_single_scenario(
            config,
            env_dt,
            i+1,
            output_dir,
            power_loss_lon=args.power_loss_lon,
            power_loss_lat=args.power_loss_lat,
            power_loss_radius_m=args.power_loss_radius_m,
            skip_env=args.no_env
        )
        results.append(result)
    results_df = pd.DataFrame(results)
    summary_file = output_dir / 'environmental_sensitivity_results.csv'
    results_df.to_csv(summary_file, index=False)
    print(f"\n{'='*60}")
    print("SUMMARY STATISTICS")
    print(f"{'='*60}")
    print(f"Total runs: {len(results_df)}")
    print(f"CBOFS loaded: {results_df['cbofs_loaded'].sum()} / {len(results_df)}")
    print(f"Collisions: {results_df['collision'].sum()} / {len(results_df)}")
    print(f"Collision rate: {results_df['collision'].mean()*100:.1f}%")
    print(f"\nMean distance to bridge: {results_df['final_distance_to_bridge_m'].mean():.1f}m")
    print(f"Min distance to bridge: {results_df['final_distance_to_bridge_m'].min():.1f}m")
    print(f"Max distance to bridge: {results_df['final_distance_to_bridge_m'].max():.1f}m")
    print(f"\nResults saved to: {summary_file}")

if __name__ == '__main__':
    main()
