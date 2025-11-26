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
                        power_loss_lon=None, power_loss_lat=None, power_loss_radius_m=None):
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
    try:
        print(f"  Loading CBOFS for {env_datetime}...")
        sim.load_environmental_forcing(start=env_datetime)
        cbofs_loaded = True
        print(f"  CBOFS loaded successfully")
    except Exception as e:
        print(f"  Warning: Could not load CBOFS: {e}")
    sim.waypoints = [waypoints_lonlat]
    sim.spawn_speed = speed_ms
    sim.spawn()
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
    parser.add_argument('--power-loss-lon', type=float, default=None, help='Longitude of spatial power loss trigger')
    parser.add_argument('--power-loss-lat', type=float, default=None, help='Latitude of spatial power loss trigger')
    parser.add_argument('--power-loss-radius-m', type=float, default=None, help='Radius (m) around lon/lat to trigger power loss')
    args = parser.parse_args()
    if args.seed:
        np.random.seed(args.seed)
    with open(args.config, 'r') as f:
        config = json.load(f)
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
            power_loss_radius_m=args.power_loss_radius_m
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
