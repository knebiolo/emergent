import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import json
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from emergent.ship_abm.simulation_core import simulation

def load_config(path):
    with open(path) as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    
    config = load_config(args.config)
    cfg = config['simulation']
    
    if cfg.get('random_seed'):
        np.random.seed(cfg['random_seed'])
    
    dt_str = config['environmental']['start_datetime'].replace('Z', '+00:00')
    start_dt = datetime.fromisoformat(dt_str)
    power_loss_time = config['events']['power_loss_schedule'][0]['time_s']
    
    print(f"MV Dali Scenario - {cfg['port_name']}")
    print(f"Start: {start_dt}, Duration: {cfg['T']}s")
    print(f"Power loss scheduled at t={power_loss_time}s")
    
    sim = simulation(
        port_name=cfg['port_name'],
        dt=cfg['dt'],
        T=cfg['T'],
        n_agents=cfg['n_agents'],
        load_enc=cfg.get('load_enc', True),
        verbose=False
    )
    
    if config['environmental'].get('use_cbofs', True):
        print("Loading CBOFS environmental forcing...")
        sim.load_environmental_forcing(start=start_dt)
    
    waypoints = config['route']['waypoints_lonlat']
    speed_ms = config['vessel']['initial_conditions']['speed_knots'] * 0.514444
    
    sim.waypoints = [waypoints]
    sim.spawn_speed = speed_ms
    sim.spawn()
    
    print(f"Vessel spawned: {waypoints[0]} → {waypoints[1]}")
    print("Running simulation...")
    
    # Custom run loop to inject power loss
    n_steps = int(cfg['T'] / cfg['dt'])
    for step in range(n_steps):
        t = step * cfg['dt']
        sim.t = t
        
        # Inject power loss event
        if abs(t - power_loss_time) < cfg['dt']/2:
            print(f"\n⚠ POWER LOSS at t={t:.1f}s\n")
            sim.ship.cut_power(0)
        
        # Update goals
        if sim.test_mode not in ("zigzag", "turning_circle"):
            sim._update_goals()
        
        # Controls and dynamics
        nu = np.vstack([sim.state[0], sim.state[1], sim.state[3]])
        hd, sp, rud = sim._compute_controls_and_update(nu, t)
        sim.hd_cmds = hd
        sim._step_dynamics(hd, sp, rud)
        
        # Collision check
        collision_events = sim._check_collision(t)
        if collision_events:
            print(f"\n🚨 COLLISION DETECTED at t={t:.1f}s")
            for ev in collision_events:
                print(f"   Vessels {ev['i']} and {ev['j']}")
            break
    
    print(f"Simulation complete at t={t:.1f}s")
    print(f"Check logs/cut_power_events.log for details")

if __name__ == '__main__':
    main()