import numpy as np
from scipy.optimize import differential_evolution
from emergent.ship_abm.simulation_core import simulation
from emergent.ship_abm.simulation_core import compute_zigzag_metrics, compute_turning_circle_metrics
from emergent.ship_abm.config import PROPULSION

# --- USER-DEFINED TARGET METRICS FOR ULCS ---
L = 400.0  # ship length in meters

# These reflect typical maneuvering performance for ultra-large container ships
# Reference values drawn from ABS/IMO and ship builder reports

target_metrics = {
    "zigzag_overshoot_deg": 7.0,
    "settling_time_s": 40.0,
    "oscillation_period_s": 60.0,
    "tactical_diameter_m": 3.2 * L,
    "advance_m": 3.0 * L,
    "transfer_m": 1.5 * L
}

# --- PARAM NAMES TO OPTIMIZE (physics only) ---
param_names = [
    "Ixx", "Izz", "xG", "zG", "Xu", "Yv", "Yr", "Nv", "Nr", "Ydelta", "Ndelta", "max_rudder_rate"]


# --- BASELINE CONFIG FOR DIFFERENT SHIP TYPES ---
SHIP_TYPES = {
    "ULCS": {
        'length': 400.0,            # Length overall of the ship (m)
        'beam': 60.0,               # Beam (width) of the ship (m)
        'draft': 10.0,              # Vertical draft of the ship (m)
        'm': 1384224260.293955,                   # Mass of the vessel (kg)
        'Ixx': 9627947989.566050,                 # Moment of inertia about the x‐axis (roll) (kg·m²)
        'Izz': 1.0e10,                 # Moment of inertia about the z‐axis (yaw) (kg·m²)
        'xG': 0.0,                  # Longitudinal offset of center of gravity from geometric center (m)
        'zG': -7.035222,                 # Vertical offset of center of gravity below waterline (m)
        'Xu': -6504565.964343,               # Surge force derivative w.r.t. surge velocity (kg/s)
        'Xv': 0.0,                  # Surge force derivative w.r.t. sway velocity (kg/s)
        'Xp': 0.0,                  # Surge force derivative w.r.t. roll rate (kg·m)
        'Xr': 0.0,                  # Surge force derivative w.r.t. yaw rate (kg·m)
        'Yu': 0.0,                  # Sway force derivative w.r.t. surge velocity (kg/s)
        'Yv': -632813.296804,             # Sway force derivative w.r.t. sway velocity (kg/s)
        'Yp': 0.0,                  # Sway force derivative w.r.t. roll rate (kg·m)
        'Yr': -60675001.243096,                  # Sway force derivative w.r.t. yaw rate (kg·m)
        'Ku': 0.0,                  # Roll moment derivative w.r.t. surge velocity (kg·m²/s)
        'Kv': 0.0,                  # Roll moment derivative w.r.t. sway velocity (kg·m²/s)
        'Kp': 0.0,                  # Roll moment derivative w.r.t. roll rate (kg·m²)
        'Kr': 0.0,                  # Roll moment derivative w.r.t. yaw rate (kg·m²)
        'Nu': 0.0,                  # Yaw moment derivative w.r.t. surge velocity (kg·m²/s)
        'Nv': -15629206373.796013,                # Yaw moment derivative w.r.t. sway velocity (kg·m²/s)
        'Np': 0.0,                  # Yaw moment derivative w.r.t. roll rate (kg·m²)
        'Nr': -15041061.340577,                  # Yaw moment derivative w.r.t. yaw rate (kg·m²)
        'Ydelta': 38125441.531057,   # 3.4e7 N of lateral force per radian of rudder
        'Kdelta': 0.0,     # N·m of roll moment per radian (ignored)
        'Ndelta': 263166673.737946,   # 6.8e9 N·m of yaw torque per radian of rudder
        'linear_damping': 1.0e5,      # Linear hull damping coefficient (N per m/s)
        'quad_damping': 1.0e4,        # Quadratic hull damping coefficient (N per (m/s)²)
        'max_rudder': np.radians(35),         # Maximum rudder deflection (rad)
        'max_rudder_rate': np.radians(20.0),  # Maximum rudder rate change (rad/s)
        'drag_coeff': 0.0012
    }
}

# --- BOUND GENERATOR ---
def make_bounds(center_dict, frac=0.5):
    return [(v * (1 - frac), v * (1 + frac)) for v in center_dict.values()]

# --- PATCH TO UPDATE HYDRO COEFFS ---
def update_hydro_coeffs(params):
    from emergent.ship_abm import ship_model
    for k, v in params.items():
        setattr(ship_model, k, v)

# --- WRAPPER TO RUN SIM AND EXTRACT METRICS ---
def run_test_and_get_metrics(params):
    update_hydro_coeffs(params)
    L = SHIP_TYPES["ULCS"]["length"]

    # Run zigzag
    sim_zigzag = simulation(
        port_name="Baltimore",
        dt=0.1,
        T=300.0,
        n_agents=1,
        test_mode="zigzag",
        zigzag_deg=20,
        zigzag_hold=40.0,
        use_ais=False,
        load_enc=False
    )
    sim_zigzag.spawn()
    sim_zigzag.run()
    actual_rad = sim_zigzag.psi_history
    commanded_rad = sim_zigzag.hd_cmd_history
    t = np.linspace(0, sim_zigzag.t, len(actual_rad))
    zigzag = compute_zigzag_metrics(t, actual_rad, commanded_rad)

    # Run turning circle
    sim_turn = simulation(
        port_name="Baltimore",
        dt=0.1,
        T=600.0,
        n_agents=1,
        test_mode="turning_circle",
        use_ais=False,
        load_enc=False,
    )
    sim_turn.spawn()
    sim_turn.run()
    from emergent.ship_abm.simulation_core import (
        compute_turning_advance,
        compute_turning_transfer,
        compute_tactical_diameter,
    )
    
    turning = {
        "advance_m": compute_turning_advance(sim_turn),
        "transfer_m": compute_turning_transfer(sim_turn),
        "tactical_diameter_m": compute_tactical_diameter(sim_turn),
    }

    # Merge metrics
    return {**zigzag, **turning}

# --- OBJECTIVE FUNCTION ---
def objective(params_array):
    params = dict(zip(param_names, params_array))
    sim_metrics = run_test_and_get_metrics(params)
    loss = 0.0
    for k in target_metrics:
        if k in sim_metrics:
            diff = sim_metrics[k] - target_metrics[k]
            norm = target_metrics[k] if target_metrics[k] != 0 else 1
            loss += (diff / norm) ** 2
    return loss

# --- GENERATE BOUNDS FROM BASELINE SHIP TYPE ---
base_params = SHIP_TYPES["ULCS"]

def make_bounds(center_dict, frac=0.5):
    bounds = []
    for k in param_names:
        v = center_dict[k]
        low = v * (1 - frac)
        high = v * (1 + frac)
        if high < low:
            low, high = high, low
        bounds.append((low, high))
    return bounds
bounds = make_bounds(base_params, frac=0.5)


# --- RUN OPTIMIZATION ---
if __name__ == "__main__":
    result = differential_evolution(
        func=objective,
        bounds=make_bounds(SHIP_TYPES["ULCS"]),
        strategy='best1bin',
        maxiter=30,
        popsize=15,
        tol=1e-5,
        mutation=(0.5, 1),
        recombination=0.7,
        seed=None,
        callback=None,
        disp=True,
        polish=True,
        init='latinhypercube'
    )

    best_params = dict(zip(param_names, result.x))
    print("\nOptimized Parameters:")
    for k, v in best_params.items():
        print(f"  {k:>3} = {v:.6f}")
    print(f"\nFinal Loss: {result.fun:.6f}")
