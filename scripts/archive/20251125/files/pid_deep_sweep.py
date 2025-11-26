"""Small PID refinement sweep around recommended gains.
Writes results to sweep_results/pid_deep_summary.csv and creates per-run pid_trace CSVs.
"""
import os
import itertools
import pandas as pd
from emergent.ship_abm.simulation_core import simulation
from emergent.ship_abm.config import CONTROLLER_GAINS, ADVANCED_CONTROLLER, PID_TRACE, SHIP_PHYSICS

OUT_DIR = 'sweep_results'
os.makedirs(OUT_DIR, exist_ok=True)

KP_CAND = [0.4, 0.5, 0.6]
KI_CAND = [0.0, 0.005, 0.02]
KD_CAND = [0.25, 0.3, 0.35]

DT = 0.1
T = 300.0
WIND_SPEED = 5.0

rows = []
base_kf = ADVANCED_CONTROLLER.get('Kf_gain', 0.005)

# resume support: read existing summary and skip completed combos
existing_summary = os.path.join(OUT_DIR, 'pid_deep_summary.csv')
done_set = set()
if os.path.exists(existing_summary):
    try:
        prev = __import__('pandas').read_csv(existing_summary)
        for _, r in prev.iterrows():
            done_set.add((float(r['Kp']), float(r['Ki']), float(r['Kd'])))
        rows = prev.to_dict('records')
    except Exception:
        done_set = set()

for Kp, Ki, Kd in itertools.product(KP_CAND, KI_CAND, KD_CAND):
    if (Kp, Ki, Kd) in done_set:
        continue
    label = f"Kp{Kp:.3f}_Ki{Ki:.3f}_Kd{Kd:.3f}"
    TRACE = os.path.join(OUT_DIR, f"pid_trace_deep_{label}.csv")
    PID_TRACE['enabled'] = True
    PID_TRACE['path'] = TRACE

    try:
        # create sim
        sim = simulation(port_name='Galveston', dt=DT, T=T, n_agents=1, load_enc=False,
                         test_mode='zigzag', zigzag_deg=15, zigzag_hold=30)
        sim.spawn()
        # attach perpendicular wind
        psi0 = float(sim.psi[0])
        import math
        wx = WIND_SPEED * math.cos(psi0 + math.pi/2.)
        wy = WIND_SPEED * math.sin(psi0 + math.pi/2.)
        sim.wind_fn = lambda lon, lat, when: __import__('numpy').tile(__import__('numpy').array([[wx, wy]]),(1,1))
        sim.current_fn = lambda lon, lat, when: __import__('numpy').zeros((1,2))

        # set controller gains (per-sim) on the spawned ship object
        try:
            sim.ship.Kp = Kp
            sim.ship.Ki = Ki
            sim.ship.Kd = Kd
            sim.ship.Kf = base_kf
        except Exception:
            from emergent.ship_abm.config import CONTROLLER_GAINS, ADVANCED_CONTROLLER
            CONTROLLER_GAINS['Kp'] = Kp
            CONTROLLER_GAINS['Ki'] = Ki
            CONTROLLER_GAINS['Kd'] = Kd
            ADVANCED_CONTROLLER['Kf_gain'] = base_kf

        sim.run()

        # read pid trace
        import pandas as pd
        df = pd.read_csv(TRACE)
        df0 = df[df['agent'] == 0]
        max_err = float(df0['err_deg'].abs().max())
        mean_err = float(df0['err_deg'].abs().mean())
        max_raw = float(df0['raw_deg'].abs().max())
        max_rud = float(df0['rud_deg'].abs().max())
        sat_frac = float((df0['rud_deg'].abs() >= __import__('math').degrees(SHIP_PHYSICS['max_rudder']) - 1e-6).sum() / len(df0))

        rows.append({
            'Kp': Kp, 'Ki': Ki, 'Kd': Kd, 'trace': TRACE,
            'max_err_deg': max_err, 'mean_err_deg': mean_err, 'max_raw_deg': max_raw,
            'max_rud_deg': max_rud, 'sat_frac': sat_frac
        })

        # quick flush
        __import__('pandas').DataFrame(rows).to_csv(os.path.join(OUT_DIR, 'pid_deep_summary.csv'), index=False)
        print('Completed', label)
    except Exception as e:
        print('Error running', label, repr(e))
        # continue with the next combo
        continue

print('Done sweep. Summary at', os.path.join(OUT_DIR, 'pid_deep_summary.csv'))
