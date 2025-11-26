import numpy as np
import time
from emergent.ship_abm.simulation_core import simulation
from emergent.ship_abm.config import SIMULATION_BOUNDS

def zero_wind(lon, lat, when):
    # lon, lat may be arrays or sequences; return shape (2, n)
    try:
        n = len(lon)
    except Exception:
        n = 1
    # Return shape (n,2) so caller can transpose -> (2,n)
    return np.zeros((n, 2))

def zero_current(lon, lat, when):
    try:
        n = len(lon)
    except Exception:
        n = 1
    # Return shape (n,2) so caller can transpose -> (2,n)
    return np.zeros((n, 2))

def main():
    # choose a valid port from config (fallback to first key)
    port = 'TestPort'
    if port not in SIMULATION_BOUNDS:
        port = next(iter(SIMULATION_BOUNDS.keys()))

    # short run: 3 agents, dt=0.5s, T=10s
    sim = simulation(port_name=port, dt=0.5, T=5.0, n_agents=3, verbose=False, use_ais=False, test_mode='headless', load_enc=False)

    # override environmental functions to simple zeros to avoid external data
    sim.wind_fn = zero_wind
    sim.current_fn = zero_current

    print('Starting minimal headless simulation...')
    # ensure simple waypoints exist to avoid ENC-dependent spawning logic
    try:
        n_ships = getattr(sim, 'n_agents', None) or getattr(sim, 'n', None) or (getattr(sim, 'pos', None).shape[1] if getattr(sim, 'pos', None) is not None else None)
    except Exception:
        n_ships = None
    if not n_ships:
        n_ships = 3
    # create a single goal 500m east for each ship
    # Use plain Python lists of (x,y) tuples to satisfy truthiness checks in sim
    try:
        sim.waypoints = [[(500.0, 0.0)] for _ in range(int(n_ships))]
    except Exception:
        try:
            # fallback: minimal structure
            sim.waypoints = [[(500.0, 0.0)] * 1 for _ in range(int(n_ships))]
        except Exception:
            pass

    # Spawn agents to instantiate ship model and state
    try:
        sim.spawn()
    except Exception as e:
        print('spawn() failed:', repr(e))

    start = time.time()
    try:
        sim.run()
        print('Simulation finished in', time.time() - start)
        # summary
        try:
            print('Final positions (x,y):')
            print(sim.pos)
        except Exception:
            pass
    except Exception as e:
        print('Simulation run failed:', repr(e))
        # print partial state if available
        try:
            print('Positions at failure (if any):')
            print(getattr(sim, 'pos', None))
        except Exception:
            pass
        raise

if __name__ == '__main__':
    main()
