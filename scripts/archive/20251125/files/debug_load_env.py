import traceback
from datetime import datetime

print('DEBUG: starting simulation import and env load test')
try:
    from emergent.ship_abm.simulation_core import simulation
    print('DEBUG: imported simulation')
except Exception as e:
    print('DEBUG: import failed')
    traceback.print_exc()
    raise

try:
    sim = simulation(port_name='Baltimore', dt=0.1, T=60, n_agents=0, load_enc=False, verbose=True)
    print('DEBUG: simulation instance created')
except Exception as e:
    print('DEBUG: simulation construction failed')
    traceback.print_exc()
    raise

try:
    print('DEBUG: calling load_environmental_forcing() (threaded, 30s timeout)')
    import threading
    err = [None]
    def target():
        try:
            sim.load_environmental_forcing(start=datetime.utcnow())
        except Exception as e:
            err[0] = e
            traceback.print_exc()
    th = threading.Thread(target=target)
    th.start()
    th.join(30.0)
    if th.is_alive():
        print('DEBUG: load_environmental_forcing did not finish within 30s (likely network/blocking).')
    elif err[0] is not None:
        print('DEBUG: load_environmental_forcing raised an exception (see traceback above)')
    else:
        print('DEBUG: load_environmental_forcing completed within timeout')
except Exception as e:
    print('DEBUG: load_environmental_forcing failed')
    traceback.print_exc()
    raise

print('DEBUG: done')
