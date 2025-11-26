import time
from emergent.ship_abm.simulation_core import simulation
from emergent.ship_abm.config import xml_url

sim = simulation(port_name='Galveston', load_enc=False, verbose=True)
print('Starting background ENC load...')
import threading

def run_load():
    sim.load_enc_features(xml_url, verbose=True)

th = threading.Thread(target=run_load)
th.start()

while th.is_alive():
    prog = getattr(sim, '_enc_progress', None)
    processed = getattr(sim, '_enc_processed', None)
    print(f'Progress: {prog} processed: {processed}')
    time.sleep(1.0)

print('ENC load done')
print('Final progress:', getattr(sim, '_enc_progress', None))
