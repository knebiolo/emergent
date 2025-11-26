from emergent.ship_abm.simulation_core import simulation

# test various waypoint formats to exercise spawn conversion
cases = {
    'good_lonlat': [[(-122.7, 48.2), (-122.65, 48.2)]],
    'swapped_latlon': [[(48.2, -122.7), (48.2, -122.65)]],
    'strings': [[('-122.7', '48.2'), ('-122.65', '48.2')]],
    'projected_m': [[(500000.0, 5350000.0), (500500.0, 5350000.0)]],
    'bad_values': [[(None, None), ('nan', 'inf')]],
}

for name, wps in cases.items():
    print('---', name)
    sim = simulation(port_name='Rosario Strait', dt=0.1, T=10, n_agents=1, verbose=True, load_enc=False)
    sim.waypoints = wps
    try:
        state0, pos0, psi0, goals = sim.spawn()
        print('spawn ok:', pos0[:,0], psi0[0], goals[:,0])
    except Exception as e:
        print('spawn failed:', e)
