import importlib.util
from pathlib import Path

tests_dir = Path(__file__).resolve().parent
pkg_dir = tests_dir.parent
sim_path = pkg_dir / 'simulation.py'
spec = importlib.util.spec_from_file_location('salmon_abm.simulation', str(sim_path))
sim_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sim_mod)
simulation = sim_mod.simulation


def test_simulation_instantiation():
    sim = simulation(model_dir='.', model_name='m', crs=None, basin='b', water_temp=10.0, start_polygon=None, env_files={}, longitudinal_profile=None, num_agents=5)
    assert sim.num_agents == 5
    assert sim.run(n=2) is True
