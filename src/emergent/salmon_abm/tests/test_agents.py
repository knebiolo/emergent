import numpy as np
import importlib.util
from pathlib import Path

tests_dir = Path(__file__).resolve().parent
pkg_dir = tests_dir.parent
agents_path = pkg_dir / 'agents.py'
spec = importlib.util.spec_from_file_location('salmon_abm.agents', str(agents_path))
agents = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agents)


def test_generate_sex_length_weight_bodydepth():
    n = 10
    sex = agents.generate_sex(n, seed=42)
    assert sex.shape == (n,)
    length = agents.generate_length(n, seed=42)
    assert length.shape == (n,)
    assert np.all(length >= 475.0)
    weight = agents.generate_weight(length)
    assert weight.shape == (n,)
    body = agents.generate_body_depth(n, seed=42)
    assert body.shape == (n,)
