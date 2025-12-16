import os
import numpy as np

from emergent.fish_passage.movie_maker import movie_maker


def test_movie_maker_fallback_frames(tmp_path):
    # Create small X/Y arrays: 3 agents, 5 timesteps
    X = np.random.rand(3, 5) * 1000
    Y = np.random.rand(3, 5) * 1000
    np.save(os.path.join(str(tmp_path), 'X.npy'), X)
    np.save(os.path.join(str(tmp_path), 'Y.npy'), Y)

    out = movie_maker(str(tmp_path), 'testmodel', None, 1.0, None)
    assert os.path.exists(out)
    # should be a frames directory with at least 5 entries
    entries = os.listdir(out)
    assert len(entries) >= 5
