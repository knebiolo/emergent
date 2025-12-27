import numpy as np
import os

out = 'outputs/diagnostics/test_arbitrate_synth.npz'
N = 10
# synthetic cues: cohesion (pointing right), alignment (pointing up), rheo (pointing left)
cohesion = np.tile(np.array([1.0, 0.0]), (N,1))
alignment = np.tile(np.array([0.0, 1.0]), (N,1))
rheo = np.tile(np.array([-1.0, 0.0]), (N,1))
# sum
head = cohesion + alignment + rheo
# normalize head for realism (but keep direction same)
# produce also magnitudes
np.savez_compressed(out, cohesion_vec=cohesion, alignment_vec=alignment, rheo_vec=rheo, cohesion_mag=np.linalg.norm(cohesion,axis=1), alignment_mag=np.linalg.norm(alignment,axis=1), rheo_mag=np.linalg.norm(rheo,axis=1), head_vec=head)
print('Wrote synthetic NPZ:', out)
