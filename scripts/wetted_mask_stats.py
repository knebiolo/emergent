import h5py
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

p = 'data/Nuyakuk_Production_.p08.hdf'
with h5py.File(p,'r') as h:
    ds = h['Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/2D area/Cell Hydraulic Depth']
    coords = np.array(h['Geometry/2D Flow Areas/2D area/Cells Center Coordinate'])
    t=90
    depth = np.array(ds[t])
    mask = np.isfinite(depth) & (depth > 0.05)
    cand_idx = np.nonzero(mask)[0]
    print('initial wetted count', cand_idx.size, 'of', depth.size)
    if cand_idx.size > 1:
        cand_coords = coords[cand_idx][:, :2]
        tree = cKDTree(cand_coords)
        dists, inds = tree.query(cand_coords, k=2)
        median_spacing = float(np.median(dists[:, 1])) if dists.shape[1] > 1 else float(np.median(dists))
        radius = max(median_spacing * 1.5, median_spacing + 1e-6)
        pairs = tree.query_pairs(r=radius, output_type='ndarray')
        if pairs.size > 0:
            row = pairs[:, 0]
            col = pairs[:, 1]
            data = np.ones(len(row), dtype=np.int8)
            row_sym = np.concatenate([row, col])
            col_sym = np.concatenate([col, row])
            data_sym = np.concatenate([data, data])
            graph = csr_matrix((data_sym, (row_sym, col_sym)), shape=(len(cand_coords), len(cand_coords)))
            ncomp, labels = connected_components(csgraph=graph, directed=False)
            counts = np.bincount(labels)
            largest = int(np.argmax(counts))
            keep_local = (labels == largest)
            new_mask = np.zeros_like(mask, dtype=bool)
            new_mask[cand_idx[keep_local]] = True
            print('kept after connectivity filter', new_mask.sum())
        else:
            print('no connectivity pairs found')
    else:
        print('not enough wetted points')
