"""Check what distance_to looks like in the HDF."""
import h5py
import numpy as np

hdf_path = r"data\salmon_abm\20240506\Nuyakuk_Production_.p05.hdf"

with h5py.File(hdf_path, 'r') as f:
    print("Keys in HDF:")
    for key in f.keys():
        print(f"  {key}")
    
    if 'environment' in f:
        print("\nKeys in environment:")
        for key in f['environment'].keys():
            print(f"  {key}")
        
        if 'distance_to' in f['environment']:
            dist = f['environment/distance_to'][:]
            print(f"\nDistance_to shape: {dist.shape}")
            print(f"Distance_to range: {dist.min():.2f} to {dist.max():.2f}")
            print(f"Distance_to dtype: {dist.dtype}")
            print(f"Pixels > 0: {np.sum(dist > 0)}")
            print(f"Pixels > 0.5: {np.sum(dist > 0.5)}")
            print(f"Pixels > 1: {np.sum(dist > 1)}")
            print(f"\nSample values (first 10x10):")
            print(dist[:10, :10])
