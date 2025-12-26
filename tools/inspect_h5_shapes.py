import sys
import h5py

def inspect(path):
    f = h5py.File(path, 'r')
    print('Top keys:', list(f.keys()))
    if 'environment' in f:
        print('Environment keys:', list(f['environment'].keys()))
    if 'agent_data' in f:
        print('Agent_data keys:', list(f['agent_data'].keys()))
        for k in f['agent_data'].keys():
            try:
                print(k, 'shape=', f['agent_data'][k].shape)
            except Exception as e:
                print(k, 'error reading shape', e)
    f.close()

if __name__ == '__main__':
    p = sys.argv[1] if len(sys.argv)>1 else 'outputs/diagnostics/det_rheo_50x2_headless.h5'
    inspect(p)
