import h5py
import sys

def copy_memory(src_headless, dst_preseed):
    with h5py.File(src_headless, 'r') as src, h5py.File(dst_preseed, 'a') as dst:
        if 'memory' not in src:
            print('Source has no memory group:', src_headless)
            return 1
        # remove existing memory in dst
        if 'memory' in dst:
            del dst['memory']
        dst.copy(src['memory'], 'memory')
        print('Copied memory group from', src_headless, 'to', dst_preseed)
    return 0

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: python copy_memory_group.py <src_headless.h5> <dst_preseed.h5>')
        sys.exit(2)
    src = sys.argv[1]
    dst = sys.argv[2]
    sys.exit(copy_memory(src, dst))
