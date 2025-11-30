import sys
import pstats

def main(pstats_path='tools/profile_2000_final_steady.pstats', top=30):
    p = pstats.Stats(pstats_path)
    p.sort_stats('cumulative')
    p.print_stats(top)

if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else 'tools/profile_2000_final_steady.pstats'
    top = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    main(path, top)
