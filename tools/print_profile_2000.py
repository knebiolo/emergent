import pstats
ps = pstats.Stats('tools/profile_2000.pstats')
ps.sort_stats('cumulative').print_stats(50)
