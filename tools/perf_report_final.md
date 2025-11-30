# Final Performance Report

Benchmark: 2000 agents × 50 timesteps

## premerge
- pstats file: `tools/profile_2000_premerge.pstats`
- total time (pstats total_tt): 0.5453372000000009
- `timestep` cumulative time: 0.5446381

## postmerge
- pstats file: `tools/profile_2000_postmerge.pstats`
- total time (pstats total_tt): 0.5186600000000005
- `timestep` cumulative time: 0.5173091000000001

## final
- pstats file: `tools/profile_2000_final.pstats`
- total time (pstats total_tt): 0.9491227000000005
- `timestep` cumulative time: 0.9478846000000001

Post-merge vs premerge: -5.02% change in `timestep`
Final (battery merge) vs premerge: 74.04% change in `timestep`
Final vs postmerge: 83.23% change in `timestep`
