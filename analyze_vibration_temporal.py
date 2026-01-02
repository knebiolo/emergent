"""
Analyze temporal pattern of heading vibration - is it initialization chaos?
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt

with h5py.File('outputs/pid_diagnostic_test/sim_pid_test.h5', 'r') as h5:
    heading_delta = h5['agent_data/heading_delta'][:]  # (N, T)
    
N_agents, T_steps = heading_delta.shape
dt = 0.1  # seconds per timestep

# Convert to degrees
heading_delta_deg = np.rad2deg(np.abs(heading_delta))

# Analyze by time windows
window_size = 20  # 2 seconds = 20 timesteps at dt=0.1
n_windows = T_steps // window_size

print("=" * 60)
print("TEMPORAL VIBRATION ANALYSIS")
print("=" * 60)
print(f"Total timesteps: {T_steps} ({T_steps * dt:.1f}s)")
print(f"Analyzing in {window_size}-timestep windows ({window_size * dt:.1f}s each)")
print()

# Compute statistics per time window
vibration_threshold = 5.0  # degrees/timestep

for w in range(n_windows):
    start_t = w * window_size
    end_t = (w + 1) * window_size
    
    window_data = heading_delta_deg[:, start_t:end_t]
    
    mean_change = np.mean(window_data)
    max_change = np.max(window_data)
    vibration_count = np.sum(window_data > vibration_threshold)
    vibration_pct = 100 * vibration_count / window_data.size
    
    time_label = f"{start_t * dt:.1f}-{end_t * dt:.1f}s"
    print(f"Window {w+1} ({time_label}):")
    print(f"  Mean: {mean_change:.3f}°  Max: {max_change:.1f}°  Vibration: {vibration_pct:.1f}% (>{vibration_threshold}°)")

print()
print("=" * 60)

# Compare first 2 seconds vs last 18 seconds
early_cutoff = int(2.0 / dt)  # First 2 seconds = 20 timesteps
early_data = heading_delta_deg[:, :early_cutoff]
late_data = heading_delta_deg[:, early_cutoff:]

early_mean = np.mean(early_data)
late_mean = np.mean(late_data)
early_max = np.max(early_data)
late_max = np.max(late_data)
early_vib_pct = 100 * np.sum(early_data > vibration_threshold) / early_data.size
late_vib_pct = 100 * np.sum(late_data > vibration_threshold) / late_data.size

print("EARLY (0-2s) vs LATE (2-20s) COMPARISON:")
print(f"  Early mean: {early_mean:.3f}°  Late mean: {late_mean:.3f}°")
print(f"  Early max:  {early_max:.1f}°   Late max:  {late_max:.1f}°")
print(f"  Early vibration: {early_vib_pct:.1f}%  Late vibration: {late_vib_pct:.1f}%")
print()

if early_vib_pct > 2 * late_vib_pct:
    print("✓ CONFIRMED: Vibration is primarily an INITIALIZATION issue")
    print("  → Most heading chaos occurs in first 2 seconds")
    print("  → Likely caused by:")
    print("    - Random initial positions creating immediate collision responses")
    print("    - Agents initializing heading from flow direction (may conflict with spawned heading)")
    print("    - Schooling cues stabilizing after initial neighbor discovery")
else:
    print("✗ Vibration persists throughout simulation")
    print("  → Continuous behavioral issue, not just initialization")

# Plot vibration over time
fig, ax = plt.subplots(figsize=(12, 6))
time = np.arange(T_steps) * dt

# Plot max vibration per timestep across all agents
max_per_timestep = np.max(heading_delta_deg, axis=0)
mean_per_timestep = np.mean(heading_delta_deg, axis=0)

ax.plot(time, max_per_timestep, 'r-', alpha=0.5, linewidth=0.8, label='Max (any agent)')
ax.plot(time, mean_per_timestep, 'b-', linewidth=1.5, label='Mean (all agents)')
ax.axhline(vibration_threshold, color='orange', linestyle='--', label=f'Vibration threshold ({vibration_threshold}°)')
ax.axvline(2.0, color='green', linestyle='--', alpha=0.5, label='2s mark')

ax.set_xlabel('Time (s)', fontsize=12)
ax.set_ylabel('|Heading Change| (deg/timestep)', fontsize=12)
ax.set_title('Heading Vibration Over Time', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/pid_diagnostic_test/vibration_temporal.png', dpi=150)
print(f"\nTemporal plot saved to: outputs/pid_diagnostic_test/vibration_temporal.png")
# plt.show()  # Disabled to avoid blocking
