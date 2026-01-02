"""
Diagnostic script to analyze PID vibration in turbulent flow.

Runs a short simulation and analyzes:
1. Heading change rate (degrees/timestep)
2. PID error magnitude over time
3. PID adjustment magnitude over time
4. Correlation between heading changes and vibration

Outputs diagnostic plots and statistics.
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_pid_diagnostics(h5_path):
    """Analyze PID diagnostic data from simulation output."""
    
    with h5py.File(h5_path, 'r') as h5:
        # Load diagnostic arrays
        heading_delta = h5['agent_data/heading_delta'][:]  # (N, T) radians/timestep
        error_magnitude = h5['agent_data/error_magnitude'][:]  # (N, T) m/s
        pid_adjustment_magnitude = h5['agent_data/pid_adjustment_magnitude'][:]  # (N, T) m/s
        
        # Load position for trajectory analysis
        X = h5['agent_data/X'][:]  # (N, T)
        Y = h5['agent_data/Y'][:]  # (N, T)
        
        # Load velocity/flow data if available
        x_vel = h5['agent_data/x_vel'][:] if 'agent_data/x_vel' in h5 else None
        y_vel = h5['agent_data/y_vel'][:] if 'agent_data/y_vel' in h5 else None
        
    N_agents, T_steps = heading_delta.shape
    dt = 0.1  # TODO: read from config
    
    print("=" * 60)
    print("PID VIBRATION DIAGNOSTICS")
    print("=" * 60)
    print(f"Agents: {N_agents}")
    print(f"Timesteps: {T_steps}")
    print(f"Duration: {T_steps * dt:.1f}s")
    print()
    
    # Statistics
    heading_delta_deg = np.rad2deg(heading_delta)
    
    print("HEADING CHANGE RATE (degrees/timestep):")
    print(f"  Mean:   {np.nanmean(np.abs(heading_delta_deg)):.3f}")
    print(f"  Median: {np.nanmedian(np.abs(heading_delta_deg)):.3f}")
    print(f"  Max:    {np.nanmax(np.abs(heading_delta_deg)):.3f}")
    print(f"  95th percentile: {np.nanpercentile(np.abs(heading_delta_deg), 95):.3f}")
    print()
    
    print("PID ERROR MAGNITUDE (m/s):")
    print(f"  Mean:   {np.nanmean(error_magnitude):.4f}")
    print(f"  Median: {np.nanmedian(error_magnitude):.4f}")
    print(f"  Max:    {np.nanmax(error_magnitude):.4f}")
    print(f"  95th percentile: {np.nanpercentile(error_magnitude, 95):.4f}")
    print()
    
    print("PID ADJUSTMENT MAGNITUDE (m/s):")
    print(f"  Mean:   {np.nanmean(pid_adjustment_magnitude):.4f}")
    print(f"  Median: {np.nanmedian(pid_adjustment_magnitude):.4f}")
    print(f"  Max:    {np.nanmax(pid_adjustment_magnitude):.4f}")
    print(f"  95th percentile: {np.nanpercentile(pid_adjustment_magnitude, 95):.4f}")
    print()
    
    # Identify "vibrating" fish (high heading change rate)
    vibration_threshold = 5.0  # degrees/timestep
    vibrating_mask = np.abs(heading_delta_deg) > vibration_threshold
    vibration_fraction = np.sum(vibrating_mask) / vibrating_mask.size
    print(f"VIBRATION EVENTS (>{vibration_threshold}°/timestep): {vibration_fraction*100:.1f}% of timesteps")
    print()
    
    # Create diagnostic plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Heading change rate over time (all agents)
    ax = axes[0, 0]
    time = np.arange(T_steps) * dt
    for i in range(min(10, N_agents)):  # Plot first 10 agents
        ax.plot(time, np.abs(heading_delta_deg[i, :]), alpha=0.5, linewidth=0.8)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('|Heading Change| (deg/timestep)')
    ax.set_title('Heading Change Rate (first 10 agents)')
    ax.grid(True, alpha=0.3)
    ax.axhline(vibration_threshold, color='r', linestyle='--', label=f'Vibration threshold ({vibration_threshold}°)')
    ax.legend()
    
    # Plot 2: Error magnitude over time
    ax = axes[0, 1]
    for i in range(min(10, N_agents)):
        ax.plot(time, error_magnitude[i, :], alpha=0.5, linewidth=0.8)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('PID Error Magnitude (m/s)')
    ax.set_title('PID Error Over Time (first 10 agents)')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Histograms
    ax = axes[1, 0]
    ax.hist(np.abs(heading_delta_deg).flatten(), bins=50, alpha=0.7, label='Heading Δ (deg)', density=True)
    ax.axvline(vibration_threshold, color='r', linestyle='--', label=f'Threshold ({vibration_threshold}°)')
    ax.set_xlabel('|Heading Change| (deg/timestep)')
    ax.set_ylabel('Density')
    ax.set_title('Distribution of Heading Changes')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Plot 4: Correlation - heading change vs error magnitude
    ax = axes[1, 1]
    sample_indices = np.random.choice(heading_delta.size, min(10000, heading_delta.size), replace=False)
    ax.scatter(np.abs(heading_delta_deg.flatten()[sample_indices]), 
               error_magnitude.flatten()[sample_indices],
               alpha=0.1, s=1)
    ax.set_xlabel('|Heading Change| (deg/timestep)')
    ax.set_ylabel('PID Error Magnitude (m/s)')
    ax.set_title('Heading Change vs Error (10k random samples)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_dir = Path(h5_path).parent
    plot_path = output_dir / 'pid_diagnostics.png'
    plt.savefig(plot_path, dpi=150)
    print(f"Diagnostic plot saved to: {plot_path}")
    plt.show()
    
    # Compute correlation
    corr = np.corrcoef(np.abs(heading_delta.flatten()), error_magnitude.flatten())[0, 1]
    print(f"\nCorrelation (|heading_delta| vs error_magnitude): {corr:.3f}")
    
    # Recommendations
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS:")
    print("=" * 60)
    
    mean_heading_change = np.nanmean(np.abs(heading_delta_deg))
    mean_error = np.nanmean(error_magnitude)
    
    if mean_heading_change > 2.0:
        print("⚠ HIGH heading change rate detected (>2°/timestep)")
        print("  → Behavioral cues may be updating too rapidly")
        print("  → Consider smoothing heading changes or adding heading rate limits")
    
    if mean_error > 0.05:
        print("⚠ HIGH PID error magnitude detected (>0.05 m/s)")
        print("  → Current PID gains (k_p=1.0, k_i=0, k_d=0) may be insufficient")
        print("  → Recommend adding derivative term: k_d = 0.1 to 0.5")
    
    if vibration_fraction > 0.1:
        print(f"⚠ HIGH vibration rate ({vibration_fraction*100:.1f}% of timesteps)")
        print("  → PID is oscillating - needs damping")
        print("  → Recommend k_d = 0.2 or error filtering (moving average)")
    
    if corr > 0.5:
        print(f"✓ Strong correlation ({corr:.2f}) between heading change and error")
        print("  → Confirms hypothesis: rapid heading changes cause PID oscillation")
        print("  → Solution: Add derivative damping OR limit heading change rate")
    
    print()

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        h5_path = sys.argv[1]
    else:
        # Use most recent output
        import glob
        outputs = sorted(glob.glob('outputs/*/sim_*.h5'), key=lambda p: Path(p).stat().st_mtime, reverse=True)
        if outputs:
            h5_path = outputs[0]
            print(f"Using most recent output: {h5_path}\n")
        else:
            print("No HDF5 outputs found. Run a simulation first.")
            sys.exit(1)
    
    analyze_pid_diagnostics(h5_path)
