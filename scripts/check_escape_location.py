#!/usr/bin/env python
"""Check where agents escaped the domain."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('outputs/production_5000agents_900sec_convexhull/nuyakuk_headless_trace.csv')
print(f'Trace has {len(df)} rows (5000 agents × {len(df)//5000} timesteps)')
print(f'Last timestep: {df["timestep"].max()}')

# Get final positions
last_t = df["timestep"].max()
final = df[df["timestep"] == last_t]

print(f'\nFinal state (t={last_t}):')
print(f'  Mean Position: ({final["x"].mean():.2f}, {final["y"].mean():.2f})')
print(f'  Position Spread: x_std={final["x"].std():.2f}, y_std={final["y"].std():.2f}')
print(f'  X range: {final["x"].min():.2f} to {final["x"].max():.2f}')
print(f'  Y range: {final["y"].min():.2f} to {final["y"].max():.2f}')

print(f'\nStarting polygon bounds (convex hull):')
print(f'  X: 549702.46 to 550003.42')
print(f'  Y: 6641288.94 to 6641440.62')

print(f'\nEscape location from error: (549728.48, 6641574.41)')
print(f'  This is {6641574.41 - 6641440.62:.2f}m NORTH of polygon max Y')
print(f'  Agents pushed out the north boundary by collision forces')

# Plot final positions
plt.figure(figsize=(10, 8))
plt.scatter(final["x"], final["y"], s=1, alpha=0.5, label='Agents')
plt.axhline(6641440.62, color='red', linestyle='--', label='North boundary (start polygon)')
plt.axhline(6641288.94, color='red', linestyle='--', label='South boundary (start polygon)')
plt.axvline(549702.46, color='red', linestyle='--', label='West boundary')
plt.axvline(550003.42, color='red', linestyle='--', label='East boundary')
plt.scatter([549728.48], [6641574.41], s=100, color='red', marker='X', label='Escape location')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.title(f'Agent positions at t={last_t} (failure)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.tight_layout()
plt.savefig('outputs/production_5000agents_900sec_convexhull/escape_location.png', dpi=150)
print(f'\nSaved plot to: outputs/production_5000agents_900sec_convexhull/escape_location.png')
plt.show()
