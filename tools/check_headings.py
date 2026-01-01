import pandas as pd
import numpy as np
import sys

trace_file = sys.argv[1] if len(sys.argv) > 1 else "outputs/collision_fixed/nuyakuk_headless_trace.csv"

df = pd.read_csv(trace_file)
final = df[df["timestep"] == df["timestep"].max()]

# Calculate headings from velocity
headings = np.degrees(np.arctan2(final["y_vel"], final["x_vel"]))
headings = (450 - headings) % 360  # Convert to geographic (0=North, 90=East)

# Mutually exclusive quadrants
north_count = np.sum((headings >= 315) | (headings < 45))  # 315-360 and 0-45
east_count = np.sum((headings >= 45) & (headings < 135))  # 45-135
south_count = np.sum((headings >= 135) & (headings < 225))  # 135-225  
west_count = np.sum((headings >= 225) & (headings < 315))  # 225-315

print(f"Final headings (timestep {final['timestep'].iloc[0]}):")
print(f"North: {north_count} ({100*north_count/len(final):.1f}%)")
print(f"East: {east_count} ({100*east_count/len(final):.1f}%)")
print(f"South: {south_count} ({100*south_count/len(final):.1f}%)")
print(f"West: {west_count} ({100*west_count/len(final):.1f}%)")
print(f"Total: {len(final)} agents")
