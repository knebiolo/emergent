"""
Galveston Ship Simulation Test
Test ship ABM with real environmental forcing (currents + winds)

Galveston is the only harbor with confirmed working currents AND winds:
- Currents: NGOFS2 FVCOM (9.3s load, 0.044 m/s spatial variation)
- Winds: NGOFS2 stations (30.3s load, 3.0 m/s spatial variation)
"""
import sys
sys.path.insert(0, r"c:\Users\Kevin.Nebiolo\OneDrive - Kleinschmidt Associates\Software\emergent\src")

import numpy as np
from datetime import datetime, timedelta
from emergent.ship_abm.ofs_loader import get_current_fn
from emergent.ship_abm.atmospheric import wind_sampler
from emergent.ship_abm.config import SIMULATION_BOUNDS

# Galveston bounds
HARBOR = "Galveston"
bounds = SIMULATION_BOUNDS[HARBOR]
bbox = (bounds['minx'], bounds['maxx'], bounds['miny'], bounds['maxy'])

print("="*70)
print("GALVESTON SHIP SIMULATION TEST")
print("="*70)
print(f"\nHarbor: {HARBOR}")
print(f"Bounds: lon=[{bounds['minx']:.2f}, {bounds['maxx']:.2f}], "
      f"lat=[{bounds['miny']:.2f}, {bounds['maxy']:.2f}]")

# Simulation parameters
START_TIME = datetime.now()
SIM_DURATION = timedelta(hours=1)  # 1 hour simulation
TIMESTEP = 10.0  # seconds
NUM_SHIPS = 3

print(f"\nSimulation Parameters:")
print(f"  Start time: {START_TIME}")
print(f"  Duration: {SIM_DURATION}")
print(f"  Timestep: {TIMESTEP}s")
print(f"  Number of ships: {NUM_SHIPS}")

# Load environmental forcing
print(f"\n{'='*70}")
print("LOADING ENVIRONMENTAL FORCING")
print(f"{'='*70}")

print("\n1. Loading ocean currents (NGOFS2)...")
current_fn = get_current_fn(HARBOR, START_TIME)
print("   ✓ Current function ready")

print("\n2. Loading winds (NGOFS2 stations)...")
wind_fn = wind_sampler(bbox, START_TIME)
print("   ✓ Wind function ready")

# Test environmental forcing at a point
test_lon = (bounds['minx'] + bounds['maxx']) / 2
test_lat = (bounds['miny'] + bounds['maxy']) / 2
print(f"\n3. Testing at center point ({test_lon:.2f}, {test_lat:.2f}):")

lons = np.array([test_lon])
lats = np.array([test_lat])

current = current_fn(lons, lats, START_TIME)
u_curr, v_curr = current[0, 0], current[0, 1]
curr_speed = np.sqrt(u_curr**2 + v_curr**2)
print(f"   Current: u={u_curr:.3f} m/s, v={v_curr:.3f} m/s, speed={curr_speed:.3f} m/s")

wind = wind_fn(lons, lats, START_TIME)
u_wind, v_wind = wind[0, 0], wind[0, 1]
wind_speed = np.sqrt(u_wind**2 + v_wind**2)
print(f"   Wind:    u={u_wind:.3f} m/s, v={v_wind:.3f} m/s, speed={wind_speed:.3f} m/s")

# Initialize ships
print(f"\n{'='*70}")
print("INITIALIZING SHIPS")
print(f"{'='*70}")

ships = []
for i in range(NUM_SHIPS):
    # Spread ships across the domain
    frac = (i + 1) / (NUM_SHIPS + 1)
    lon = bounds['minx'] + frac * (bounds['maxx'] - bounds['minx'])
    lat = bounds['miny'] + frac * (bounds['maxy'] - bounds['miny'])
    
    # Random heading
    heading = np.random.uniform(0, 360)
    
    # Create simple ship dictionary (not using ship_model.py class)
    ship = {
        'id': i,
        'x': lon,
        'y': lat,
        'heading': heading,
        'speed': 5.0,  # 5 m/s (~10 knots) target speed
        'length': 100.0,  # 100m vessel
        'beam': 20.0,  # 20m beam
        'draft': 8.0,  # 8m draft
    }
    ships.append(ship)
    
    print(f"\nShip {i}:")
    print(f"  Position: ({lon:.4f}, {lat:.4f})")
    print(f"  Heading: {heading:.1f}°")
    print(f"  Target speed: 5.0 m/s (~10 knots)")

# Run simulation
print(f"\n{'='*70}")
print("RUNNING SIMULATION")
print(f"{'='*70}")

current_time = START_TIME
end_time = START_TIME + SIM_DURATION
step_count = 0
max_steps = int(SIM_DURATION.total_seconds() / TIMESTEP)

print(f"\nSimulating {max_steps} timesteps...")
print(f"Progress: ", end="", flush=True)

# Track statistics
position_history = {i: [] for i in range(NUM_SHIPS)}
current_history = []
wind_history = []

try:
    while current_time < end_time:
        step_count += 1
        
        # Progress indicator every 10%
        if step_count % (max_steps // 10) == 0:
            print(f"{int(100 * step_count / max_steps)}% ", end="", flush=True)
        
        # Get ship positions
        ship_lons = np.array([s['x'] for s in ships])
        ship_lats = np.array([s['y'] for s in ships])
        
        # Sample environment at ship positions
        currents = current_fn(ship_lons, ship_lats, current_time)
        winds = wind_fn(ship_lons, ship_lats, current_time)
        
        # Store statistics
        for i, ship in enumerate(ships):
            position_history[i].append((ship['x'], ship['y']))
        current_history.append(np.mean(np.sqrt(currents[:, 0]**2 + currents[:, 1]**2)))
        wind_history.append(np.mean(np.sqrt(winds[:, 0]**2 + winds[:, 1]**2)))
        
        # Update each ship
        for i, ship in enumerate(ships):
            u_curr, v_curr = currents[i]
            u_wind, v_wind = winds[i]
            
            # Simple dead reckoning with drift
            # (In full simulation, ship_model.py would handle dynamics)
            
            # Convert heading to radians
            heading_rad = np.deg2rad(ship['heading'])
            
            # Ship velocity in earth frame (5 m/s target speed)
            u_ship = ship['speed'] * np.sin(heading_rad)
            v_ship = ship['speed'] * np.cos(heading_rad)
            
            # Add current drift (full effect)
            u_total = u_ship + u_curr
            v_total = v_ship + v_curr
            
            # Add wind drift (reduced effect, ~5% for this test)
            u_total += 0.05 * u_wind
            v_total += 0.05 * v_wind
            
            # Update position (simple Euler integration)
            # Convert m/s to degrees (approximate at this latitude)
            meters_per_degree_lon = 111320 * np.cos(np.deg2rad(ship['y']))
            meters_per_degree_lat = 110540
            
            ship['x'] += (u_total / meters_per_degree_lon) * TIMESTEP
            ship['y'] += (v_total / meters_per_degree_lat) * TIMESTEP
            
            # Simple heading variation (±5 degrees per minute)
            ship['heading'] += np.random.uniform(-0.5, 0.5)
            ship['heading'] = ship['heading'] % 360
        
        # Advance time
        current_time += timedelta(seconds=TIMESTEP)

    print("100% ✓")
    
except KeyboardInterrupt:
    print("\n\n⚠ Simulation interrupted by user")
except Exception as e:
    print(f"\n\n✗ Simulation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Results
print(f"\n{'='*70}")
print("SIMULATION RESULTS")
print(f"{'='*70}")

print(f"\n✓ Simulation completed successfully!")
print(f"  Total steps: {step_count}")
print(f"  Simulated time: {(current_time - START_TIME).total_seconds() / 60:.1f} minutes")

print(f"\nEnvironmental Forcing Statistics:")
print(f"  Average current speed: {np.mean(current_history):.3f} m/s")
print(f"  Current speed range: [{np.min(current_history):.3f}, {np.max(current_history):.3f}] m/s")
print(f"  Average wind speed: {np.mean(wind_history):.3f} m/s")
print(f"  Wind speed range: [{np.min(wind_history):.3f}, {np.max(wind_history):.3f}] m/s")

print(f"\nShip Movement:")
for i, ship in enumerate(ships):
    history = position_history[i]
    start_pos = history[0]
    end_pos = history[-1]
    
    # Calculate distance traveled
    dx = (end_pos[0] - start_pos[0]) * 111320 * np.cos(np.deg2rad(start_pos[1]))
    dy = (end_pos[1] - start_pos[1]) * 110540
    distance = np.sqrt(dx**2 + dy**2)
    
    print(f"  Ship {i}:")
    print(f"    Start: ({start_pos[0]:.4f}, {start_pos[1]:.4f})")
    print(f"    End:   ({end_pos[0]:.4f}, {end_pos[1]:.4f})")
    print(f"    Distance: {distance:.1f} m ({distance/1852:.2f} nm)")
    print(f"    Final heading: {ship['heading']:.1f}°")

print(f"\n{'='*70}")
print("VALIDATION CHECKS")
print(f"{'='*70}")

# Validation
validation_passed = True

# Check 1: Ships moved
total_distance = sum([
    np.sqrt(
        ((position_history[i][-1][0] - position_history[i][0][0]) * 111320)**2 +
        ((position_history[i][-1][1] - position_history[i][0][1]) * 110540)**2
    )
    for i in range(NUM_SHIPS)
])
if total_distance > 100:  # At least 100m total movement
    print("✓ Ships moved during simulation")
else:
    print("✗ Ships didn't move enough")
    validation_passed = False

# Check 2: Environmental forcing non-zero
if np.mean(current_history) > 0.001:
    print("✓ Ocean currents active (non-zero)")
else:
    print("✗ Ocean currents appear to be zero")
    validation_passed = False

if np.mean(wind_history) > 0.001:
    print("✓ Winds active (non-zero)")
else:
    print("✗ Winds appear to be zero")
    validation_passed = False

# Check 3: Ships stayed in bounds
all_in_bounds = True
for i in range(NUM_SHIPS):
    for pos in position_history[i]:
        if not (bounds['minx'] <= pos[0] <= bounds['maxx'] and
                bounds['miny'] <= pos[1] <= bounds['maxy']):
            all_in_bounds = False
            break

if all_in_bounds:
    print("✓ All ships stayed within simulation bounds")
else:
    print("⚠ Some ships left simulation bounds (expected for open boundaries)")

# Check 4: No NaN values
if not any(np.isnan(current_history)) and not any(np.isnan(wind_history)):
    print("✓ No NaN values in environmental forcing")
else:
    print("✗ NaN values detected in environmental forcing")
    validation_passed = False

print(f"\n{'='*70}")
if validation_passed:
    print("🎉 ALL VALIDATION CHECKS PASSED! 🎉")
    print("\nGalveston simulation with real environmental forcing is working!")
    print("Ready to proceed with full-scale simulations.")
else:
    print("⚠ SOME VALIDATION CHECKS FAILED")
    print("\nReview the issues above before running full simulations.")

print(f"{'='*70}")

print("\n📝 Next Steps:")
print("  1. Review SHIP_ABM_TODO.md for remaining work")
print("  2. Fix Baltimore wind KDTree issue")
print("  3. Test remaining harbors (New Orleans should work like Galveston)")
print("  4. Integrate with full ship_model.py dynamics")
print("  5. Add visualization with vector field overlays")
