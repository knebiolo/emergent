import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon
import matplotlib.transforms as transforms
from matplotlib.path import Path
import os

log_file_path = os.path.join(os.getcwd(), 'simulation_debug.log')

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler()  # Outputs to console as well
    ]
)
# Suppress Matplotlib font debug logs
logging.getLogger('matplotlib').setLevel(logging.WARNING)

log = logging.getLogger(__name__)  # Create the logger

class FossenShip:
    def __init__(self, ID, position, heading, speed, length, beam, draft, max_rudder_angle=60, goal_position=None):
        self.ID = ID
        self.current_heading = heading  # Initialize with a default or provided value
        self.default_heading = heading
        self.max_rudder_angle = max_rudder_angle  # Default to 30 degrees
        self.currentPos = np.array(position, dtype=float)  # Position in [x, y]
        self.psi = heading  # Heading (radians)
        self.nu = np.array([speed, 0.0, 0.0])  # Velocity in [surge, sway, yaw rate]
        self.max_speed = speed * 1.2
        self.length = length
        self.beam = beam
        self.draft = draft
        self.desired_speed = speed  # Set desired speed to the initial speed
        self.goal_position = np.array(goal_position, dtype=float) if goal_position is not None else np.array([0.0, 0.0])

        # Vessel mass and hydrodynamics
        self.mass = self.estimate_mass(length)  # Dynamic mass based on length
        self.added_mass = np.diag([2.0e3, 1.0e3, 1.0e3])  # Scale added mass appropriately
        self.damping = np.diag([0, 1500, 1500])  # Significantly reduce linear damping

        # Increase PID gains for faster heading corrections
        self.Kp = 9.0  # Strong proportional response
        self.Ki = 0.1  # Moderate integral response
        self.Kd = 2.0  # Strong derivative response
        self.integral = 0.0
        self.previous_error = 0.0

        # Propeller and drag model
        self.propeller_rpm = 1000  # Initial RPM
        self.max_rpm = 3000  # Increase maximum RPM
        self.rpm_to_thrust = 10.0  # Increase thrust per RPM
        self.drag_coefficient = 0.01  # Drag coefficient
        
        # COLREGS stuff
        self.role = "neutral"  # Default role for plotting
        
    def get_polygon(self):
        """
        Get the polygon representing the vessel shape.
        """
        # Define the vessel shape as a rectangle centered on (0, 0)
        half_length = self.length / 2
        half_beam = self.beam / 2
        local_polygon = np.array([
            [-half_length, -half_beam],
            [-half_length, half_beam],
            [half_length, half_beam],
            [half_length, -half_beam],
        ])

        # Rotate and translate to world coordinates
        rotation_matrix = np.array([
            [np.cos(self.psi), -np.sin(self.psi)],
            [np.sin(self.psi), np.cos(self.psi)]
        ])
        world_polygon = (rotation_matrix @ local_polygon.T).T + self.currentPos
        return world_polygon
    
        
    def estimate_mass(self, length):
        """
        Estimate vessel mass as a function of length.
        """
        k = 8.0  # Proportionality constant (kg/m^3) based on typical designs
        return k * length**2.2

    def calculate_thrust(self):
        """
        Compute thrust based on propeller RPM with efficiency modeling.
        """
    
        # Example: Efficiency decreases at very high RPMs
        efficiency = max(0.5, 1.0 - (self.propeller_rpm / self.max_rpm)**2)
        
        # Compute thrust, capping at a realistic maximum thrust
        max_thrust = 20000  # Example max thrust in N
        thrust_force = min(self.propeller_rpm * self.rpm_to_thrust * efficiency, max_thrust)
    
        # Ensure minimum thrust is maintained
        min_thrust = 500  # Minimum thrust (N)
        thrust_force = max(thrust_force, min_thrust)
    
        # Log for debugging
        log.debug(f"Vessel {self.ID}: Thrust = {thrust_force:.2f}")
        log.debug(f"Vessel {self.ID}: Efficiency = {efficiency:.2f}")
    
        return np.array([thrust_force, 0, 0])  # Thrust in surge direction

    def compute_drag(self):
        """
        Compute drag as a function of the square of surge speed with dynamic scaling.
        """
    
        speed = self.nu[0]  # Surge speed
        
        # Example: Increase drag coefficient at higher speeds (turbulent flow)
        base_drag_coefficient = self.drag_coefficient
        dynamic_drag_coefficient = base_drag_coefficient * (1 + 0.8 * speed)

    
        # Calculate drag force
        drag_force = base_drag_coefficient * speed**2
    
        # Log for debugging
        # log.debug(f"Vessel {self.ID}: Speed = {speed:.2f}")
        # log.debug(f"Vessel {self.ID}: Drag Coefficient = {dynamic_drag_coefficient:.3f}")
        # log.debug(f"Vessel {self.ID}: Drag = {drag_force:.2f}")
    
        # Drag opposes the direction of motion
        return np.array([-drag_force, 0, 0])

    def adjust_rpm(self, desired_speed, dt):
        """
        Adjust propeller RPM to achieve the desired speed using proportional control.
        """

        speed_error = desired_speed - self.nu[0]
        rpm_change = 100 * speed_error  # Increase gain for aggressive speed adjustments
        self.propeller_rpm = np.clip(self.propeller_rpm + rpm_change * dt, 0, self.max_rpm)

    def environmental_forces(self, wind, current):
        wind_force = 0.5 * self.drag_coefficient * wind['speed']**2 * np.array([
            np.cos(wind['direction']),
            np.sin(wind['direction']),
            0
        ])
        current_force = 0.5 * self.drag_coefficient * current['speed']**2 * np.array([
            np.cos(current['direction']),
            np.sin(current['direction']),
            0
        ])
        return wind_force + current_force

    def coriolis_matrix(self):
        """
        Compute the Coriolis and centripetal matrix for the vessel.
        """
        u, v, r = self.nu
        return np.array([
            [0, 0, -self.mass * v],
            [0, 0, self.mass * u],
            [self.mass * v, -self.mass * u, 0]
        ])

    def quadratic_damping(self):
        return np.array([
            0,  # 0.9 * abs(self.nu[1]) * self.nu[1] Surge modeled with drag (keep as is)
            500 * abs(self.nu[1]) * self.nu[1],  #sway damping
            1.0e6 * abs(self.nu[2]) * self.nu[2],  #yaw damping
        ])
    
    def compute_rudder_torque(self, rudder_angle):
        """
        Compute rudder torque as a function of rudder angle, vessel length, and surge velocity.
        """
        surge_velocity = self.nu[0]  # Surge velocity (forward speed)
        
        # Scale rudder torque by rudder angle, vessel length, and surge velocity
        rudder_torque = rudder_angle * self.length * max(surge_velocity, 0)
        
        return rudder_torque
    
    def compute_attractive_force(self):
        """
        Compute the attractive force directing the vessel towards its goal position.
        """
        direction_to_goal = self.goal_position - self.currentPos
        unit_vector = direction_to_goal / np.linalg.norm(direction_to_goal)  # Normalize to unit vector
        attractive_force = 500 * unit_vector  # Scale the force (adjust 500 as needed)
    
        return attractive_force


    def pid_control(self, desired_heading, dt):
        """
        PID controller for rudder angle.
        """
        # Compute heading error (wrap to -π to π)
        error = (desired_heading - self.psi + np.pi) % (2 * np.pi) - np.pi

        # PID calculations
        self.integral += error * dt
        derivative_error = (error - self.previous_error) / dt

        # Compute rudder command
        rudder_angle = (
            self.Kp * error +
            self.Ki * self.integral +
            self.Kd * derivative_error
        )

        # Clamp rudder angle to maximum
        rudder_angle = np.clip(rudder_angle, -self.max_rudder_angle, self.max_rudder_angle)

        self.rudder_angle = rudder_angle
        # Save error for next iteration
        self.previous_error = error

        # Debug logging
        # log.debug( f"Vessel {self.ID}: PID Error = {error:.2f}")
        # log.debug(f"Vessel {self.ID}: Integral = {self.integral:.2f}")
        # log.debug(f"Vessel {self.ID}: Derivative = {derivative_error:.2f}")
        # log.debug(f"Vessel {self.ID}: Rudder Command = {rudder_angle:.2f}")
        # log.debug(f"Vessel {self.ID}: Current Heading = {np.degrees(self.psi):.1f}°")
        # log.debug(f"Vessel {self.ID}: Desired Heading = {np.degrees(desired_heading):.1f}°")
        # log.debug(f"Vessel {self.ID}: Rudder = {rudder_angle:.2f}°")

        return rudder_angle

    def colregs(self, other_vessels):
        """
        Determine avoidance maneuvers based on COLREGS and introduce a stricter crossing logic with a lock.
        """
        safe_distance = 2000  # Increased safe distance
        clearance_distance = 1000  # Distance required to consider crossing complete
        avoidance_heading = self.psi
        desired_speed = self.desired_speed
        role = self.role  # Keep the current role unless explicitly changed
        crossing_lock = getattr(self, "crossing_lock", None)
    
        for other in other_vessels:
            relative_position = other.currentPos - self.currentPos
            distance = np.linalg.norm(relative_position)
            if distance > safe_distance:
                continue
    
            angle_to_other = np.arctan2(relative_position[1], relative_position[0])
            relative_bearing = (angle_to_other - self.psi + np.pi) % (2 * np.pi) - np.pi

            if crossing_lock:  # Maintain give-way behavior until safely cleared
                # Ensure the give-way vessel stays out of the crossing situation
                relative_velocity = self.nu[0] * np.array([np.cos(self.psi), np.sin(self.psi)]) - \
                                    other.nu[0] * np.array([np.cos(other.psi), np.sin(other.psi)])
                future_position = self.currentPos + relative_velocity * 5  # Predict future position
                distance_to_other = np.linalg.norm(other.currentPos - future_position)
            
                if distance_to_other < clearance_distance and 0 < relative_bearing < np.pi / 2:
                    role = "give_way"
                    avoidance_heading = (self.psi + np.radians(-120)) % (2 * np.pi)
                    desired_speed = max(self.desired_speed * 0.3, 1.0)  # Slow down more aggressively
                else:
                    # Clear the crossing lock when the clearance distance is satisfied
                    self.crossing_lock = False
                    role = "neutral"  # Return to neutral role when the crossing is resolved
    
            elif 0 < relative_bearing < np.pi / 2:  # Entering a crossing situation
                role = "give_way"
                avoidance_heading = (self.psi + np.radians(-120)) % (2 * np.pi)
                # Adjust speed more aggressively based on distance
                slowdown_factor = max(0.3, min(1.0, distance / safe_distance))  # Scale slowdown by distance
                desired_speed = max(self.desired_speed * slowdown_factor, 1.0)  # Minimum speed is 1.0
                self.crossing_lock = True  # Activate the crossing lock

            elif abs(relative_bearing) < np.radians(10):  # Head-on
                role = "give_way"
                avoidance_heading = (self.psi + np.radians(30)) % (2 * np.pi)
                desired_speed = max(self.desired_speed * 0.7, 5.0)
    
            elif -np.pi / 8 < relative_bearing < np.pi / 8:  # Overtaking
                role = "give_way"
                avoidance_heading = (self.psi + np.radians(10)) % (2 * np.pi)
                desired_speed = min(self.desired_speed * 1.2, self.max_speed)
    
            else:  # Stand-on vessel
                role = "stand_on"
    
        # If neutral or stand-on, compute the attractive force towards the goal
        if role in ["neutral", "stand_on"]:
            attractive_force = self.compute_attractive_force()
            avoidance_heading = np.arctan2(attractive_force[1], attractive_force[0])
            desired_speed = self.desired_speed
    
        self.role = role
        return avoidance_heading, desired_speed, role

    def reset_behavior(self):
        """
        Reset vessel behavior when its role changes.
        """
        self.current_speed = self.desired_speed  # Reset to default cruising speed
        self.current_heading = self.default_heading  # Reset to default heading
        log.debug(f"[Vessel {self.ID}] Behavior reset to default values.")

    def update(self, tau, wind, current, dt, desired_speed):
        """
        Update vessel dynamics using the 3DOF equations of motion.
        """
        # Adjust RPM to match desired speed
        self.adjust_rpm(desired_speed, dt)
    
        # Compute forces
        thrust = self.calculate_thrust()
        drag = self.compute_drag()
        damping_force = np.dot(self.damping, self.nu) + self.quadratic_damping()
        environmental_force = self.environmental_forces(wind, current)
        C = self.coriolis_matrix()
        # Total force
        net_force = tau + drag + environmental_force - np.dot(C, self.nu) - damping_force
    
        # Compute acceleration
        M = self.mass * np.eye(3) + self.added_mass
        acceleration = np.linalg.inv(M).dot(net_force)
        
        # Update velocities
        self.nu[2] = np.clip(self.nu[2], -np.radians(10), np.radians(10))  # Limit to ±10°/s

        self.nu += acceleration * dt
        u, v, r = self.nu  # r is the yaw rate
    
        # Update heading and position
        self.psi += r * dt
        self.psi = self.psi % (2 * np.pi)  # Normalize to [0, 2*pi)
        self.currentPos += np.array([
            u * np.cos(self.psi) - v * np.sin(self.psi),
            u * np.sin(self.psi) + v * np.cos(self.psi)
        ]) * dt
    
        # Log final state
        log.debug(f"Vessel {self.ID}: Thrust = {thrust[0]}")
        log.debug(f"Vessel {self.ID}: Drag = {drag[0]}")
        log.debug(f"Vessel {self.ID}: Damping Force = {damping_force}")
        log.debug(f"Vessel {self.ID}: Environmental Force = {environmental_force}")
        log.debug(f"Vessel {self.ID}: Coriolis Forces = {np.dot(C, self.nu)}")
        log.debug(f"Vessel {self.ID}: Linear Damping = {np.dot(self.damping, self.nu)}") 
        log.debug(f"Vessel {self.ID}: Quadratic Damping = {self.quadratic_damping()}")
        log.debug(f"Vessel {self.ID}: Net Force = {net_force}")
        log.debug(f"Vessel {self.ID}: Surge = {u}")
        log.debug(f"Vessel {self.ID}: Sway = {v}")
        log.debug(f"Vessel {self.ID}: Yaw = {r}")
        log.debug(f"Vessel {self.ID}: Acceleration = {acceleration}")
        log.debug(f"Vessel {self.ID}: Speed = {self.nu[0]:.2f}")
        log.debug(f"Vessel {self.ID}: Position = {self.currentPos}")

            
class Simulation:
    def __init__(self, vessels, wind, current, dt=0.1, simulation_time=1000):
        """
        Initialize the simulation environment.
        :param vessels: List of FossenShip objects.
        :param wind: Dictionary with 'speed' and 'direction' (radians).
        :param current: Dictionary with 'speed' and 'direction' (radians).
        :param dt: Time step (s).
        :param simulation_time: Total simulation time (s).
        """
        if not vessels or not isinstance(vessels, list):
            raise ValueError("Vessels must be a non-empty list of FossenShip objects.")
        self.vessels = vessels
        self.wind = wind
        self.current = current
        self.dt = dt
        self.time = np.arange(0, simulation_time, dt)

        # Visualization setup
        self.text_labels = {}
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        # self.ax.set_xlim(-400, 400)  # Adjust based on simulation area
        # self.ax.set_ylim(-400, 400)
        self.ax.set_aspect('equal', adjustable='datalim')  # Ensure 1:1 scale for x and y axes
        self.ax.set_title("Real-Time Ship Traffic Simulation")
        self.ax.set_xlabel("X Position")
        self.ax.set_ylabel("Y Position")
        self.ax.grid()


        # Plot objects for vessels
        self.plots = {}
        self.trajectories = {}
        
        for vessel in self.vessels:
            # Calculate vertices
            vertices = [
                (-vessel.length / 2, -vessel.beam / 2),  # Stern left corner
                (vessel.length / 4, -vessel.beam / 2),  # Bow base left corner
                (vessel.length / 2, 0),                 # Bow tip (triangle point)
                (vessel.length / 4, vessel.beam / 2),   # Bow base right corner
                (-vessel.length / 2, vessel.beam / 2)   # Stern right corner
            ]
            # Debug vertices
            log.debug(f"Vessel {vessel.ID} vertices: {vertices}")
        
            # Add polygon to the plot for vessel shape
            self.plots[vessel.ID] = self.ax.add_patch(Polygon(
                xy=vertices,
                closed=True,
                color='blue'
            ))
        
            # Add a separate line for the trajectory
            self.trajectories[vessel.ID] = self.ax.plot([], [], linestyle='--', color='blue')[0]

        # # Initialize plots with initial positions
        # for vessel in vessels:
        #     self.plots[vessel.ID].set_data(
        #         list(vessel.currentPos[0:1]), list(vessel.currentPos[1:2])
        #     )
        
    def check_collision(self, vessel1, vessel2):
        """
        Check if two vessels collide based on their polygons.
        """
    
        # Get polygons in world coordinates
        polygon1 = vessel1.get_polygon()
        polygon2 = vessel2.get_polygon()
    
        # Check for intersection
        path1 = Path(polygon1)
        path2 = Path(polygon2)
        return path1.intersects_path(path2)

    def _update_vessels(self):
        for vessel in self.vessels:
            other_vessels = [v for v in self.vessels if v.ID != vessel.ID]
            
            # Check for collisions
            for other in other_vessels:
                if self.check_collision(vessel, other):
                    log.error(f"Collision detected between Vessel {vessel.ID} and Vessel {other.ID}!")
                    self.stop_simulation = True
                    return
    
            # Determine avoidance maneuvers and role using COLREGS
            avoidance_heading, desired_speed, new_role = vessel.colregs(other_vessels)
            
            # Update role if it changes
            if vessel.role != new_role:
                vessel.role = new_role
                log.debug(f"Vessel {vessel.ID} changed role to {vessel.role}")
            
            # Update vessel state
            rudder_angle = vessel.pid_control(avoidance_heading, self.dt)
            tau = np.array([0, 0, rudder_angle])
            vessel.update(tau, self.wind, self.current, self.dt, desired_speed)

    def _update_plot(self, frame):
        self._update_vessels()  # Update vessel states
    
        # Dynamically adjust the axes
        all_positions = np.array([vessel.currentPos for vessel in self.vessels])
        x_min, x_max = all_positions[:, 0].min(), all_positions[:, 0].max()
        y_min, y_max = all_positions[:, 1].min(), all_positions[:, 1].max()
        buffer = max(100, 0.2 * max(x_max - x_min, y_max - y_min))
        self.ax.set_xlim(x_min - buffer, x_max + buffer)
        self.ax.set_ylim(y_min - buffer, y_max + buffer)
        self.ax.set_aspect('equal', adjustable='box')
    
        for vessel in self.vessels:
            # Update color based on role
            color = {'neutral': 'blue', 'give_way': 'red', 'stand_on': 'green'}.get(vessel.role, 'blue')
            self.plots[vessel.ID].set_facecolor(color)
        
            # Define the detailed shape of the ship (e.g., triangular bow)
            base_shape = np.array([
                (-vessel.length / 2, -vessel.beam / 2),  # Stern left corner
                (vessel.length / 4, -vessel.beam / 2),  # Bow base left corner
                (vessel.length / 2, 0),                 # Bow tip (triangle point)
                (vessel.length / 4, vessel.beam / 2),   # Bow base right corner
                (-vessel.length / 2, vessel.beam / 2)   # Stern right corner
            ])
        
            # Rotate and translate the shape
            rotation_matrix = np.array([
                [np.cos(vessel.psi), -np.sin(vessel.psi)],
                [np.sin(vessel.psi),  np.cos(vessel.psi)]
            ])
            rotated_shape = base_shape @ rotation_matrix.T  # Apply rotation
            translated_shape = rotated_shape + vessel.currentPos  # Apply translation
            self.plots[vessel.ID].set_xy(translated_shape)  # Update the polygon's vertices
        
            # Update trajectory
            if not hasattr(vessel, 'trajectory'):
                vessel.trajectory = []  # Initialize trajectory if not present
            vessel.trajectory.append(vessel.currentPos.copy())  # Track position
            trajectory = np.array(vessel.trajectory)
            self.trajectories[vessel.ID].set_data(trajectory[:, 0], trajectory[:, 1])  # Update line plot
        
            # Update label directly instead of appending to an array
            if vessel.ID not in self.text_labels:
                # Create label only once
                self.text_labels[vessel.ID] = self.ax.text(
                    vessel.currentPos[0] + 20, vessel.currentPos[1] + 10, "", fontsize=6, color='black', ha='left'
                )
            # Update label text and position
            label_text = (
                f"ID: {vessel.ID}\n"
                f"Heading: {np.degrees(vessel.psi):.1f}°\n"
                f"Speed: {vessel.nu[0]:.2f}"
            )
            self.text_labels[vessel.ID].set_position((vessel.currentPos[0] + 20, vessel.currentPos[1] + 10))
            self.text_labels[vessel.ID].set_text(label_text)
    
        plt.pause(0.001)

    def get_current_role(self, vessel_id):
        """
        Get the current role of the vessel based on the most recent trajectory data.
        """
        for role in ['give_way', 'stand_on', 'neutral']:  # Priority order for roles
            if len(self.trajectories[vessel_id][role]) > 0:
                return role
        return 'neutral'  # Default to neutral if no data is available

    def run(self):
        """
        Main simulation loop to update vessels and animate their positions.
        """
        frame = 0
        self.stop_simulation = False  # Initialize collision flag

        while frame < len(self.time):
            for vessel in self.vessels:
                
                other_vessels = [v for v in self.vessels if v.ID != vessel.ID]
    
                # Apply COLREGS for avoidance heading, speed, and role
                avoidance_heading, desired_speed, vessel_role = vessel.colregs(other_vessels)
                vessel.role = vessel_role  # Update role for plotting
    
                # Apply PID control for heading and RPM adjustment for speed
                rudder_angle = vessel.pid_control(avoidance_heading, self.dt)
                vessel.adjust_rpm(desired_speed, self.dt)
    
                # Use rudder angle to compute yaw torque (tau[2]) and update state
                rudder_torque = vessel.compute_rudder_torque(rudder_angle)

                tau = np.array([0, 0, rudder_torque])  # Apply yaw torque
    
                # Update vessel state with yaw torque included
                vessel.update(tau, self.wind, self.current, self.dt, desired_speed)
            
            if self.stop_simulation:  # Exit the loop if a collision is detected
                log.error("Simulation stopped due to collision.")
                break

            # Update the plot
            self._update_plot(frame)
            frame += 1


        self._update_plot(frame)
        frame += 1

# Initialize vessels
ship1 = FossenShip(ID=1, position=[-1000, -1000], heading=np.radians(45), speed=10, length=50, beam=15, draft=5, goal_position=[100000, 100000])
ship2 = FossenShip(ID=2, position=[1000, -1000], heading=np.radians(135), speed=10, length=50, beam=15, draft=5, goal_position=[-100000, 100000])
vessels = [ship1, ship2]

# Environmental forces
wind = {'speed': 0.000001, 'direction': np.pi / 3}  # Example wind
current = {'speed': 0.000001, 'direction': np.pi / 4}  # Example current

# Create and run the simulation
sim = Simulation(vessels, wind, current, dt=0.1, simulation_time=100)
sim.run()

