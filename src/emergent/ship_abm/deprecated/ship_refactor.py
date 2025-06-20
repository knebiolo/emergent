import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class FossenShip:
    def __init__(self, ID, position, heading, speed, length, beam, draft):
        self.ID = ID
        self.currentPos = np.array(position, dtype=float)  # Position in [x, y]
        self.psi = heading  # Heading (radians)
        self.nu = np.array([speed, 0.0, 0.0])  # Velocity in [surge, sway, yaw rate]
        self.length = length
        self.beam = beam
        self.draft = draft
        self.desired_speed = speed  # Set desired speed to the initial speed

        # Vessel mass and hydrodynamics
        self.mass = 1.0e4  # Mass (kg)
        self.added_mass = np.diag([5.0e3, 1.0e3, 2.0e3])  # Added mass (kg)
        self.damping = np.diag([500, 500, 100])  # Significantly reduce linear damping
        # Increase rudder angle limit for more aggressive turns
        
        # Increase PID gains for faster heading corrections
        self.Kp = 8.0  # Strong proportional response
        self.Ki = 0.5  # Moderate integral response
        self.Kd = 2.0  # Strong derivative response
        #self.integral_error = 0.0
        #self.previous_heading_error = 0.0
        self.integral = 0.0
        self.previous_error = 0.0

        # Propeller and drag model
        self.propeller_rpm = 1000  # Initial RPM
        self.max_rpm = 3000  # Increase maximum RPM
        self.rpm_to_thrust = 10.0  # Increase thrust per RPM
        self.drag_coefficient = 0.05  # Drag coefficient

    def calculate_thrust(self):
        """
        Compute thrust based on propeller RPM.
        """

        min_thrust = 500  # Minimum thrust (N)
        thrust_force = max(self.propeller_rpm * self.rpm_to_thrust, min_thrust)
        print(f"Vessel {self.ID}: Thrust = {thrust_force:.2f}")
        return np.array([thrust_force, 0, 0])  # Thrust in surge direction

    def compute_drag(self):
        """
        Compute drag as a function of the square of surge speed.
        """
        speed = self.nu[0]  # Surge speed
        drag_force = self.drag_coefficient * speed**2
        return np.array([-drag_force, 0, 0])  # Drag opposes surge direction

    def adjust_rpm(self, desired_speed, dt):
        """
        Adjust propeller RPM to achieve the desired speed using proportional control.
        """
        speed_error = desired_speed - self.nu[0]
        rpm_change = 50 * speed_error  # Increase gain for faster RPM adjustments
        self.propeller_rpm = np.clip(self.propeller_rpm + rpm_change * dt, 0, self.max_rpm)
        print(f"Vessel {self.ID}: Desired Speed = {desired_speed:.2f}, Current Speed = {self.nu[0]:.2f}, "
              f"RPM = {self.propeller_rpm:.2f}")


    def environmental_forces(self, wind, current):
        """
        Placeholder for environmental forces (wind and current).
        """
        return np.array([0, 0, 0])  # Zero environmental forces for now

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
        """
        Compute reduced quadratic damping forces.
        """
        return np.array([
            50 * abs(self.nu[0]) * self.nu[0],  # Surge damping
            50 * abs(self.nu[1]) * self.nu[1],  # Sway damping
            50 * abs(self.nu[2]) * self.nu[2]   # Yaw damping
        ])

    def pid_control(self, desired_heading, dt):
        """
        PID controller for heading control.
        """
        Kp = 1.0  # Proportional gain
        Ki = 0.01  # Integral gain
        Kd = 0.1  # Derivative gain
    
        # Calculate the heading error
        heading_error = desired_heading - self.psi
        heading_error = (heading_error + np.pi) % (2 * np.pi) - np.pi  # Normalize to [-pi, pi]
    
        # Proportional term
        proportional = Kp * heading_error
    
        # Integral term
        self.integral += heading_error * dt
        integral = Ki * self.integral
    
        # Derivative term
        derivative = Kd * (heading_error - self.previous_error) / dt
        self.previous_error = heading_error
    
        # Calculate rudder angle (output of PID)
        rudder_angle = proportional + integral + derivative
        rudder_angle = np.clip(rudder_angle, -np.radians(30), np.radians(30))  # Limit to ±30°
    
        return rudder_angle


    def colregs(self, vessel, other_vessels):
        """
        Enhanced COLREGS logic for maritime navigation.
        - Overtaking: The overtaking vessel must keep clear.
        - Crossing: A vessel must yield to traffic on its starboard side.
        - Head-On: Vessels should pass port-to-port.
        """
        safe_distance = 100  # Safety distance threshold
        avoidance_heading = vessel.psi  # Default to current heading
        desired_speed = vessel.desired_speed  # Default to maintain current speed
        priority = 0  # Priority of the scenario: 1 = overtaking, 2 = crossing, 3 = head-on
    
        for other in other_vessels:
            # Calculate relative position and distance
            relative_position = other.currentPos - vessel.currentPos
            distance = np.linalg.norm(relative_position)
            if distance > safe_distance:
                continue  # Skip distant vessels
    
            # Relative velocity and closing speed
            relative_velocity = other.nu[:2] - vessel.nu[:2]
            closing_speed = np.dot(relative_position, relative_velocity) / distance
    
            # Angle to the other vessel
            angle_to_other = np.arctan2(relative_position[1], relative_position[0])
            angle_to_other = (angle_to_other + 2 * np.pi) % (2 * np.pi)
            vessel_heading = (vessel.psi + 2 * np.pi) % (2 * np.pi)
    
            # Relative bearing (-pi to pi)
            relative_bearing = (angle_to_other - vessel_heading + np.pi) % (2 * np.pi) - np.pi
    
            # Determine COLREGS scenario
            if closing_speed < 0:  # Approaching
                # Overtaking scenario (±22.5° astern)
                if -np.pi / 8 < relative_bearing < np.pi / 8 and priority < 1:
                    avoidance_heading = vessel_heading + np.radians(30)  # Turn away from the target
                    desired_speed = max(vessel.desired_speed * 0.8, 5.0)  # Reduce speed slightly
                    priority = 1
    
                # Crossing scenario (target on starboard side)
                elif 0 < relative_bearing < np.pi / 2 and priority < 2:
                    avoidance_heading = vessel_heading + np.radians(-90)  # Turn sharply port
                    desired_speed = max(vessel.desired_speed * 0.7, 5.0)  # Reduce speed more
                    priority = 2
    
                # Head-on scenario (within ±10° ahead)
                elif abs(relative_bearing) < np.radians(10) and priority < 3:
                    avoidance_heading = vessel_heading + np.radians(20)  # Turn starboard
                    desired_speed = max(vessel.desired_speed * 0.6, 5.0)  # Slow significantly
                    priority = 3
    
        return avoidance_heading, desired_speed


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

        # Total force
        M = self.mass * np.eye(3) + self.added_mass
        C = self.coriolis_matrix()
        net_force = tau + thrust + drag + environmental_force - np.dot(C, self.nu) - damping_force
        acceleration = np.linalg.inv(M).dot(net_force)

        print(f"Vessel {self.ID}: Thrust = {thrust[0]:.2f}, Drag = {drag[0]:.2f}, "
              f"Damping = {damping_force[0]:.2f}, Net Force = {net_force[0]:.2f}, "
              f"Acceleration = {acceleration[0]:.2f}")

        # Update velocities
        self.nu += acceleration * dt
        u, v, r = self.nu

        # Update heading and position
        self.psi += r * dt
        self.psi = self.psi % (2 * np.pi)
        self.currentPos += np.array([
            u * np.cos(self.psi) - v * np.sin(self.psi),
            u * np.sin(self.psi) + v * np.cos(self.psi)
        ]) * dt

        # Debug position and speed
        print(f"Vessel {self.ID}: Position = {self.currentPos}, Speed = {self.nu[0]:.2f}, "
              f"Heading = {np.degrees(self.psi):.2f}")





class Simulation:
    def __init__(self, vessels, wind, current, dt=0.1, simulation_time=100):
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
        # Correctly initialize trajectories as lists
        self.trajectories = {vessel.ID: [] for vessel in vessels}

        print(f"Trajectories Initialized: {self.trajectories}")
        # Visualization setup
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.ax.set_xlim(-400, 400)  # Adjust based on simulation area
        self.ax.set_ylim(-400, 400)
        self.ax.set_title("Real-Time Ship Traffic Simulation")
        self.ax.set_xlabel("X Position")
        self.ax.set_ylabel("Y Position")
        self.ax.grid()

        # Plot objects for vessels
        self.plots = {
            vessel.ID: self.ax.plot([], [], marker='o', label=f"Vessel {vessel.ID}")[0]
            for vessel in vessels
        }
        self.trajectories_lines = {
            vessel.ID: self.ax.plot([], [], '--', alpha=0.5)[0] for vessel in vessels
        }

        # Add legend
        self.ax.legend()

        # Initialize plots with initial positions
        for vessel in vessels:
            self.plots[vessel.ID].set_data(
                list(vessel.currentPos[0:1]), list(vessel.currentPos[1:2])
            )
            
    def _update_vessels(self):
        for vessel in self.vessels:
            other_vessels = [v for v in self.vessels if v.ID != vessel.ID]
            desired_heading, desired_speed = vessel.colregs(vessel, other_vessels)
    
            tau = np.array([vessel.pid_control(desired_heading, self.dt), 0, 0])
            vessel.update(tau, self.wind, self.current, self.dt, desired_speed)
            self.trajectories[vessel.ID].append(vessel.currentPos.copy())


                
    def _update_plot(self, frame):
        print(f"Frame {frame} updating...")
        self._update_vessels()  # Update all vessels' states
    
        for vessel in self.vessels:
            print(f"Frame {frame}, Vessel {vessel.ID} Position: {vessel.currentPos}")
            self.plots[vessel.ID].set_data(
                [vessel.currentPos[0]], [vessel.currentPos[1]]
            )
    
            # Create a copy for NumPy operations
            traj = np.array(self.trajectories[vessel.ID][:]) if len(self.trajectories[vessel.ID]) > 0 else np.empty((0, 2))
            if len(traj) > 0:
                self.trajectories_lines[vessel.ID].set_data(traj[:, 0].tolist(), traj[:, 1].tolist())
    
        plt.pause(0.001)  # Force a short pause to refresh the plot
    
        return list(self.plots.values()) + list(self.trajectories_lines.values())
    
    def run(self):
        """
        Main simulation loop to update vessels and animate their positions.
        """
        frame = 0
        while True:
            for vessel in self.vessels:
                other_vessels = [v for v in self.vessels if v.ID != vessel.ID]
    
                # Apply COLREGS for avoidance heading and speed
                avoidance_heading, desired_speed = vessel.colregs(vessel, other_vessels)
    
                # Apply PID control for heading and RPM adjustment for speed
                rudder_angle = vessel.pid_control(avoidance_heading, self.dt)
                vessel.adjust_rpm(desired_speed, self.dt)
    
                # Update vessel state
                tau = np.zeros(3)  # No external forces for now
                wind = self.wind  # Placeholder for wind dictionary
                current = self.current  # Placeholder for current dictionary
                vessel.update(tau, wind, current, self.dt, desired_speed)
    
                # Debug output
                print(f"Vessel {vessel.ID}: Position = {vessel.currentPos}, "
                      f"Speed = {vessel.nu[0]:.2f}, Heading = {np.degrees(vessel.psi):.2f}")
    
            # Update the plot
            self._update_plot(frame)
    
            frame += 1
            if frame > 500:  # Limit frames for testing
                break



    def get_trajectories(self):
        """
        Retrieve vessel trajectories as arrays for analysis or testing.
        """
        result = {}
        for key in self.trajectories:
            result[key] = np.array(self.trajectories[key])  # Copy to NumPy array for external use
            print(f"Vessel {key} trajectory: {result[key]}")  # Debug output
        return result



# Initialize vessels
ship1 = FossenShip(ID=1, position=[-50, -50], heading=np.radians(45), speed=10, length=50, beam=15, draft=5)
ship2 = FossenShip(ID=2, position=[250, -50], heading=np.radians(135), speed=10, length=50, beam=15, draft=5)
vessels = [ship1, ship2]

# Environmental forces
wind = {'speed': 0.00002, 'direction': np.pi / 3}  # Example wind
current = {'speed': 0.00001, 'direction': np.pi / 4}  # Example current

# Create and run the simulation
sim = Simulation(vessels, wind, current, dt=0.1, simulation_time=10)
sim.run()

# Optionally, retrieve and analyze trajectories after the simulation
trajectories = sim.get_trajectories()
