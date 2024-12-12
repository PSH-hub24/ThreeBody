import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
from broucke_orbit_config import broucke_orbit
from equal_mass_orbit_config import equal_mass_orbit
from figure_eight_orbit_config import figure_eight_orbit
from jupiter_ganymede_io_config import jupiter_ganymede_io
from lagrange_equilateral_config import lagrange_equilateral
from random_cluster_with_center_config import random_cluster_with_center
from solar_system_config import solar_system
from stable_orbit_test_config import stable_orbit_test
from sun_earth_moon_config import sun_earth_moon
from earth_moon_lagrange_config import earth_moon_lagrange
from earth_moon_lagrange_rotating_config import earth_moon_lagrange_rotating

# Reduced softening parameter for accuracy
epsilon = np.float64(1e-10)


def compute_accelerations(bodies):
    """
    Compute the gravitational accelerations acting on each body due to all other bodies.

    Parameters:
    bodies (list): List of Body objects representing the celestial bodies.

    Returns:
    np.ndarray: Array of accelerations for each body (shape: [n, 3]).
    """
    n = len(bodies)  # Number of bodies
    accelerations = np.zeros((n, 3), dtype=np.float64)  # Initialize accelerations
    positions = np.array([body.position for body in bodies], dtype=np.float64)  # Extract positions
    masses = np.array([body.mass for body in bodies], dtype=np.float64)  # Extract masses

    # Double loop to compute pairwise interactions
    for i in range(n - 1):
        for j in range(i + 1, n):
            displacement = positions[i] - positions[j]  # Vector from body j to body i
            distance_squared = np.dot(displacement, displacement) + epsilon**2  # Softened distance squared
            distance = np.sqrt(distance_squared)  # Compute distance
            w = 1 / (distance**3)  # Weight for inverse-square law
            da = -w * displacement  # Acceleration contribution
            accelerations[i] += da * masses[j]  # Acceleration on body i due to body j
            accelerations[j] -= da * masses[i]  # Acceleration on body j due to body i (Newton's 3rd law)

    return accelerations

def compute_energy(bodies):
    """
    Compute the total energy (kinetic + potential) of the system.

    Parameters:
    bodies (list): List of Body objects.

    Returns:
    float: Total energy of the system.
    """
    kinetic_energy = 0
    potential_energy = 0
    n = len(bodies)

    # Compute kinetic energy
    for body in bodies:
        kinetic_energy += 0.5 * body.mass * np.linalg.norm(body.velocity)**2

    # Compute potential energy
    for i in range(n - 1):
        for j in range(i + 1, n):
            r_ij = np.linalg.norm(bodies[i].position - bodies[j].position) + epsilon
            potential_energy -= bodies[i].mass * bodies[j].mass / r_ij

    return kinetic_energy + potential_energy

def compute_angular_momentum(bodies):
    """
    Compute the total angular momentum of the system.

    Parameters:
    bodies (list): List of Body objects.

    Returns:
    float: Magnitude of the total angular momentum.
    """
    total_angular_momentum = np.zeros(3, dtype=np.float64)
    for body in bodies:
        total_angular_momentum += np.cross(body.position, body.mass * body.velocity)
    return np.linalg.norm(total_angular_momentum)

def symplectic_verlet(bodies, t_span, dt, axes_sim):
    """
    Symplectic Verlet Integrator with real-time visualization and energy diagnostics.

    Parameters:
    bodies (list): List of Body objects.
    t_span (tuple): Time span for the simulation (start, end).
    dt (float): Time step for integration.
    axes_sim (list): Axes for real-time 2D projections (XY, YZ, XZ).

    Returns:
    list: Positions of all bodies over time (list of arrays).
    """
    t_start, t_end = t_span  # Unpack start and end times
    t = np.float64(t_start)  # Initialize current time
    dt = np.float64(dt)  # Convert time step to float64
    positions = [[] for _ in bodies]  # Store positions over time for all bodies

    step_count = 0  # Track the number of steps
    plt.ion()  # Enable interactive mode for real-time plotting

        # Initialize plot markers for real-time animation
    markers_xy = [axes_sim[0].plot([], [], 'o')[0] for _ in bodies]
    markers_yz = [axes_sim[1].plot([], [], 'o')[0] for _ in bodies]
    markers_xz = [axes_sim[2].plot([], [], 'o')[0] for _ in bodies]

    # Initial accelerations
    accelerations = compute_accelerations(bodies)

    # Initial energy for diagnostics
    initial_total_energy = compute_energy(bodies)
    print(f"Initial Energy ->  Total E: {initial_total_energy:.6f}")

    try:
        while t < t_end:
            # Increment time
            t += dt
            step_count += 1

            # Update position using current velocities and accelerations
            for i, body in enumerate(bodies):
                body.position += body.velocity * dt + 0.5 * accelerations[i] * dt**2

            # Compute new accelerations based on updated positions
            new_accelerations = compute_accelerations(bodies)

            # Update velocities using the average of old and new accelerations
            for i, body in enumerate(bodies):
                body.velocity += 0.5 * (accelerations[i] + new_accelerations[i]) * dt

            # Update accelerations for the next iteration
            accelerations = new_accelerations

            # Store positions for analysis
            for i, body in enumerate(bodies):
                positions[i].append(body.position.copy())

            # Energy diagnostics every 50 steps
            if step_count % 50 == 0:
                current_total_energy = compute_energy(bodies)
                energy_drift = abs(current_total_energy - initial_total_energy) / abs(initial_total_energy)
                angular_momentum = compute_angular_momentum(bodies)
                print(
                    f"Time: {t:.3f}, "
                    f"Total E: {current_total_energy:.6f}, "
                    f"Energy Drift: {energy_drift:.6e}, "
                    f"Angular Momentum: {angular_momentum:.6f}"
                )

            # Update real-time plots every 10 steps
            if  step_count % 10 == 0:
                for i, marker in enumerate(markers_xy):
                    marker.set_data(positions[i][-1][0], positions[i][-1][1])  # XY projection
                for i, marker in enumerate(markers_yz):
                    marker.set_data(positions[i][-1][1], positions[i][-1][2])  # YZ projection
                for i, marker in enumerate(markers_xz):
                    marker.set_data(positions[i][-1][0], positions[i][-1][2])  # XZ projection

                # Update titles for all axes
                for ax, proj in zip(axes_sim, ['XY', 'YZ', 'XZ']):
                    ax.set_title(f"Real-Time {proj} Projection | Time: {t:.3f}")
                plt.pause(0.0001)  # Pause to update the plots

    except KeyboardInterrupt:
        print("\nSimulation interrupted. Finalizing...")
    finally:
        print("Simulation stopped.")
        final_total_energy = compute_energy(bodies)
        print(f"Final Energy -> Total E: {final_total_energy:.6f}")

    return [np.array(p,np.float128) for p in positions]


# Prompt user for configuration choice
print("Choose a configuration:")
print("1. Sun-Earth-Moon")
print("2. Equal Mass Orbit")
print("3. Figure-Eight Orbit")
print("4. Broucke Orbit")
print("5. Lagrange Equilateral")
print("6. Jupiter-Ganymede-Io")
print("7. Solar System")
print("8. Random Cluster with Central Mass")
print("9. Earth-Moon Lagrange Points")
print("10. Earth-Moon Lagrange Points (Rotating Frame)")
print("11. Stability Test")
choice = int(input("Enter the number of your choice: "))
if choice == 1:
    config = sun_earth_moon()
elif choice == 2:
    config = equal_mass_orbit()
elif choice == 3:
    config = figure_eight_orbit()
elif choice == 4:
    config = broucke_orbit()
elif choice == 5:
    config = lagrange_equilateral()
elif choice == 6:
    config = jupiter_ganymede_io()
elif choice == 7:
    config = solar_system()
elif choice == 8:
    num_particles = int(input("Enter the number of bodies: "))
    config = random_cluster_with_center(num_particles=num_particles)
elif choice == 9:
    config = earth_moon_lagrange()
elif choice == 10:
    config = earth_moon_lagrange_rotating()
elif choice == 11:
    print("Running Stability Test...")
    # Get configuration from stable_orbit_test
    bodies, t_span, dt = stable_orbit_test()
    
    # Run the simulation with the provided bodies, time span, and time step
    # Real-time plotting axes for simulation
    fig_sim, axes_sim = plt.subplots(1, 3, figsize=(18, 6))
    projections = ['XY', 'YZ', 'XZ']

    # Set up plot limits and titles for each projection
    for ax, proj in zip(axes_sim, projections):
        ax.set_xlim(-50, 50)
        ax.set_ylim(-50, 50)
        ax.set_title(f"Real-Time {proj} Projection")
        ax.set_xlabel(proj[0])
        ax.set_ylabel(proj[1])
        ax.grid()

    positions = symplectic_verlet(bodies, t_span, dt, axes_sim)
else:
    raise ValueError("Invalid choice. Exiting...")
 

# Center of mass motion
total_mass = sum(body.mass for body in config)
total_momentum = sum(body.mass * body.velocity for body in config)
velocity_correction = total_momentum / total_mass
for body in config:
    body.velocity = body.velocity.astype(np.float64)  # Ensure float64 type
    body.velocity -= velocity_correction


# Prompt user for total simulation time and time step
t_total = np.float64(input("Enter total simulation time: "))
dt = np.float64(input("Enter time step: "))

# Set simulation span
t_span = (0, t_total)

# Real-time plotting axes for simulation
fig_sim, axes_sim = plt.subplots(1, 3, figsize=(18, 6))
projections = ['XY', 'YZ', 'XZ']

# Set up plot limits and titles for each projection
for ax, proj in zip(axes_sim, projections):
    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)
    ax.set_title(f"Real-Time {proj} Projection")
    ax.set_xlabel(proj[0])
    ax.set_ylabel(proj[1])
    ax.grid()

# Run simulation


positions = symplectic_verlet(config, t_span, dt, axes_sim)





# Post-simulation projections
fig_proj, axes_proj = plt.subplots(1, 3, figsize=(18, 6))
projections = ['XY', 'YZ', 'XZ']
for ax, proj in zip(axes_proj, projections):
    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)
    ax.set_title(f'{proj} Projection')
    ax.set_xlabel(proj[0])
    ax.set_ylabel(proj[1])
    ax.grid()

markers_xy = [axes_proj[0].plot([], [], 'o')[0] for _ in config]
markers_yz = [axes_proj[1].plot([], [], 'o')[0] for _ in config]
markers_xz = [axes_proj[2].plot([], [], 'o')[0] for _ in config]

def init():
    for markers in [markers_xy, markers_yz, markers_xz]:
        for marker in markers:
            marker.set_data([], [])
    return markers_xy + markers_yz + markers_xz

def update(frame):
    for i, marker in enumerate(markers_xy):
        marker.set_data(positions[i][frame, 0], positions[i][frame, 1])
    for i, marker in enumerate(markers_yz):
        marker.set_data(positions[i][frame, 1], positions[i][frame, 2])
    for i, marker in enumerate(markers_xz):
        marker.set_data(positions[i][frame, 0], positions[i][frame, 2])
    return markers_xy + markers_yz + markers_xz

frames = min(len(positions[0]), 500)
anim = FuncAnimation(fig_proj, update, frames=frames, init_func=init, blit=True)

plt.tight_layout()
plt.show()