import numpy as np
import matplotlib.pyplot as plt
from body import Body
from mpl_toolkits.mplot3d import Axes3D 

# Reduced softening parameter for accuracy
epsilon = np.float64(1e-10)


def two_body_config(mass1, mass2, distance):
    """Two-body configuration for Earth and Moon in a rotating frame."""
    # Scale mass1 and mass2
    total_mass = mass1 + mass2
    mass1 = mass1 / total_mass
    mass2 = mass2 / total_mass

    distance = distance / 384400000.0 # Scaled distance between mass1 and mass2, divided by the earth-moon distance

    # Correct angular velocity based on the total mass and distance
    omega = np.sqrt((mass1 + mass2) / distance**3)

    # Define Earth and Moon positions relative to the center of mass
    position1 = np.array([-mass2 / (mass1 + mass2), 0, 0], dtype=np.float64)
    position2 = np.array([mass1 / (mass1 + mass2), 0, 0], dtype=np.float64)

    # Angular velocity vector (rotating about the z-axis)
    omega_vector = np.array([0, 0, omega], dtype=np.float64)

    # Compute velocities using cross product
    velocity1 = np.cross(omega_vector, position1)
    velocity2 = np.cross(omega_vector, position2)

    # Define Earth and Moon as Body objects
    body1 = Body(position=position1, velocity=velocity1, mass=mass1)
    body2 = Body(position=position2, velocity=velocity2, mass=mass2)

    return [body1, body2]

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
    return np.linalg.norm(total_angular_momentum), total_angular_momentum

def symplectic_verlet(bodies, t_span, dt):
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
    velocities = [[] for _ in bodies] # Store velocities over time for all bodies

    step_count = 0  # Track the number of steps

    # Initial accelerations
    accelerations = compute_accelerations(bodies)

    # Initial energy for diagnostics
    initial_total_energy = compute_energy(bodies)
    # print(f"Initial Energy ->  Total E: {initial_total_energy:.6f}")

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
                velocities[i].append(body.velocity.copy())

            # Energy diagnostics every 50 steps
            if step_count % 50 == 0:
                current_total_energy = compute_energy(bodies)
                energy_drift = abs(current_total_energy - initial_total_energy) / abs(initial_total_energy)
                angular_momentum = compute_angular_momentum(bodies)
                # print(
                #     f"Time: {t:.3f}, "
                #     f"Total E: {current_total_energy:.6f}, "
                #     f"Energy Drift: {energy_drift:.6e}, "
                #     f"Angular Momentum: {angular_momentum:.6f}"
                # )

    except KeyboardInterrupt:
        print("\nSimulation interrupted. Finalizing...")
    finally:
        # print("Simulation stopped.")
        final_total_energy = compute_energy(bodies)
        # print(f"Final Energy -> Total E: {final_total_energy:.6f}")

    return [np.array(p,np.float64) for p in positions], [np.array(v,np.float64) for v in velocities]

# def pericenter(angular_momentum, relative_x, relative_v, total_mass): 
#     eccentricity = np.cross(relative_v, angular_momentum) / total_mass - relative_x / np.linalg.norm(relative_x)
#     eccent_norm = np.linalg.norm(eccentricity)
#     am_norm = np.linalg.norm(angular_momentum)
#     pericenter_norm = (am_norm**2/total_mass) / (1+eccent_norm)
#     return pericenter_norm * eccentricity / eccent_norm

def period_ellip(relative_x, relative_v):
    v = np.linalg.norm(relative_v)
    x = np.linalg.norm(relative_x)
    orbital_energy = v**2/2 - 1/x
    semi_major_axis = -1/(2*orbital_energy)
    return 2*np.pi*np.power(semi_major_axis, 3/2)

def period_circ(relative_x):
    x = np.linalg.norm(relative_x)
    semi_major_axis = x
    return 2*np.pi*np.power(semi_major_axis, 3/2)

def period_function(fixed_mass, mass_array, distance_array, option):
    # option: 'ellip' for elliptical orbit or "circ" for circular orbit
    body_num = mass_array.size
    distance_num = distance_array.size
    single_step_size = 1
    period_array = np.zeros((mass_array.size, distance_array.size))
    for i in range(body_num):
        for j in range(distance_num):
            total_mass = fixed_mass + mass_array[i]
            bodies = two_body_config(fixed_mass, mass_array[i], distance_array[j])
            pos, vel = symplectic_verlet(bodies, [0,single_step_size], single_step_size)
            x1 = pos[0][0]
            x2 = pos[1][0]
            v1 = vel[0][0]
            v2 = vel[1][0]
            relative_x = x1 - x2
            relative_v = v1 - v2
            angular_momentum_norm, angular_momentum = compute_angular_momentum(bodies)
            # pc = pericenter(angular_momentum, relative_x, relative_v, total_mass)
            if option == 'ellip':
                period = period_ellip(relative_x, relative_v)
                period_array[i,j] = period
            else:
                period = period_circ(relative_x)
                period_array[i,j] = period
    return period_array

fixed_mass = 5.97219e24     # mass of Earth (M0)
fixed_distance = 384400000.0   # Earth-Moon distance (R0)
T0 = np.sqrt(fixed_distance**3/(6.6743e-11*fixed_mass))
mass_array = np.linspace(0.5*fixed_mass, 2*fixed_mass, 20)
distance_array = np.linspace(0.5*fixed_distance, 2*fixed_distance, 20)

period_elliptical = period_function(fixed_mass, mass_array, distance_array, 'ellip')
period_circular = period_function(fixed_mass, mass_array, distance_array, 'circ')

X = np.tile(mass_array, (len(mass_array), 1))           
Y = np.tile(distance_array.reshape(-1, 1), (1, len(distance_array))) 
Z = period_elliptical
print(Z)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap='viridis')


ax.set_xlabel('Mass (1e25 kg)')
ax.set_ylabel('Distance between two bodies (1e8 m)')
ax.set_zlabel('Period (T0)')

plt.savefig('period3D(elliptical).png')
plt.close()

X = np.tile(mass_array, (len(mass_array), 1))           
Y = np.tile(distance_array.reshape(-1, 1), (1, len(distance_array))) 
Z = period_circular

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap='viridis')


ax.set_xlabel('Mass (1e25 kg)')
ax.set_ylabel('Distance between two bodies (1e8 m)')
ax.set_zlabel('Period (T0)')

plt.savefig('period3D(circular).png')