import numpy as np
from body import Body
def random_cluster_configuration(num_particles, central_mass):
    """Generate a random cluster of particles with positions and velocities."""
    particles = []
    for _ in range(num_particles):
        # Random radius within the cluster
        r_p = 2 * (np.random.random())**(1/3)  # Random radius scaled for distribution
        
        # Random angles
        c_p = -1 + 2 * np.random.random()  # Random cos(theta)
        phi_p = 2 * np.pi * np.random.random()  # Azimuthal angle

        # Calculate position
        x = r_p * np.sqrt(1 - c_p**2) * np.cos(phi_p)
        y = r_p * np.sqrt(1 - c_p**2) * np.sin(phi_p)
        z = r_p * c_p

        # Calculate velocity
        r_v = np.sqrt(x**2 + y**2 + z**2)
        # Random angles
        c_v = -1 + 2 * np.random.random()  # Random cos(theta)
        phi_v = 2 * np.pi * np.random.random()  # Azimuthal angle

        va = np.sqrt(central_mass / r_v) * np.random.random()**(1/3)  # Adjusted velocity magnitude
        vx = va * np.sqrt(1 - c_v**2) * np.cos(phi_v)
        vy = va * np.sqrt(1 - c_v**2) * np.sin(phi_v)
        vz = va * c_v

        # Assign mass (equal for all cluster particles)
        mass = 1.0

        # Create the particle as a Body object
        particle = Body(position=[x, y, z], velocity=[vx, vy, vz], mass=mass)
        particles.append(particle)

    return particles
