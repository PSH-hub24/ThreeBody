import numpy as np
from body import Body

def earth_moon_two_body():
    """Two-body configuration for Earth and Moon in a rotating frame."""
    # Constants (scaled units)
    earth_mass = 9.878e-1  # Scaled mass of Earth
    moon_mass = 1.22e-2    # Scaled mass of Moon

    earth_moon_distance = 1.0  # Scaled distance between Earth and Moon

    # Correct angular velocity based on the total mass and distance
    omega = np.sqrt((earth_mass + moon_mass) / earth_moon_distance**3)

    # Define Earth and Moon positions relative to the center of mass
    earth_position = np.array([-moon_mass / (earth_mass + moon_mass), 0, 0], dtype=np.float64)
    moon_position = np.array([earth_mass / (earth_mass + moon_mass), 0, 0], dtype=np.float64)

    # Angular velocity vector (rotating about the z-axis)
    omega_vector = np.array([0, 0, omega], dtype=np.float64)

    # Compute velocities using cross product
    earth_velocity = np.cross(omega_vector, earth_position)
    moon_velocity = np.cross(omega_vector, moon_position)

    # Define Earth and Moon as Body objects
    earth = Body(position=earth_position, velocity=earth_velocity, mass=earth_mass)
    moon = Body(position=moon_position, velocity=moon_velocity, mass=moon_mass)

    return [earth, moon]
