from body import Body
import numpy as np


# This file contains the function equal_mass_orbit, which returns a list of three bodies in a stable circular orbit.
def equal_mass_orbit():
    """Three equal masses in a stable circular orbit."""
    mass = 100.0  # Mass of each body
    radius = 10.0  # Distance from the center of mass to each body
    angular_velocity = np.sqrt(mass / (radius**3))  # Angular velocity for circular motion

    # Positions of the bodies (forming an equilateral triangle)
    positions = [
        [radius, 0, 0],
        [-0.5 * radius, np.sqrt(3) * radius / 2, 0],
        [-0.5 * radius, -np.sqrt(3) * radius / 2, 0]
    ]

    # Velocities of the bodies (perpendicular to the radius, for circular motion)
    velocities = [
        [0, angular_velocity * radius, 0],
        [-angular_velocity * radius * np.sqrt(3) / 2, -angular_velocity * radius / 2, 0],
        [angular_velocity * radius * np.sqrt(3) / 2, -angular_velocity * radius / 2, 0]
    ]

    # Create and return the bodies
    return [Body(position=positions[i], velocity=velocities[i], mass=mass) for i in range(3)]
