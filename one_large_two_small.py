from body import Body
import numpy as np


# This file contains the function equal_mass_orbit, which returns a list of three bodies in a stable circular orbit.
def one_large_two_small_orbit():
    """Three equal masses in a stable circular orbit."""
    mass = np.array([100.0, 1.0, 1.0])  # Mass of each body
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
        [0, angular_velocity[0] * radius, 0],
        [-angular_velocity[1] * radius * np.sqrt(3) / 2, -angular_velocity[1] * radius / 2, 0],
        [angular_velocity[2] * radius * np.sqrt(3) / 2, -angular_velocity[2] * radius / 2, 0]
    ]

    # Create and return the bodies
    bodies = [Body(position=positions[i], velocity=velocities[i], mass=mass[i]) for i in range(3)]
    return bodies
