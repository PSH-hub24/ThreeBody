import numpy as np
from body import Body

def lagrange_equilateral():
    """
    Create a stable Lagrange equilateral triangle configuration with three equal masses.

    Returns:
    list: A list of Body objects representing the configuration.
    """
    mass = 100.0  # Mass of each body

    # Define the positions of the three bodies in an equilateral triangle
    positions = [
        [1.0, 0.0, 0.0],                          # Body 1
        [-0.5, np.sqrt(3) / 2, 0.0],              # Body 2
        [-0.5, -np.sqrt(3) / 2, 0.0]              # Body 3
    ]

    # Define the velocities to maintain the configuration
    velocities = [
        [0.0, 0.5, 0.0],                          # Velocity of Body 1
        [-np.sqrt(3) / 2, -0.25, 0.0],            # Velocity of Body 2
        [np.sqrt(3) / 2, -0.25, 0.0]              # Velocity of Body 3
    ]

    # Create Body objects for the configuration
    return [Body(position=positions[i], velocity=velocities[i], mass=mass) for i in range(3)]
