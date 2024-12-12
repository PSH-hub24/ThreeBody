from body import Body
import numpy as np

def jupiter_ganymede_io():
    """
    Define a simplified Jupiter-Ganymede-Io system.

    Returns:
    list: A list of Body objects representing Jupiter, Ganymede, and Io.
    """

    # Scaled masses (arbitrary units)
    jupiter_mass = 1e5
    ganymede_mass = 1.4819e-3
    io_mass = 4.86e-5

    # Semi-major axes (scaled units)
    ganymede_distance = 15.0
    io_distance = 6.0

    # Orbital velocities (from Kepler's third law: v = sqrt(GM/r))
    ganymede_velocity = np.sqrt(jupiter_mass / ganymede_distance)
    io_velocity = np.sqrt(jupiter_mass / io_distance)

    return [
        Body(position=[0, 0, 0], velocity=[0, 0, 0], mass=jupiter_mass),                     # Jupiter
        Body(position=[ganymede_distance, 0, 0], velocity=[0, ganymede_velocity, 0], mass=ganymede_mass),  # Ganymede
        Body(position=[io_distance, 0, 0], velocity=[0, io_velocity, 0], mass=io_mass)      # Io
    ]
