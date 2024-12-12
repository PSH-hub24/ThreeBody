import numpy as np
from body import Body

def solar_system():
    """
    Define a simplified Solar System with the Sun and 8 planets.

    Returns:
    list: A list of Body objects representing the Solar System.
    """
    """Simplified Solar System configuration (Sun + 8 planets)."""
    # Approximate masses (scaled, relative to the Sun's mass)

    masses = [
        100.0,                  # Sun
        3.30e-5,              # Mercury 
        2.45e-4,              # Venus
        3.00e-4,              # Earth 
        3.23e-5,              # Mars
        9.54e-2,              # Jupiter
        2.86e-2,              # Saturn
        4.37e-3,              # Uranus
        5.15e-3               # Neptune
    ]

    # Approximate semi-major axes (scaled in AU)
    distances = [
        0.0,      # Sun
        0.39,     # Mercury
        0.72,     # Venus
        1.0,      # Earth
        1.52,     # Mars
        5.2,      # Jupiter
        9.58,     # Saturn
        19.2,     # Uranus
        30.05     # Neptune
    ]

    # Orbital velocities (v = sqrt(GM/r), scaled to match distances)
    velocities = [
        0.0,                     # Sun
        2 * np.pi / np.sqrt(0.39),  # Mercury
        2 * np.pi / np.sqrt(0.72),  # Venus
        2 * np.pi / np.sqrt(1.0),   # Earth
        2 * np.pi / np.sqrt(1.52),  # Mars
        2 * np.pi / np.sqrt(5.2),   # Jupiter
        2 * np.pi / np.sqrt(9.58),  # Saturn
        2 * np.pi / np.sqrt(19.2),  # Uranus
        2 * np.pi / np.sqrt(30.05)  # Neptune
    ]

    # Initialize bodies
    bodies = [
        Body(position=[0, 0, 0], velocity=[0, 0, 0], mass=masses[0])  # Sun
    ]
    for i in range(1, len(masses)):
        position = [distances[i], 0, 0]
        velocity = [0, velocities[i], 0]
        bodies.append(Body(position=position, velocity=velocity, mass=masses[i]))
    
    return bodies
