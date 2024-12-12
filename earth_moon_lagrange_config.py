import numpy as np
from body import Body

def earth_moon_lagrange():
    """Earth-Moon system with Lagrange points."""
    # Constants (scaled units)
    earth_mass = 9.88e-1
    moon_mass = 1.22e-2
    lagrange_mass = 1.0e-10 # Small mass for Lagrange points

    earth_moon_distance = 1.0  # Distance from Earth to Moon

    # Orbital velocities (circular approximation)
    moon_velocity = 1.0  # Moon around Earth

    # Earth position and velocity relative to the Moon
    earth_position = np.array([0, 0, 0], dtype=np.float64)
    earth_velocity_vector = np.array([0, 0, 0])

    # Moon position and velocity relative to the Earth
    moon_position = np.array([earth_moon_distance, 0, 0], dtype=np.float64)
    moon_velocity_vector = np.array([0, moon_velocity, 0], dtype=np.float64)

    # Define Lagrange points positions and velocities
    lagrange_positions = [
        [0.837, 0, 0],  # L1
        [1.16, 0, 0],  # L2
            [-1.01, 0, 0],  # L3
            [0.488,  0.866, 0],  # L4
            [0.488, -0.866, 0]  # L5
        ]
    lagrange_velocities = [
        [0, moon_velocity * 0.837, 0],  # L1
        [0, moon_velocity * 1.16, 0],  # L2
        [0, moon_velocity * -1.01, 0],  # L3
        [-moon_velocity * 0.866, moon_velocity * 0.488, 0],  # L4
        [moon_velocity * 0.866, moon_velocity * 0.488, 0]  # L5
        ]
    print("Choose a Lagrange point to place the mass:")
    print("0: L1")
    print("1: L2")
    print("2: L3")
    print("3: L4")
    print("4: L5")
    lagrange_points = int(input("Enter the number of your choice: "))
    position = lagrange_positions[lagrange_points]
    velocity = lagrange_velocities[lagrange_points]
    # Define bodies
    earth = Body(position=earth_position, velocity=earth_velocity_vector, mass=earth_mass)
    moon = Body(position=moon_position, velocity=moon_velocity_vector, mass=moon_mass)
    lagrange_points = [Body(position=position, velocity=velocity, mass=lagrange_mass)]

    return [earth, moon] + lagrange_points
