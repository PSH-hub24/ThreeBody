import numpy as np
from body import Body

def earth_moon_lagrange_rotating():
    """Earth-Moon system with Lagrange points in a rotating frame."""
    # Constants (scaled units)
    earth_mass = 9.878e-1
    moon_mass = 1.22e-2
    lagrange_mass = 1.0e-10  # Small mass for Lagrange points

    earth_moon_distance = 1.0  # Distance from Earth to Moon

    # Correct angular velocity based on the total mass and distance
    omega = np.sqrt((earth_mass + moon_mass) / earth_moon_distance**3)

    # Define Earth and Moon positions relative to the center of mass
    earth_position = np.array([-moon_mass / (earth_mass + moon_mass), 0, 0], dtype=np.float64)
    moon_position = np.array([earth_moon_distance * earth_mass / (earth_mass + moon_mass), 0, 0], dtype=np.float64)

    # Angular velocity vector
    omega_vector = np.array([0, 0, omega], dtype=np.float64)

    # Compute velocities using cross product
    earth_velocity = np.cross(omega_vector, earth_position)
    moon_velocity = np.cross(omega_vector, moon_position)

    # Define Lagrange points positions
    lagrange_positions = [
        [0.8369, 0, 0],  # L1
        [1.1557, 0, 0],  # L2
        [-1.0051, 0, 0],  # L3
        [0.4878, 0.866, 0],  # L4
        [0.4878, -0.866, 0]  # L5
    ]

    # User selects a Lagrange point
    print("Choose a Lagrange point to place the mass:")
    print("0: L1")
    print("1: L2")
    print("2: L3")
    print("3: L4")
    print("4: L5")
    lagrange_choice = int(input("Enter the number of your choice: "))

    if lagrange_choice < 0 or lagrange_choice >= len(lagrange_positions):
        raise ValueError("Invalid choice for Lagrange point.")

    lagrange_position = np.array(lagrange_positions[lagrange_choice], dtype=np.float64)
    lagrange_velocity = np.cross(omega_vector, lagrange_position)  # Correct velocity in rotating frame

    # Print the selected point details
    print(f"Lagrange Point L{lagrange_choice + 1}: Position = {lagrange_position}, Velocity = {lagrange_velocity}")

    # Define bodies
    earth = Body(position=earth_position, velocity=earth_velocity, mass=earth_mass)
    moon = Body(position=moon_position, velocity=moon_velocity, mass=moon_mass)
    lagrange_body = Body(position=lagrange_position, velocity=lagrange_velocity, mass=lagrange_mass)

    return [earth, moon, lagrange_body]
