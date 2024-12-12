import numpy as np
from body import Body

def sun_earth_moon():
    """Sun-Earth-Moon configuration with correct initial conditions."""
    # Constants (scaled units)
    sun_mass = 100.0
    earth_mass = 3.0e-4
    moon_mass = 3.7e-6

    earth_sun_distance = 1.0  # Distance from Earth to Sun
    moon_earth_distance = 0.00257  # Distance from Moon to Earth

    # Orbital velocities (circular approximation)
    earth_velocity = np.sqrt(sun_mass / earth_sun_distance)  # Earth around Sun
    moon_velocity = np.sqrt(earth_mass / moon_earth_distance)  # Moon around Earth

    # Earth position and velocity relative to the Sun
    earth_position = np.array([earth_sun_distance, 0, 0], dtype=np.float64)
    earth_velocity_vector = np.array([0, earth_velocity, 0])

    # Moon position and velocity relative to the Earth
    moon_position_relative = np.array([moon_earth_distance, 0, 0], dtype=np.float64)
    moon_velocity_relative = np.array([0, moon_velocity, 0], dtype=np.float64)

    # Moon position and velocity in the Sun-centered frame
    moon_position = earth_position + moon_position_relative
    moon_velocity = earth_velocity_vector + moon_velocity_relative

    # Define bodies
    sun = Body(position=[0, 0, 0], velocity=[0, 0, 0], mass=sun_mass)
    earth = Body(position=earth_position, velocity=earth_velocity_vector, mass=earth_mass)
    moon = Body(position=moon_position, velocity=moon_velocity, mass=moon_mass)

    return [sun, earth, moon]
