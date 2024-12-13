from body import Body  # Importing the Body class from the body module
import numpy as np

def two_large_equal_mass_and_small_particle():
    """Setup for two large equal mass particles and one small particle."""
    r0 = 5.0  # Distance between the two large particles (dimensionless)
    mass = 100.0  # Mass of the large particles (dimensionless)
    
    # Set up the positions and velocities for the large particles
    position_large_1 = np.array([-r0 / 2, 0.0, 0.0])  # Particle 1 at (-r0/2, 0)
    position_large_2 = np.array([r0 / 2, 0.0, 0.0])   # Particle 2 at (r0/2, 0)

    # Compute the orbital velocity for the large particles (in dimensionless units)
    orbital_velocity = np.sqrt( mass / (2*r0))  # For circular orbits in dimensionless units

    # Velocities for the large particles (perpendicular to the line connecting them)
    velocity_large_1 = np.array([0.0, orbital_velocity, 0.0])  # Particle 1 has velocity in y-direction
    velocity_large_2 = np.array([0.0, -orbital_velocity, 0.0]) # Particle 2 has velocity in negative y-direction

    # Set up the position and velocity for the small particle
    position_small = np.array([50 , 0.0, 0.0])  # Placed along the x-axis
    velocity_small = np.array([0.5,0.0 , 0.0])  # Initial velocity of small particle

    # Create the bodies in dimensionless units
    body_large_1 = Body(position=position_large_1, velocity=velocity_large_1, mass=mass)
    body_large_2 = Body(position=position_large_2, velocity=velocity_large_2, mass=mass)
    body_small = Body(position=position_small, velocity=velocity_small, mass=mass * 0.01)  # Small mass in dimensionless units
    
    # Return the bodies in a list for further simulation
    return [body_large_1, body_large_2, body_small]

