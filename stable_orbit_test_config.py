import numpy as np
from body import Body

def stable_orbit_test():
    """
    Configure a system for testing stability with one large mass and two smaller masses.
    Returns the list of bodies, time span, and time step for the simulation.
    """
    # Define masses
    large_mass = 100.0
    small_mass_1 = 1.0

    # Initial conditions for the large mass
    large_body = Body(position=[0, 0, 0], velocity=[0, 0, 0], mass=large_mass)

    # Circular orbit for the first small mass
    radius_1 = 10.0
    velocity_1 = np.sqrt(large_mass / radius_1)
    small_body_1 = Body(position=[radius_1, 0, 0], velocity=[0, velocity_1, 0], mass=small_mass_1)

    # Prompt user for third mass initial conditions
    print("Define initial conditions for the third mass.")
    mass_3 = float(input("Mass of the third mass: "))
   
    x = float(input("Initial x-position: "))
    y = float(input("Initial y-position: "))
    z = float(input("Initial z-position: "))
    vx = float(input("Initial x-velocity: "))
    vy = float(input("Initial y-velocity: "))
    vz = float(input("Initial z-velocity: "))

   

    small_body_2 = Body(position=[x, y, z], velocity=[vx, vy, vz], mass=mass_3)

    # Create the system
    bodies = [large_body, small_body_1, small_body_2]

    # Prompt for time span and step
    t_total = np.float64(input("Enter total simulation time: "))
    dt = np.float64(input("Enter time step: "))
    t_span = (0, t_total)

    return bodies, t_span, dt
