from body import Body

# This file contains the initial conditions for the Broucke periodic orbit.
# The Broucke periodic orbit is a solution to the three-body problem that is stable and periodic.
# The initial conditions for the Broucke periodic orbit are defined as follows:

def broucke_orbit():
    """Broucke periodic orbit."""
    mass = [0.44, 0.87, 1.0]
    positions = [
        [-1.219929, 0.0, 0.0],
        [1, 0.0 , 0.0],
        [0.0 , 0.0 , 0.0]
    ]
    velocities = [
        [0.0, -0.992252, 0.0],
        [0.0, -0.513024, 0.0],
        [0.0, 0.88292176, 0.0]
    ]
    return [Body(position=positions[i], velocity=velocities[i], mass=mass[i]) for i in range(3)]
