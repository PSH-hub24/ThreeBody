import numpy as np
# Define the Body class to represent celestial bodies
class Body:
    def __init__(self, position, velocity, mass):
        """
        Initialize a celestial body with a position, velocity, and mass.
        Parameters:
        - position (list): [x, y, z] coordinates of the body.
        - velocity (list): [vx, vy, vz] velocity components of the body.
        - mass (float): Mass of the body.
        """
        self.position = np.array(position, dtype=np.float64)
        self.velocity = np.array(velocity, dtype=np.float64)
        self.mass = mass
