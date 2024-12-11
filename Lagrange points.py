import numpy as np
from scipy.optimize import fsolve

import matplotlib.pyplot as plt

# Constants
G = 6.67430e-11  # gravitational constant
M1 = 5.972e24    # mass of the first body (e.g., Earth)
M2 = 7.348e22    # mass of the second body (e.g., Moon)
d = 3.844e8      # distance between the two bodies (e.g., Earth-Moon distance)
Ω = np.sqrt(G * (M1 + M2) / d**3)  # angular velocity for circular orbit

# Effective potential function
def effective_potential(x, y):
    r1 = np.sqrt((x + M2 * d / (M1 + M2))**2 + y**2)
    r2 = np.sqrt((x - M1 * d / (M1 + M2))**2 + y**2)
    U = -G * M1 / r1 - G * M2 / r2 - 0.5 * Ω**2 * (x**2 + y**2)
    return U

# Grid for plotting
x = np.linspace(-2 * d, 2 * d, 400)
y = np.linspace(-2 * d, 2 * d, 400)
X, Y = np.meshgrid(x, y)
Z = effective_potential(X, Y)

# Plotting the effective potential
plt.figure(figsize=(10, 8))
plt.contourf(X, Y, Z, levels=50, cmap='viridis')
plt.colorbar(label='Effective Potential')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.title('Effective Potential in the Rotating Frame')
plt.scatter([0, d], [0, 0], color='red', label='Massive Bodies')
plt.legend()
plt.show()

# Identifying Lagrange Points

def equations(p):
    x, y = p
    r1 = np.sqrt((x + M2 * d / (M1 + M2))**2 + y**2)
    r2 = np.sqrt((x - M1 * d / (M1 + M2))**2 + y**2)
    eq1 = -G * M1 * (x + M2 * d / (M1 + M2)) / r1**3 - G * M2 * (x - M1 * d / (M1 + M2)) / r2**3 + Ω**2 * x
    eq2 = -G * M1 * y / r1**3 - G * M2 * y / r2**3 + Ω**2 * y
    return (eq1, eq2)

# Initial guesses for Lagrange points
initial_guesses = [(d / 2, 0), (-d / 2, 0), (d, 0), (0, d), (0, -d)]
lagrange_points = [fsolve(equations, guess) for guess in initial_guesses]

# Plotting Lagrange Points
plt.figure(figsize=(10, 8))
plt.contourf(X, Y, Z, levels=50, cmap='viridis')
plt.colorbar(label='Effective Potential')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.title('Effective Potential with Lagrange Points')
plt.scatter([0, d], [0, 0], color='red', label='Massive Bodies')
for point in lagrange_points:
    plt.scatter(point[0], point[1], color='blue', label='Lagrange Point')
plt.legend()
plt.show()

# Lagrange Points
for i, point in enumerate(lagrange_points):
    print(f"L{i+1} Point: x = {point[0]:.2e} m, y = {point[1]:.2e} m")