import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

# Constants
G = 6.67430e-11  # Gravitational constant (m^3 kg^-1 s^-2)
M1 = 5.972e24    # Mass of the first body (e.g., Earth, in kg)
M2 = 7.348e22    # Mass of the second body (e.g., Moon, in kg)
d = 3.844e8      # Distance between the two bodies (e.g., Earth-Moon distance, in m)

# Define unitless variables
L = d  # Characteristic length scale
M = M1 + M2  # Total mass scale

# Characteristic time scale
T = np.sqrt(L**3 / (G * M))  # Orbital period scale

# Unitless gravitational constant
G_unitless = G * M * T**2 / L**3

# Unitless masses
m1 = M1 / M
m2 = M2 / M

# Unitless distance
d_unitless = d / L
Ω = np.sqrt(G_unitless * (m1 + m2) / d_unitless**3)  # Angular velocity for circular orbits

# Unitless effective potential function
def effective_potential_unitless(x_unitless, y_unitless):
    r1 = np.sqrt((x_unitless + m2 * d_unitless / (m1 + m2))**2 + y_unitless**2)
    r2 = np.sqrt((x_unitless - m1 * d_unitless / (m1 + m2))**2 + y_unitless**2)
    U = -G_unitless * m1 / r1 - G_unitless * m2 / r2 - 0.5 * (x_unitless**2 + y_unitless**2)
    return U

# Grid for plotting in unitless variables
x_unitless = np.linspace(-2 * d_unitless, 2 * d_unitless, 400)
y_unitless = np.linspace(-2 * d_unitless, 2 * d_unitless, 400)
X_unitless, Y_unitless = np.meshgrid(x_unitless, y_unitless)
Z_unitless = effective_potential_unitless(X_unitless, Y_unitless)

# Solve for Lagrange points
def equations(p):
    x_unitless, y_unitless = p
    r1 = np.sqrt((x_unitless + m2 * d_unitless / (m1 + m2))**2 + y_unitless**2)
    r2 = np.sqrt((x_unitless - m1 * d_unitless / (m1 + m2))**2 + y_unitless**2)
    eq1 = -G_unitless * m1 * (x_unitless + m2 * d_unitless / (m1 + m2)) / r1**3 - \
          G_unitless * m2 * (x_unitless - m1 * d_unitless / (m1 + m2)) / r2**3 + \
          Ω**2 * x_unitless
    eq2 = -G_unitless * m1 * y_unitless / r1**3 - \
          G_unitless * m2 * y_unitless / r2**3 + \
          Ω**2 * y_unitless
    return (eq1, eq2)

# Initial guesses for Lagrange points
initial_guesses = [
    (d_unitless * 0.8, 0),    # L1
    (d_unitless * 1.2, 0),    # L2
    (-d_unitless * 1.1, 0),   # L3
    (d_unitless / 2, np.sqrt(3) / 2 * d_unitless),  # L4
    (d_unitless / 2, -np.sqrt(3) / 2 * d_unitless)  # L5
]

# Solve for Lagrange points
lagrange_points = [fsolve(equations, guess) for guess in initial_guesses]

# Plotting the effective potential with Lagrange points
plt.figure(figsize=(10, 8))

# Filled contours
plt.contourf(X_unitless, Y_unitless, Z_unitless, levels=100, cmap='viridis')

# Massive bodies
plt.scatter([0, d_unitless], [0, 0], color='red', label='Massive Bodies')

# Lagrange points
colors = ['blue', 'green', 'orange', 'purple', 'cyan']
for i, point in enumerate(lagrange_points):
    plt.scatter(point[0], point[1], color=colors[i], label=f'Lagrange Point {i+1}')

# Plot formatting
plt.colorbar(label='Effective Potential')
plt.xlabel('x (unitless)')
plt.ylabel('y (unitless)')
plt.title('Effective Potential with Lagrange Points')
plt.legend()
plt.savefig('effective_potential_with_lagrange_points.png')
plt.show()

# Plot without Lagrange points for comparison
plt.figure(figsize=(10, 8))

# Filled contours
plt.contourf(X_unitless, Y_unitless, Z_unitless, levels=100, cmap='viridis')

# Massive bodies
plt.scatter([0, d_unitless], [0, 0], color='red', label='Massive Bodies')

# Plot formatting
plt.colorbar(label='Effective Potential')
plt.xlabel('x (unitless)')
plt.ylabel('y (unitless)')
plt.title('Effective Potential in the Rotating Frame')
plt.legend()
plt.savefig('effective_potential.png')
plt.show()

def distance_from_origin(point):
    return np.sqrt(point[0]**2 + point[1]**2)

# Print initial positions and distances of Lagrange points
print("Lagrange Points in Dimensionless Variables:")
for i, point in enumerate(lagrange_points):
    distance = distance_from_origin(point)
    print(f"L{i+1}: Position = (x = {point[0]:.4f}, y = {point[1]:.4f}), Distance from Origin = {distance:.4f}")

# Print system details
print(f"Mass ratio: m1 = {m1:.4f}, m2 = {m2:.4f}")
print(f"Dimensionless distance: d_unitless = {d_unitless:.4f}")
print(f"Angular velocity: Ω = {Ω:.4f}")