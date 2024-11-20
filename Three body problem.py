import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation

# Constants
G = 1.0  # Gravitational constant in unitless form
EPSILON = 1e-10  # Small value to avoid division by zero

class Body:
    def __init__(self, mass, position, velocity):
        self.mass = mass
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)

def compute_accelerations(bodies):
    accelerations = [np.zeros(3) for _ in bodies]
    for i, body_i in enumerate(bodies):
        for j, body_j in enumerate(bodies):
            if i != j:
                r_ij = body_j.position - body_i.position
                distance = np.linalg.norm(r_ij) + EPSILON
                accelerations[i] += G * body_j.mass * r_ij / distance**3
    return accelerations

def equations_of_motion(t, y, bodies):
    num_bodies = len(bodies)
    positions = y[:3*num_bodies].reshape((num_bodies, 3))
    velocities = y[3*num_bodies:].reshape((num_bodies, 3))
    
    for i, body in enumerate(bodies):
        body.position = positions[i]
        body.velocity = velocities[i]
    
    accelerations = compute_accelerations(bodies)
    
    dydt = np.zeros_like(y)
    dydt[:3*num_bodies] = velocities.flatten()
    dydt[3*num_bodies:] = np.array(accelerations).flatten()
    
    return dydt

def integrate_orbits(bodies, t_span, dt):
    y0 = np.zeros(6 * len(bodies))
    for i, body in enumerate(bodies):
        y0[3*i:3*(i+1)] = body.position
        y0[3*len(bodies) + 3*i:3*len(bodies) + 3*(i+1)] = body.velocity
    
    solution = solve_ivp(equations_of_motion, t_span, y0, args=(bodies,), method='RK45', t_eval=np.arange(t_span[0], t_span[1], dt))
    
    return solution

# Function to set up initial conditions for the scenario
def setup_stable_orbit_test(radii):
    # Large central mass
    M = 1.0  # Large mass
    m_small = 0.01  # Smaller mass for the orbiting particles

    # Main body (central massive particle)
    main_body = Body(M, [0, 0, 0], [0, 0, 0])

    # Smaller body in stable circular orbit around the main body
    stable_radius = 1.0
    stable_velocity = np.sqrt(G * M / stable_radius)
    orbiting_body1 = Body(m_small, [stable_radius, 0, 0], [0, stable_velocity, 0])

    # List to store results for different radii
    stability_results = {}

    for r in radii:
        # Second smaller body at varying radii
        velocity = np.sqrt(G * M / r)
        orbiting_body2 = Body(m_small, [r, 0, 0], [0, velocity, 0])

        # Run simulation with main body, orbiting_body1, and orbiting_body2
        bodies = [main_body, orbiting_body1, orbiting_body2]
        t_span = (0, 20)
        dt = 0.01

        # Integrate the orbit
        solution = integrate_orbits(bodies, t_span, dt)

        # Check stability
        max_distance1 = np.max(np.linalg.norm(solution.y[:3, :], axis=0))
        max_distance2 = np.max(np.linalg.norm(solution.y[6:9, :], axis=0))

        # Consider orbit stable if neither small particle exceeds a distance threshold
        stability_results[r] = (max_distance1 < 2 * stable_radius) and (max_distance2 < 2 * stable_radius)
        
        # Plotting the trajectories
        plot_orbits(bodies, solution, f"Orbit with Radius {r}")

    return stability_results
# Utility for plotting orbits
def plot_orbits(bodies, solution, title="Three-Body Problem"):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))  # Ensure fig is a single Figure object
    colors = ['red', 'blue', 'green']

    # Create empty line objects for each projection
    lines_xy = [axs[0].plot([], [], '.', color=colors[i], label=f'Body {i+1}')[0] for i in range(len(bodies))]
    lines_xz = [axs[1].plot([], [], '.', color=colors[i], label=f'Body {i+1}')[0] for i in range(len(bodies))]
    lines_yz = [axs[2].plot([], [], '.', color=colors[i], label=f'Body {i+1}')[0] for i in range(len(bodies))]

    for ax, label in zip(axs, ['XY Plane', 'XZ Plane', 'YZ Plane']):
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_xlabel('x' if 'X' in label else 'y')
        ax.set_ylabel('y' if 'Y' in label else 'z')
        ax.set_title(label)
        ax.legend()
    
    def init():
        for line in lines_xy + lines_xz + lines_yz:
            line.set_data([], [])
        return lines_xy + lines_xz + lines_yz

    def update(frame):
        for i, (line_xy, line_xz, line_yz) in enumerate(zip(lines_xy, lines_xz, lines_yz)):
            line_xy.set_data(solution.y[3*i, :frame], solution.y[3*i+1, :frame])
            line_xz.set_data(solution.y[3*i, :frame], solution.y[3*i+2, :frame])
            line_yz.set_data(solution.y[3*i+1, :frame], solution.y[3*i+2, :frame])
        return lines_xy + lines_xz + lines_yz

    # Check that frames is correctly set to an integer or array-like object
    ani = FuncAnimation(fig, update, frames=len(solution.t), init_func=init, blit=True)
    
    # Ensure saving format is correctly handled
    ani.save(f"{title.replace(' ', '_')}.gif", writer="pillow")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Test with varying radii
    radii = [1.1, 1.5, 2.0, 2.5, 3.0]
    results = setup_stable_orbit_test(radii)

    # Output stability results for each radius
    print("Stability Results for Varying Radii:")
    for radius, is_stable in results.items():
        print(f"Radius {radius}: {'Stable' if is_stable else 'Unstable'}")
