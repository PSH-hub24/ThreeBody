import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation

class Body:
    def __init__(self, mass, position, velocity):
        self.mass = mass
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)

def compute_accelerations(bodies):
    G = 1.0  # Gravitational constant in our unitless system
    accelerations = [np.zeros(3) for _ in bodies]
    for i, body_i in enumerate(bodies):
        for j, body_j in enumerate(bodies):
            if i != j:
                r_ij = body_j.position - body_i.position
                distance = np.linalg.norm(r_ij)
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

# Example usage
if __name__ == "__main__":
    body1 = Body(0.001, [1, -1, 0], [0, 1, 0])
    body2 = Body(1.0, [0, 0, 0], [0, 0, 0])
    body3 = Body(0.001, [-1, 1, 0], [0, -1, 0])
    
    bodies = [body1, body2, body3]
    t_span = (0, 10)
    dt = 0.01
    
    solution = integrate_orbits(bodies, t_span, dt)
    
    # Plotting the results in 2D subplots for different projections
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # Create empty line objects for each projection
    lines_xy = [axs[0].plot([], [], 'o', label=f'Body {i+1}')[0] for i in range(len(bodies))]
    lines_xz = [axs[1].plot([], [], 'o', label=f'Body {i+1}')[0] for i in range(len(bodies))]
    lines_yz = [axs[2].plot([], [], 'o', label=f'Body {i+1}')[0] for i in range(len(bodies))]

    # Set axis labels and limits
    for ax, label in zip(axs, ['XY Plane', 'XZ Plane', 'YZ Plane']):
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_xlabel('x' if 'X' in label else 'y')
        ax.set_ylabel('y' if 'Y' in label else 'z')
        ax.set_title(label)
        ax.legend()

    def init():
        for line in lines_xy + lines_xz + lines_yz:
            line.set_data([], [])
        return lines_xy + lines_xz + lines_yz

    def update(frame):
        # Update each projection
        for i, (line_xy, line_xz, line_yz) in enumerate(zip(lines_xy, lines_xz, lines_yz)):
            line_xy.set_data(solution.y[3*i, :frame], solution.y[3*i+1, :frame])  # x-y plane
            line_xz.set_data(solution.y[3*i, :frame], solution.y[3*i+2, :frame])  # x-z plane
            line_yz.set_data(solution.y[3*i+1, :frame], solution.y[3*i+2, :frame])  # y-z plane
        return lines_xy + lines_xz + lines_yz

    ani = FuncAnimation(fig, update, frames=len(solution.t), init_func=init, blit=True)
    plt.tight_layout()
    plt.show()
