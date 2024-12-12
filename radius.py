import numpy as np
from scipy.optimize import root, brentq
from scipy.fft import rfft
from golden_step import brents_method
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from prepWork import orbit, two_body_problem

# Function to test the two-body problem simulation
def test_two_body(a, e, tmax, tinterval, period_tolerance, plot = True):
    # Function to convert polar coordinates to Cartesian coordinates
    def polarToCartesian(r, phi):
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        return x, y

    # Initialization function for the animation
    def init():
        point.set_data([], [])
        trajectory.set_data([], [])
        return point, trajectory
    
    # Solve the two-body problem with given parameters
    r = two_body_problem(a, e, tmax, tinterval, period_tolerance=period_tolerance)
    
    # Print the results of the simulation
    print("Radii:", r.r)
    print("angles:", r.phi)
    print("Pericenter:", r.pericenter)
    print("period:", r.period)

    # If plot is True, set up the animation
    if(plot):
        fig, ax = plt.subplots()
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)

        # Plot elements: animated point and trajectory
        point, = ax.plot([], [], 'bo', label="mass traveling along trajectory")
        trajectory, = ax.plot([], [], 'r-', lw=1, label="Trajectory")
        
        # Plot the center of mass
        plt.scatter([0], [0], label = "Center of mass", s=20)
        
        # Convert polar coordinates to Cartesian for the trajectory
        x, y = polarToCartesian(r.r, r.phi)
        x = np.array(x)
        y = np.array(y)

        # Update function for the animation
        def update(frame):
            point.set_data(x[frame:frame+1], y[frame:frame+1])  # Update the point position
            trajectory.set_data(x[:frame], y[:frame])  # Update the trajectory
            return point, trajectory
        
        # Set plot labels and properties
        plt.xlabel("position in horizontal direction (scaled units)")
        plt.ylabel("position in vertical direction (scaled units)")
        tick_size = 10  # Specify the tick size
        ax.tick_params(axis='both', which='major', labelsize=tick_size)
        plt.xlim(-10, 10)
        plt.ylim(-10, 10)
        ax.set_aspect('equal')  # Ensure equal scaling on both axes

        # Create the animation
        ani = FuncAnimation(fig, update, frames=len(r.times), init_func=init, blit=True, interval=5)

        # Show the animation with legend
        plt.legend()
        plt.show()

    return r

