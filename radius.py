import numpy as np
from scipy.optimize import root, brentq
from scipy.fft import rfft
from golden_step import brents_method
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from prepWork import orbit

class two_body_problem(orbit):
    def __init__(self, a, e, tmax, tinterval, period_tolerance):
        self.a = a
        self.e = e
        rFunc = self.radius
        super().__init__(rFunc = rFunc,tmax = tmax,tinterval = tinterval,period_tolerance = period_tolerance)
        # self.tmax = tmax
        # self.tinterval = tinterval

        # self.times = np.arange(0, tmax, tinterval)
        # self.r = self.radius(self.times)
        # self.pericenter = self.pericenter(self.r, self.times)
        # self.period = self.period()
    def t(self, phi):
        return 2 * np.arctan(np.sqrt((1-self.e)/(1+self.e))*np.tan(phi/2))-(self.e * np.sqrt(1-self.e ** 2)*np.sin(phi))/(1+self.e*np.cos(phi))
    def radius(self, time):
        def offset_time(time, tolerance = 1e-8):
            def offset(phi):
                n = int((time+np.pi)/(2*np.pi))
                t0 = time - 2*np.pi*(n)
                return self.t(phi) - t0 - tolerance
            return offset
        self.times = time
        phi = []
        for t in self.times:
            phi.append(brentq(offset_time(t), -np.pi, np.pi))
        phi = np.array(phi)
        self.phi = phi
        r = (self.a * (1- self.e ** 2))/(1+self.e*np.cos(phi))
        return r
def test_two_body(a, e, tmax, tinterval, period_tolerance, save_path):
    fig, ax = plt.subplots()
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)

    # Plot elements
    point, = ax.plot([], [], 'bo', label="orbit")  # Animated point
    trajectory, = ax.plot([], [], 'r-', lw=1, label="Trajectory")  # Path
    def polarToCartesian(r, phi):
        x = r * np.cos(phi)
        y = r* np.sin(phi)
        return x, y
    def init():
        point.set_data([], [])
        trajectory.set_data([], [])
        return point, trajectory

    r = two_body_problem(a, e, tmax, tinterval, period_tolerance=period_tolerance)

    print("Radii:", r.r)
    print("angles:", r.phi)
    print("Pericenter:", r.pericenter)
    print("period:", r.period)

    x, y = polarToCartesian(r.r, r.phi)
    x = np.array(x)
    y = np.array(y)
    # Update function for animation
    def update(frame):
        point.set_data(x[frame:frame+1], y[frame:frame+1])  # Pass a sequence with one element at a time
        trajectory.set_data(x[:frame], y[:frame])  # Update the trajectory
        return point, trajectory
    # Create the animation
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    ani = FuncAnimation(fig, update, frames=len(r.times), init_func=init, blit=True, interval=10)
    ani.save(save_path + ".gif", writer='ffmpeg', fps=30)
    # Show the animation
    plt.legend()
    plt.show()
    return r
    


# a = 5
# e = 0.99
# tmax = 15*np.pi
# tinterval = 0.01
# period_tolerance = 0.5
# test_two_body(a, e, tmax, tinterval, period_tolerance)