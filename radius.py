import numpy as np
from scipy.optimize import root, brentq
from scipy.fft import rfft
from golden_step import brents_method
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class radius:
    def __init__(self, a, e, times):
        self.a = a
        self.e = e
        self.times = times
        self.r = self.radius(self.times)
        self.pericenter = self.pericenter(self.r, self.times)
        self.period = self.period(self.r, self.times.size)
    def t(self, phi):
        return 2 * np.arctan(np.sqrt((1-self.e)/(1+self.e))*np.tan(phi/2))-(self.e * np.sqrt(1-self.e ** 2)*np.sin(phi))/(1+self.e*np.cos(phi))
    def radius(self, time):
        def offset_time(time):
            def offset(phi):

                return self.t(phi) - time
            return offset
        self.times = time
        phi = []
        for t in self.times:
            phi.append(brentq(offset_time(t), -np.pi, np.pi))
        phi = np.array(phi)
        self.phi = phi
        return (self.a * (1- self.e ** 2))/(1+self.e*np.cos(phi))
    #Find the pericenter of the orbit
    def pericenter(self, r, times): #time intervals of 1

        #Use Golden Step minimization to find the minimum value of the magnitude
        a = times[0]
        b = times[int(times.size/2)]
        c = times[-1]
        
        rmin = np.min(r)
        return rmin

    #Find the orbital period
    def period(self, r, tCutoff = 10e3):
        rMag = r ** 2
        roots = rfft(rMag, tCutoff)
        return max(roots)
    
# Create the figure and axis
fig, ax = plt.subplots()
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)

# Plot elements
point, = ax.plot([], [], 'bo', label="Moving Point")  # Animated point
trajectory, = ax.plot([], [], 'r-', lw=1, label="Trajectory")  # Path
def polarToCartesian(r, phi):
    x = r * np.cos(phi)
    y = r* np.sin(phi)
    return x, y
def init():
    point.set_data([], [])
    trajectory.set_data([], [])
    return point, trajectory



times = np.arange(-np.pi, np.pi, 0.01)
r = radius(5, 0.9, times)

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
ani = FuncAnimation(fig, update, frames=len(r.times), init_func=init, blit=True, interval=50)

# Show the animation
plt.legend()
plt.show()