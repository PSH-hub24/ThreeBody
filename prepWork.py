import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import rfft
from golden_step import brents_method
from matplotlib.animation import FuncAnimation
from scipy.optimize import root, brentq

# Orbit class to represent and compute properties of a two-body problem
class orbit():
    def __init__(self, rFunc, tmax, tinterval, period_tolerance = 0.5):
        super().__init__()
        self.rFunc = rFunc  # Radius function
        self.tmax = tmax  # Maximum time for the simulation
        self.tinterval = tinterval  # Time interval for each step
        self.times = np.arange(0, tmax, tinterval)  # Array of time values
        self.r = rFunc(self.times)  # Calculate radius values over time
        self.pericenter = self.pericenter(self.r, self.times)  # Find the pericenter
        self.period = self.period(self.r, self.pericenter, self.tmax, self.tinterval, tolerance=period_tolerance)  # Calculate the orbital period

    # Method to find the squared radius at given times
    def rSquared(self, times):
        return self.rFunc(times) ** 2

    # Method to find the pericenter of the orbit
    def pericenter(self, r, times):  # Time intervals of 1
        # Use Golden Step minimization to find the minimum value of the magnitude
        a = times[0]
        b = times[int(times.size/2)]
        c = times[-1]
        
        rmin = np.sqrt(np.min(r ** 2))  # Minimum radius value
        return rmin

    # Method to find the orbital period
    def period(self, r, pericenter, tmax, tinterval, tolerance=0.5):
        # Identify times where r is close to the pericenter
        pericenter_times = np.where(np.isclose(r, pericenter, atol=tolerance))[0]
        
        # Remove redundant points (if two are very close, keep only one)
        valid_differences = np.diff(pericenter_times) > 1
        pericenter_times = pericenter_times[np.insert(valid_differences, 0, True)]  # Keep the first point

        # Calculate the periods as differences between consecutive pericenter times
        periods = np.diff(pericenter_times)
        
        # Average period in terms of time steps
        time_step_period = np.mean(periods)
        
        # Total time steps in the simulation
        total_time_steps = tmax / tinterval
        
        # Convert the period to physical time
        period = (time_step_period / total_time_steps) * tmax
        
        return period

# Two-body problem class inheriting from orbit class
class two_body_problem(orbit):
    def __init__(self, a, e, tmax, tinterval, period_tolerance):
        self.a = a  # Semi-major axis
        self.e = e  # Eccentricity
        rFunc = self.radius  # Radius function
        super().__init__(rFunc = rFunc, tmax = tmax, tinterval = tinterval, period_tolerance = period_tolerance)

    # Method to calculate time as a function of angle (phi)
    def t(self, phi):
        return 2 * np.arctan(np.sqrt((1-self.e)/(1+self.e))*np.tan(phi/2))-(self.e * np.sqrt(1-self.e ** 2)*np.sin(phi))/(1+self.e*np.cos(phi))

    # Method to calculate radius as a function of time
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
            phi.append(brentq(offset_time(t), -np.pi, np.pi))  # Solve for phi using Brent's method
        phi = np.array(phi)
        self.phi = phi
        r = (self.a * (1- self.e ** 2))/(1+self.e*np.cos(phi))  # Calculate the radius
        return r
