import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import rfft
from golden_step import brents_method
from matplotlib.animation import FuncAnimation
from scipy.optimize import root, brentq

class orbit():
    def __init__(self, rFunc, tmax, tinterval, period_tolerance = 0.5):
        super().__init__()
        self.rFunc = rFunc
        self.tmax = tmax
        self.tinterval = tinterval
        self.times = np.arange(0, tmax, tinterval)
        self.rFunc = rFunc
        self.r = rFunc(self.times)
        self.pericenter = self.pericenter(self.r, self.times)
        self.period = self.period(self.r, self.pericenter, self.tmax, self.tinterval, tolerance=period_tolerance)

    #Find the pericenter of the orbit
    def rSquared(self, times):
        return self.rFunc(times) ** 2
    def pericenter(self, r, times): #time intervals of 1

        #Use Golden Step minimization to find the minimum value of the magnitude
        a = times[0]
        b = times[int(times.size/2)]
        c = times[-1]
        
        rmin = np.sqrt(np.min(r ** 2))
        return rmin
    #Find the orbital period
    def period(self, r, pericenter, tmax, tinterval, tolerance=0.5):
        # Identify times where r is close to the pericenter
        pericenter_times = np.where(np.isclose(r, pericenter, atol=tolerance))[0]
        
        # Remove redundant points (if two are very close, keep only one)
        # Use slicing instead of np.insert to avoid infinity-related issues
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



