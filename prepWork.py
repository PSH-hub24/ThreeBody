import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as optimize
from golden_step import golden_step

class orbit():
    def __init__(self, rFunc, tCutoff = 10e3):
        self.rFunc = rFunc

        #Evaluate the function until the times and get the radii and magnitude
        self.times = np.arange(0, tCutoff)
        self.r = rFunc(self.times)
        self.rMag = self.r ** 2

        self.tCutoff = tCutoff
        self.pericenter, self.tmin = self.pericenter(self.r, self.times, self.tCutoff)


    def pericenter(r, times, tCutoff = 10e3): #time intervals of 1

        #Use Golden Step minimization to find the minimum value of the magnitude
        a = times[0]
        b = times[times.size/2]
        c = times[-1]
        
        tmin = golden_step(r ** 2, a, b, c)
        pericenter = r[tmin]
        return pericenter, tmin

    def period(r, pericenter, tCutoff = 10e3):
        rMag = r ** 2
        minFunc = rMag - pericenter
        roots = optimize.fsolve




