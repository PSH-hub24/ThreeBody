import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import rfft
from golden_step import brents_method

class orbit():
    def __init__(self, rFunc, tCutoff = 10e3):
        self.rFunc = rFunc

        #Evaluate the function until the times and get the radii and magnitude
        self.times = np.arange(0, tCutoff, 1)
        self.rFunc = rFunc
        self.r = rFunc(self.times)
        self.tCutoff = tCutoff
        self.pericenter, self.tmin = self.pericenter(self.times)
        self.period = self.period(self.r, tCutoff= self.tCutoff)

    #Find the pericenter of the orbit
    def rSquared(self, times):
        return self.rFunc(times) ** 2
    def pericenter(self, times): #time intervals of 1

        #Use Golden Step minimization to find the minimum value of the magnitude
        a = times[0]
        b = times[times.size/2]
        c = times[-1]
        
        tmin = brents_method(self.rSquared(times), a, b, c)
        pericenter = self.rFunc(tmin)
        return pericenter, tmin

    #Find the orbital period
    def period(self, r, tCutoff = 10e3):
        rMag = r ** 2
        roots = rfft(rMag, tCutoff)
        return max(roots)




