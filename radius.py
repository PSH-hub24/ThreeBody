import numpy as np
from scipy.optimize import root
class radius:
    def __init__(self, a, e, m1, m2):
        self.a = a
        self.e = e
        self.m1 = m1
        self.m2 = m2
    def t(self, phi):
        return 2 * np.arctan(np.sqrt((1-self.e)/(1+e))*np.tan(phi/2))-(self.e * np.sqrt(1-self.e ** 2)*np.sin(phi))/(1+self.e*np.cos(phi))
    def radius(self, times):
        tvphi = 