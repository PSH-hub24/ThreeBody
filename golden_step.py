import numpy as np
from scipy.optimize import brent

def parabolic_step(func=None, a=None, b=None, c=None):
    """returns the minimum of the function as approximated by a parabola"""
    fa = func(a)
    fb = func(b)
    fc = func(c)
    denom = (b - a) * (fb - fc) - (b -c) * (fb - fa)
    numer = (b - a)**2 * (fb - fc) - (b -c)**2 * (fb - fa)
    # If singular, just return b 
    if(np.abs(denom) < 1.e-15):
        x = b
    else:
        x = b - 0.5 * numer / denom
    return(x)

def golden_step(func=None, astart=None, bstart=None, cstart=None, tol=1.e-5):
    # xgrid = -12. + 25. * np.arange(10000) / 10000. 
    # plt.plot(xgrid, func(xgrid))
    gsection = (3. - np.sqrt(5)) / 2
    a = astart
    b = bstart
    c = cstart

    if((b - a) > (c - b)):
        x = b
        b = b - gsection * (b - a)
    else:
        x = b + gsection * (c - b)

    return x


def brents_method(f, left, right, tol=1e-10, max_iter=1000):
    a = left
    c = right
    b = (left+right)/2

    fb = f(b)
    laststep = np.abs(b-a)
    for _ in range(max_iter):
        if(b>c or b<a):
            b = golden_step(f, a, b, c)
        beff = parabolic_step(f, a, b, c)
        if(laststep < np.abs(beff-b)):
            b = golden_step(f, a, b, c)
        else:
            b = beff

        newBound = (c-a)/2+a
        if(newBound>b):
            c = newBound
        if(newBound<b):
            a = newBound
    return b


def f(x):
    return (x - 0.3) ** 2 *np.exp(x)

# Minimize f(x) on the interval [0, 4]
# xmin = brents_method(f, 0, 4)
# scipymin = brent(f, brack = (0, 4))
# print(f"Minimum at x = {xmin} with value f(x) = {f(xmin)}")
# print(f"scipy min at x = {scipymin} with value f(x) = {f(scipymin)}")
