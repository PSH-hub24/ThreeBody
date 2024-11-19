from radius import two_body_problem, test_two_body
import numpy as np
import pytest

#nearly circular radius
a = 5
e = 0.01
tmax = 2 * np.pi
tinterval = 0.01
period_tolerance = 1e-5

unit_tolerance = 1e-1

r = test_two_body(a, e, tmax, tinterval, period_tolerance, "plots/circular")

print(f" testing pericenter: {np.abs(np.abs(np.min(r.r)) - np.abs(r.pericenter))}, pericenter: {r.pericenter}")
assert np.abs(np.abs(np.min(r.r)) - np.abs(r.pericenter)) < unit_tolerance
print(f"testing period: {np.abs(r.period - 2*np.pi)}, period: {r.period}")
assert np.abs(r.period - 2*np.pi) < unit_tolerance

#elliptical radius
a = 5
e = 0.5
tmax = 15 * np.pi
tinterval = 0.01
period_tolerance = 0.5

unit_tolerance = 1e-1

r = test_two_body(a, e, tmax, tinterval, period_tolerance, "plots/elliptical")

print(f" testing pericenter: {np.abs(np.abs(np.min(r.r)) - np.abs(r.pericenter))}, pericenter: {r.pericenter}")
assert np.abs(np.abs(np.min(r.r)) - np.abs(r.pericenter)) < unit_tolerance
print(f"testing period: {np.abs(r.period - 2*np.pi)}, period: {r.period}")
assert np.abs(r.period - 2*np.pi) < unit_tolerance

#highly elliptical radius
a = 5
e = 0.99
tmax = 15 * np.pi
tinterval = 0.01
period_tolerance = 0.5

unit_tolerance = 1e-1

r = test_two_body(a, e, tmax, tinterval, period_tolerance, "plots/highly_elliptical")

print(f" testing pericenter: {np.abs(np.abs(np.min(r.r)) - np.abs(r.pericenter))}, pericenter: {r.pericenter}")
assert np.abs(np.abs(np.min(r.r)) - np.abs(r.pericenter)) < unit_tolerance
print(f"testing period: {np.abs(r.period - 2*np.pi)}, period: {r.period}")
assert np.abs(r.period - 2*np.pi) < unit_tolerance
