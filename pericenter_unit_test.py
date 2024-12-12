from radius import two_body_problem, test_two_body
import numpy as np
import pytest

def run_unit_tests(a, e, tmax, tinterval, period_tolerance, unit_tolerance):
    r = test_two_body(a, e, tmax, tinterval, period_tolerance, plot=False)
    
    print(f"Testing pericenter: {np.abs(np.abs(np.min(r.r)) - np.abs(r.pericenter))}, Pericenter: {r.pericenter}")
    assert np.abs(np.abs(np.min(r.r)) - np.abs(r.pericenter)) < unit_tolerance
    
    print(f"Testing period: {np.abs(r.period - 2*np.pi)}, Period: {r.period}")
    assert np.abs(r.period - 2*np.pi) < unit_tolerance

# Parameters for different test cases
test_cases = [
    (5, 0.01, 2 * np.pi, 0.01, 1e-5, 1e-1),  # Nearly circular radius
    (5, 0.5, 15 * np.pi, 0.01, .01, 1e-1),  # Elliptical radius
    (5, 0.90, 15 * np.pi, 0.01, .01, 1e-1)  # Highly elliptical radius
]

# Execute the unit tests for each test case
for params in test_cases:
    run_unit_tests(*params)
