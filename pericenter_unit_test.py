from prepWork import orbit
from radius import radius
import numpy as np
import pytest

#circular radius
a = 1
e = 0
times = np.arange(0, 10e3, 1)
expected_circular = radius(a, e)
orbit_circular = orbit(radius.radius)

expected_radius = expected_circular.radius(times)

