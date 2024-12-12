from body import Body

def figure_eight_orbit():
    """Three-body figure-eight orbit."""
    mass = 1.0
    positions = [
        [0.97000436, -0.24308753, 0.0],
        [-0.97000436, 0.24308753, 0.0],
        [0.0, 0.0, 0.0]
    ]
    velocities = [
        [0.466203685, 0.43236573, 0.0],
        [0.466203685, 0.43236573, 0.0],
        [-0.93240737, -0.86473146, 0.0]
    ]
    return [Body(position=positions[i], velocity=velocities[i], mass=mass) for i in range(3)]
