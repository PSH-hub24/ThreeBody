import numpy as np
def center_cluster(cluster):
    """
    Adjusts the cluster's center of mass and ensures the black hole is centered at the origin.
    Also adjusts velocities to be relative to the center of mass and ensures angular momentum consistency.

    Parameters:
    cluster (list): List of Body objects representing the cluster, where the first body is the central black hole.

    Returns:
    list: Updated list of Body objects with corrected positions and velocities.
    """
    # Center the black hole (first body) at the origin
    cluster[0].position = np.array([0.0, 0.0, 0.0])
    cluster[0].velocity = np.array([0.0, 0.0, 0.0])

    # Compute total mass
    total_mass = sum(body.mass for body in cluster)

    # Compute center of mass position and velocity
    center_of_mass_pos = np.zeros(3)
    center_of_mass_vel = np.zeros(3)
    for body in cluster:
        center_of_mass_pos += body.mass * body.position
        center_of_mass_vel += body.mass * body.velocity
    center_of_mass_pos /= total_mass
    center_of_mass_vel /= total_mass

    # Adjust positions and velocities relative to the center of mass
    for body in cluster:
        body.position -= center_of_mass_pos
        body.velocity -= center_of_mass_vel

    # Ensure angular momentum consistency for the cluster
    for body in cluster[1:]:  # Exclude the central black hole
        angular_momentum_z = body.position[0] * body.velocity[1] - body.position[1] * body.velocity[0]
        if angular_momentum_z <= 0:
            body.velocity = -body.velocity

    return cluster
