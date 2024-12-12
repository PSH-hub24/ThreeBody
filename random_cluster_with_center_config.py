from random_cluster_configuration_config import random_cluster_configuration
from center_cluster_config import center_cluster
from body import Body

def random_cluster_with_center(num_particles):
    """Random cluster configuration with a central massive object."""
    central_mass = 200.0  # Define central mass

    # Central body
    central_body = Body(position=[0, 0, 0], velocity=[0, 0, 0], mass=central_mass)

    # Cluster particles
    cluster_particles = random_cluster_configuration(num_particles, central_mass)

    # Combine central body with cluster particles
    cluster = [central_body] + cluster_particles

    # Apply center-of-mass correction and angular momentum consistency
    cluster = center_cluster(cluster)

    return cluster
