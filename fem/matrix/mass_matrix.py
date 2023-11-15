import numpy as np

from fem.mesh.geo import triangle_area


def mass_node_local(nodes: np.ndarray) -> np.ndarray:
    """
    Creates the local mass matrix of nodal basis functions.
    :param nodes: The triangle nodes. Matrix of size 2x3.
    :return: Local 3x3 mass matrix.
    """

    val = triangle_area(nodes) / 12
    return np.full((3, 3), val) + np.diag([val, val, val])
