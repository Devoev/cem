import numpy as np

from fem.mesh.geo import triangle_area


def mass_node_local(nodes: np.ndarray) -> np.ndarray:
    """
    Creates the local mass matrix of nodal basis functions.
    :param nodes: The triangle nodes in 2D. Matrix of size 2x3.
    :return: Local 3x3 mass matrix.
    """

    val = triangle_area(nodes) / 12
    return np.array([
        [2*val, val, val],
        [val, 2*val, val],
        [val, val, 2*val]
    ])


def mass_vol_local(nodes: np.ndarray) -> float:
    """
    Creates the local mass matrix of volume basis functions.
    :param nodes: Triangle nodes in 2D. Matrix of size 2x3.
    :return: Local scalar mass matrix.
    """

    return 1/triangle_area(nodes)
