import numpy as np

from fem.mesh.geo import triangle_area


def mass_node_local(nodes: np.ndarray) -> np.ndarray:
    """
    Creates the local mass matrix of nodal basis functions.
    :param nodes: The triangle nodes. Matrix of size 2x3.
    :return: Local 3x3 mass matrix.
    """

    S = triangle_area(nodes)
    mat = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            mat[i, j] = S / 6 if i == j else S / 12

    return mat
