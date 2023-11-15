import numpy as np

from util.geo import area_triangle_2d


def stiffness_node_local(nodes: np.ndarray) -> np.ndarray:
    """
    Creates the local stiffness matrix of nodal basis functions.
    :param nodes: The triangle nodes in 2D. Matrix of size 2x3.
    :return: Local 3x3 mass matrix.
    """

    a0, a1, a2 = nodes.T
    S = area_triangle_2d(nodes)
    J = np.vstack([a1 - a0, a2 - a0])                   # Jacobian
    G_inv = np.linalg.inv(J.T @ J)                      # Inverse gram matrix
    grad_b = np.array([[-1, -1], [1, 0], [0, 1]]).T     # Gradients of nodal basis in reference triangle

    return S * grad_b.T @ G_inv @ grad_b
