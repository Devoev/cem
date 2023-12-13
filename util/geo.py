import numpy as np


def area_triangle_2d(nodes: np.ndarray) -> float:
    """
    Computes the area of a triangle in 2D.
    :param nodes: Nodes of the triangle. Matrix of size ``(3,2)``.
    :return: Area of the triangle.
    """

    d = np.diff(nodes.T)
    return 0.5 * abs(d[0, 0] * d[1, 1] - d[1, 0] * d[0, 1])


def jacobian(nodes: np.ndarray) -> np.ndarray:
    """
    Returns the jacobian matrix of a triangle in 2D.
    :param nodes: Nodes of the triangle. Matrix of size ``(3,2)``.
    :return: Matrix of size ``(2,2)``.
    """
    a0, a1, a2 = nodes
    return np.stack([a1 - a0, a2 - a0], axis=-1)


def gram_inv(nodes: np.ndarray) -> np.ndarray:
    """
    Returns the inverse of the gram matrix of a triangle in 2D
    :param nodes: Nodes of the triangle. Matrix of size ``(3,2)``.
    :return: Matrix of size ``(2,2)``.
    """
    J = jacobian(nodes)
    return np.linalg.inv(J.T @ J)
