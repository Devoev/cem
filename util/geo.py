import numpy as np


def area_triangle_2d(nodes: np.ndarray) -> float:
    """
    Computes the area of a triangle in 2D.
    :param nodes: Nodes of the triangle. Matrix of size ``(3,2)``.
    :return: Area of the triangle.
    """

    d = np.diff(nodes.T)
    return 0.5 * abs(d[0,0]*d[1,1] - d[1,0]*d[0,1])
