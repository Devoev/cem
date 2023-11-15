import numpy as np


def area_triangle_2d(nodes: np.ndarray) -> float:
    """
    Computes the area of a triangle in 2D.
    :param nodes: Nodes of the triangle.
    :return:
    """

    d1 = nodes[:, 0] - nodes[:, 1]
    d2 = nodes[:, 1] - nodes[:, 2]
    return 0.5 * abs(d1[0] * d2[1] - d1[1] * d2[0])
