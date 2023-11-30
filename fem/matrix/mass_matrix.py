import numpy as np
import scipy.sparse as sp

from util.geo import area_triangle_2d


def mass_node(nodes: np.ndarray, triangles: np.ndarray) -> sp.spmatrix:
    """
    Creates the global mass matrix of nodal basis functions.
    :param nodes: Mesh nodes in 2D. Matrix of size ``(np,2)``
    :param triangles: Triangle node indices. Matrix of size ``(nt,3)``.
    :return:
    """
    pass


def mass_node_local(nodes: np.ndarray) -> np.ndarray:
    """
    Creates the local mass matrix of nodal basis functions.
    :param nodes: The triangle nodes in 2D. Matrix of size ``(2,3)``.
    :return: Local ``(3,3)`` mass matrix.
    """

    val = area_triangle_2d(nodes) / 12
    return np.array([
        [2*val, val, val],
        [val, 2*val, val],
        [val, val, 2*val]
    ])


def mass_edge_local(nodes: np.ndarray) -> np.ndarray:
    """
    Creates the local mass matrix of edge basis functions.
    :param nodes: The triangle nodes in 2D. Matrix of size ``(2,3)``.
    :return: Local ``(3,3)`` mass matrix.
    """
    pass


def mass_vol_local(nodes: np.ndarray) -> float:
    """
    Creates the local mass matrix of volume basis functions.
    :param nodes: Triangle nodes in 2D. Matrix of size ``(2,3)``.
    :return: Local scalar mass matrix.
    """

    return 1/area_triangle_2d(nodes)
