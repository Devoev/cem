import numpy as np
import scipy.sparse as sp

from fem.mesh.mesh_2d import Mesh2D
from util.geo import area_triangle_2d


def mass_node(msh: Mesh2D) -> sp.spmatrix:
    """
    Creates the global mass matrix of nodal basis functions.
    :param msh: Mesh object.
    :return: Global mass matrix.
    """

    n = 9 * msh.E                       # Amount of matrix entries
    N = msh.N                           # Matrix dimension
    vals = np.zeros(n)                  # Nonzero values of tge matrix
    rows = np.zeros(n, dtype='int')     # Row indices for the entries
    cols = np.zeros(n, dtype='int')     # Column indices for the entries

    for e in range(msh.E):
        idx = msh.elems_to_nodes[e]
        idx_e = np.arange(9*e,9*(e+1))

        rows[idx_e] = np.repeat(idx, 3)
        cols[idx_e] = np.reshape([idx, idx, idx], 9)

        vals[idx_e] = mass_node_local(msh.elems[e].T).flatten()

    return sp.coo_matrix((vals, (rows, cols)), shape=(N, N))


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
