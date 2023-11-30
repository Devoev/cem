import numpy as np
import scipy.sparse as sp

from fem.mesh.mesh import Mesh
from util.geo import area_triangle_2d


def mass_node(msh: Mesh) -> sp.spmatrix:
    """
    Creates the global mass matrix of nodal basis functions.
    :param msh: Mesh object.
    :return: Global mass matrix.
    """

    n = msh.num_elems * 9               # Amount of matrix entries
    m = msh.num_node                    # Dimension of Knu matrix
    mat = np.zeros(n)                   # Nonzero entries of the Knu matrix
    rows = np.zeros(n, dtype='int')     # Row indices for the entries
    cols = np.zeros(n, dtype='int')     # Column indices for the entries

    for e in range(msh.num_elems):
        idx = msh.elems[e]
        idx_e = np.arange(9*e,9*(e+1))
        rows[idx_e] = np.repeat(idx, 3)
        cols[idx_e] = np.reshape([idx, idx, idx], 9)
        mat[idx_e] = mass_node_local(msh.elem_nodes[idx_e]).flatten()

    return sp.csr_matrix((mat, (rows, cols)), shape=(m, m))


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
