from typing import Callable

import numpy as np

from fem.mesh.mesh_2d import Mesh2D
import scipy.sparse as sp


def build_nodal_mat(msh: Mesh2D, build_local_mat: Callable[[int], np.ndarray]) -> sp.coo_matrix:
    """
    Creates a global matrix of nodal basis functions.
    :param msh: Mesh object.
    :param build_local_mat: Builder for the local nodal matrix.
    :return: Global matrix of size ``(N,N)``.
    """

    return build_mat(msh, msh.N, lambda t: msh.elems_to_nodes[t], build_local_mat)


def build_edge_mat(msh: Mesh2D, build_local_mat: Callable[[int], np.ndarray]) -> sp.coo_matrix:
    """
    Creates a global matrix of edge basis functions.
    :param msh: Mesh object.
    :param build_local_mat: Builder for the local edge matrix.
    :return: Global matrix of size ``(E,E)``.
    """

    return build_mat(msh, msh.E, lambda t: msh.find_edges_by_elem(t), build_local_mat)


def build_mat(msh: Mesh2D,
              n: int,
              find_idx_by_elem: Callable[[int], np.ndarray],
              build_local_mat: Callable[[int], np.ndarray]) -> sp.coo_matrix:
    """
    Creates a global matrix by iterating over all local element matrices.
    :param msh: Mesh object.
    :param n: Dimension of matrix.
    :param find_idx_by_elem: Function to find the local indices for the given element.
    :param build_local_mat: Builder for the local element matrix.
    :return: Global matrix of size ``(n,n)``.
    """

    m = 9 * msh.T                       # Amount of matrix entries
    vals = np.zeros(m)                  # Nonzero values of tge matrix
    rows = np.zeros(m, dtype='int')     # Row indices for the entries
    cols = np.zeros(m, dtype='int')     # Column indices for the entries

    for t in range(msh.T):
        global_idx = find_idx_by_elem(t)
        local_idx = np.arange(9*t,9*(t+1))

        rows[local_idx] = np.repeat(global_idx, 3)
        cols[local_idx] = np.reshape([global_idx, global_idx, global_idx], 9)
        vals[local_idx] = build_local_mat(t).flatten()

    return sp.coo_matrix((vals, (rows, cols)), shape=(n,n))
