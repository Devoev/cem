from typing import Callable

import numpy as np

from fem.mesh.mesh_2d import Mesh2D
import scipy.sparse as sp


def build_nodal_mat(msh: Mesh2D, build_local_mat: Callable[[int], np.ndarray]) -> sp.coo_matrix:
    """
    Creates a global matrix of nodal basis functions.
    :param msh: Mesh object.
    :param build_local_mat: Builder for the local nodal matrix.
    :return: Global matrix.
    """

    n = 9 * msh.T                       # Amount of matrix entries
    N = msh.N                           # Matrix dimension
    vals = np.zeros(n)                  # Nonzero values of tge matrix
    rows = np.zeros(n, dtype='int')     # Row indices for the entries
    cols = np.zeros(n, dtype='int')     # Column indices for the entries

    for e in range(msh.T):
        nodes = msh.elems_to_nodes[e]
        idx = np.arange(9*e,9*(e+1))

        rows[idx] = np.repeat(nodes, 3)
        cols[idx] = np.reshape([nodes, nodes, nodes], 9)
        vals[idx] = build_local_mat(e).flatten()

    return sp.coo_matrix((vals, (rows, cols)), shape=(N, N))
