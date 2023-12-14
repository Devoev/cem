from typing import Callable, List

import numpy as np
import scipy.sparse as sp

from fem.mesh.mesh_2d import Mesh2D
from util.geo import area_triangle_2d
from util.quadrature import int_ref_triangle_2d


def load_node(msh: Mesh2D, f: Callable[[np.ndarray], float], q: int) -> np.ndarray:
    """
    Creates the global load vector of nodal basis functions.
    :param msh: Mesh object.
    :param f: Right hand side function.
    :param q: Order of integration.
    :return: Global load vector of size ``(N)``.
    """

    m = 3 * msh.T                       # Amount of matrix entries
    vals = np.zeros(m)                  # Nonzero values of tge matrix
    rows = np.zeros(m, dtype='int')     # Row indices for the entries
    cols = np.zeros((m,))               # Column indices for the entries

    for t in range(msh.T):
        global_idx = msh.elems_to_nodes[t]
        local_idx = np.arange(3*t,3*(t+1))

        rows[local_idx] = global_idx
        vals[local_idx] = load_node_local(f, msh.elems[t], q)

    return sp.coo_matrix((vals, (rows,cols)), shape=(msh.N,1)).toarray()[:,0]


def load_node_local(f: Callable[[np.ndarray], float], nodes: np.ndarray, q: int):
    """
    Creates the local load vector of nodal basis functions.
    :param f: Right hand side function.
    :param nodes: The triangle nodes in 2D. Matrix of size ``(3,2)``.
    :param q: Order of integration.
    :return: Local load vector of size ``(3)``.
    """

    a0, a1, a2 = nodes
    J = np.vstack([a1 - a0, a2 - a0]).T
    S = area_triangle_2d(nodes)
    b: List[Callable[[np.ndarray], float]] = [lambda p: 1 - p[0] - p[1], lambda p: p[0], lambda p: p[1]]
    return 2 * S * np.array([int_ref_triangle_2d(lambda p: f(a0 + J @ p) * b[i](p), q) for i in range(3)])
