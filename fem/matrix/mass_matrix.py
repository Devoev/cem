from typing import List, Callable

import numpy as np
import scipy.sparse as sp

from fem.matrix.build_nodal_mat import build_nodal_mat
from fem.mesh.mesh_2d import Mesh2D
from util.geo import area_triangle_2d
from util.quadrature import int_triangle_2d


def mass_node(msh: Mesh2D) -> sp.coo_matrix:
    """
    Creates the global mass matrix of nodal basis functions.
    :param msh: Mesh object.
    :return: Global mass matrix.
    """

    return build_nodal_mat(msh, lambda e: mass_node_local(msh.elems[e].T))


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
    :param nodes: The triangle nodes in 2D. Matrix of size ``(3,2)``.
    :return: Local ``(3,3)`` mass matrix.
    """

    a0, a1, a2 = nodes
    nodes_ref = np.array([[0,0],[1,0],[0,1]]).T
    J = np.vstack([a1 - a0, a2 - a0])               # Jacobian
    J_inv = np.linalg.inv(J.T)                      # Inverse transpose Jacobian
    grad_b = np.array([[-1, -1], [1, 0], [0, 1]])   # Gradients of nodal basis in reference triangle
    grad_b_ref = J_inv @ grad_b.T                   # Gradients on reference triangle

    b: List[Callable[[np.ndarray], float]] = [lambda p: 1 - p[0] - p[1], lambda p: p[0], lambda p: p[1]]
    mat = np.zeros((3,3))
    for i in range(3):
        for j in range(3):
            def bij(l: int, p: np.ndarray) -> np.ndarray:
                k = (l+1) % 3
                return b[l](p) * grad_b_ref[:,k] - b[k](p) * grad_b_ref[:,l]

            def fun(p: np.ndarray) -> float:
                return np.dot(bij(i, p), bij(j, p))

            mat[i,j] = int_triangle_2d(fun, nodes_ref)

    return mat


def mass_vol_local(nodes: np.ndarray) -> float:
    """
    Creates the local mass matrix of volume basis functions.
    :param nodes: Triangle nodes in 2D. Matrix of size ``(2,3)``.
    :return: Local scalar mass matrix.
    """

    return 1/area_triangle_2d(nodes)
