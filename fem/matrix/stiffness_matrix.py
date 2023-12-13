import numpy as np
import scipy.sparse as sp

from fem.basis.basis_node import basis_node_ref_grad
from fem.matrix.build_mat import build_nodal_mat
from fem.mesh.mesh_2d import Mesh2D
from util.geo import area_triangle_2d, gram_inv


def stiffness_node(msh: Mesh2D) -> sp.coo_matrix:
    """
    Creates the global stiffness matrix of nodal basis functions.
    :param msh: Mesh object.
    :return: Global stiffness matrix.
    """

    return build_nodal_mat(msh, lambda e: stiffness_node_local(msh.elems[e]))


def stiffness_node_local(nodes: np.ndarray) -> np.ndarray:
    """
    Creates the local stiffness matrix of nodal basis functions.
    :param nodes: The triangle nodes in 2D. Matrix of size ``(3,2)``.
    :return: Local ``(3x3)`` mass matrix.
    """

    S = area_triangle_2d(nodes)
    G_inv = gram_inv(nodes)
    grad_b = basis_node_ref_grad()

    return S * grad_b.T @ G_inv @ grad_b
