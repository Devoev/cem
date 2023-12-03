import numpy as np
import scipy.sparse as sp

from fem.matrix.build_nodal_mat import build_nodal_mat
from fem.mesh.mesh_2d import Mesh2D
from util.geo import area_triangle_2d


def stiffness_node(msh: Mesh2D) -> sp.spmatrix:
    """
    Creates the global stiffness matrix of nodal basis functions.
    :param msh: Mesh object.
    :return: Global stiffness matrix.
    """

    return build_nodal_mat(msh, lambda e: stiffness_node_local(msh.elems[e].T))


def stiffness_node_local(nodes: np.ndarray) -> np.ndarray:
    """
    Creates the local stiffness matrix of nodal basis functions.
    :param nodes: The triangle nodes in 2D. Matrix of size 2x3.
    :return: Local 3x3 mass matrix.
    """

    a0, a1, a2 = nodes.T
    S = area_triangle_2d(nodes)  # Triangle area = jacobian determinant
    J = np.vstack([a1 - a0, a2 - a0])  # Jacobian
    G_inv = np.linalg.inv(J.T @ J)  # Inverse gram matrix
    grad_b = np.array([[-1, -1], [1, 0], [0, 1]]).T  # Gradients of nodal basis in reference triangle

    return S * grad_b.T @ G_inv @ grad_b
