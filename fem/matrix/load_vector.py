from typing import Callable, List

import numpy as np

from fem.mesh.mesh_2d import Mesh2D
from util.geo import area_triangle_2d
from util.quadrature import int_ref_triangle_2d


def load_node(msh: Mesh2D, f: Callable[[np.ndarray], float], q: int):
    """
    Creates the global load vector of nodal basis functions.
    :param msh: Mesh object.
    :param f: Right hand side function.
    :param q: Order of integration.
    :return: Global load vector.
    """
    pass


def load_node_local(f: Callable[[np.ndarray], float], nodes: np.ndarray, q: int):
    """
    Creates the local load vector of nodal basis functions.
    :param f: Right hand side function.
    :param nodes: The triangle nodes in 2D. Matrix of size ``(2,3)``.
    :param q: Order of integration.
    :return: Local load vector of size ``(3)``.
    """

    a0, a1, a2 = nodes
    J = np.vstack([a1 - a0, a2 - a0]).T
    S = area_triangle_2d(nodes.T)
    b: List[Callable[[np.ndarray], float]] = [lambda p: 1 - p[0] - p[1], lambda p: p[0], lambda p: p[1]]
    return 2 * S * np.array([int_ref_triangle_2d(lambda p: f(a0 + J @ p) * b[i](p), q) for i in range(3)])
