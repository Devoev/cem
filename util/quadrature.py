from typing import Callable

import numpy as np

from util.geo import area_triangle_2d


def int_triangle_2d(fun: Callable[[np.ndarray], float], nodes: np.ndarray) -> float:
    """
    Computes the integral on a triangle of the given ``fun`` by using a 2nd order quadrature rule.
    :param fun: Function to integrate.
    :param nodes: Triangle nodes in 2D.
    :return: Value of the integral.
    """

    a0, a1, a2 = nodes.T
    m1 = (a0 + a1)/2
    m2 = (a1 + a2)/2
    m3 = (a2 + a0)/2
    S = area_triangle_2d(nodes)
    return S/3 * (fun(m1) + fun(m2) + fun(m3))
