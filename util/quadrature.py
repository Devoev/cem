from typing import Callable, Tuple

import numpy as np

from util.geo import area_triangle_2d


def int_triangle_2d(fun: Callable[[np.ndarray], float], nodes: np.ndarray) -> float:
    """
    Computes the integral on a triangle of the given ``fun`` by using a 2nd order quadrature rule.
    :param fun: Function to integrate.
    :param nodes: Triangle nodes in 2D. Matrix of size ``(3,2)``.
    :return: Value of the integral.
    """

    a0, a1, a2 = nodes
    m1 = (a0 + a1) / 2
    m2 = (a1 + a2) / 2
    m3 = (a2 + a0) / 2
    S = area_triangle_2d(nodes)
    return S / 3 * (fun(m1) + fun(m2) + fun(m3))


def int_ref_triangle_2d(fun: Callable[[np.ndarray], float], q: int) -> float:
    """
    Computes the integral on the reference triangle using `Gauss` quadrature.
    :param fun: Function to integrate.
    :param q: Order of quadrature.
    :return: Value of the integral.
    """

    x, w = gauss_quad_ref_triangle_2d(q)
    return sum([w[i] * fun(x[i]) for i in range(w.size)])


def gauss_quad_ref_triangle_2d(q: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the knots ``x`` and weights ``w`` of the `Gauss` quadrature on the reference triangle.
    :param q: Order of quadrature.
    :return: Knots ``x`` and weights ``w``
    """

    if q == 1:
        x = [[-1 / 3, -1 / 3]]
        w = [2]
    elif q == 2:
        x = [[-2 / 3, -2 / 3], [-2 / 3, 1 / 3], [1 / 3, -2 / 3]]
        w = [2 / 3, 2 / 3, 2 / 3]
    elif q == 3:
        x = [[-1 / 3, -1 / 3], [-0.6, -0.6], [-0.6, 0.2], [0.2, -0.6]]
        w = [-1.125, 1.041666666666667, 1.041666666666667, 1.041666666666667]
    elif q == 4:
        x = [[-0.108103018168070, -0.108103018168070],
             [-0.108103018168070, -0.783793963663860],
             [-0.783793963663860, -0.108103018168070],
             [-0.816847572980458, -0.816847572980458],
             [-0.816847572980458, 0.633695145960918],
             [0.633695145960918, -0.816847572980458]]
        w = [0.446763179356022,
             0.446763179356022,
             0.446763179356022,
             0.219903487310644,
             0.219903487310644,
             0.219903487310644]
    elif q == 5:
        x = [[-0.333333333333333, -0.333333333333333],
             [-0.059715871789770, -0.059715871789770],
             [-0.059715871789770, -0.880568256420460],
             [-0.880568256420460, -0.059715871789770],
             [-0.797426985353088, -0.797426985353088],
             [-0.797426985353088, 0.594853970706174],
             [0.594853970706174, -0.797426985353088]],
        w = [0.450000000000000,
             0.264788305577012,
             0.264788305577012,
             0.264788305577012,
             0.251878361089654,
             0.251878361089654,
             0.251878361089654]
    else:
        raise Exception(f"numerical integration of order {q} not implemented")

    return (np.array(x) + 1) / 2, np.array(w) / 4
