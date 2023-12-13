from typing import Callable, Tuple

import numpy as np

ndfun = Callable[[np.ndarray], float]
"""Function from R^n -> R."""


def basis_node_ref() -> Tuple[ndfun, ndfun, ndfun]:
    """Returns the three nodal basis functions on the reference triangle."""
    return (lambda p: 1 - p[0] - p[1],
            lambda p: p[0],
            lambda p: p[1])


def basis_node_ref_grad() -> np.ndarray:
    """Returns the constant gradients of nodal basis functions on the reference triangle. Matrix of size ``(2,3)``."""
    return np.array([[-1, -1], [1, 0], [0, 1]]).T
