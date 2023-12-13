from typing import Callable, Tuple

import numpy as np

from fem.basis.basis_node import basis_node_ref, basis_node_ref_grad

ndfun = Callable[[np.ndarray], np.ndarray]
"""Function from R^n -> R^n."""


def basis_edge_ref() -> Tuple[ndfun, ndfun, ndfun]:
    """Returns the three edge basis functions on the reference triangle."""
    b1, b2, b3 = basis_node_ref()
    grad_b1, grad_b2, grad_b3 = basis_node_ref_grad().T
    return (lambda p: b1(p) * grad_b2 - b2(p) * grad_b1,
            lambda p: b2(p) * grad_b3 - b3(p) * grad_b2,
            lambda p: b3(p) * grad_b1 - b1(p) * grad_b3)


def basis_edge_ref_curl() -> np.ndarray:
    """Returns the constant curl of nodal basis functions on the reference triangle."""
    pass  # TODO: add implementation
