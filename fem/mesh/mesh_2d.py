from dataclasses import dataclass
from functools import cached_property

import gmsh
import numpy as np


@dataclass
class Mesh2D:
    """2-dimensional triangular mesh. Nodes are index from ``0:N-1`` and elements from ``0:E-1``."""

    nodes: np.ndarray
    """Node coordinate matrix. Array of size ``(N,2)``"""

    elems_to_nodes: np.ndarray
    """Element to node connection matrix. Array of size ``(E,3)``"""

    def __post_init__(self):
        self.N = self.nodes.shape[0]
        self.E = self.elems_to_nodes.shape[0]

    @cached_property
    def elems(self):
        """Element coordinate array of size ``(E,3,2)``"""
        return self.nodes[self.elems_to_nodes]


def make_mesh() -> Mesh2D:
    """Creates an instance of a ``Mesh2D`` object using the currently active ``gmsh`` instance."""

    msh = gmsh.model.mesh

    # Nodes
    _, nodes, _ = msh.get_nodes()
    N = int(nodes.size / 3)
    nodes = np.reshape(nodes, (N, 3))[:, 0:2]

    # Elems
    element_types, _, node_tags_elements = msh.get_elements()
    idx = np.where(element_types == 2)[0][0]  # Index of triangle elements
    elems_to_notes = node_tags_elements[idx] - 1  # Get correct elements and shift 1 -> 0
    E = int(elems_to_notes.size / 3)
    elems_to_notes = np.reshape(elems_to_notes, (E, 3))

    return Mesh2D(nodes, elems_to_notes)