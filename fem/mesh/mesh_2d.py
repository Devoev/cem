from dataclasses import dataclass
from functools import cached_property
from typing import Tuple

import gmsh
import numpy as np


@dataclass
class Mesh2D:
    """2-dimensional triangular mesh. Nodes are index from ``0:N-1`` and elements from ``0:T-1``."""

    nodes: np.ndarray
    """Node coordinate matrix. Array of size ``(N,2)``."""

    elems_to_nodes: np.ndarray
    """Element to node connection matrix. Array of size ``(T,3)``."""

    def __post_init__(self):
        self.N = self.nodes.shape[0]
        self.E = self.edges_to_nodes.shape[0]
        self.T = self.elems_to_nodes.shape[0]

    @cached_property
    def x(self):
        """``x``-coordinates of nodes."""
        return self.nodes[:, 0]

    @cached_property
    def y(self):
        """``y``-coordinates of nodes."""
        return self.nodes[:, 1]

    def find_node(self, p: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Finds the nearest node to the given node ``p``

        :param p: Node coordinates. Array of size ``2``.
        :return: Node coordinates and node index.
        """
        node = np.argmin(np.linalg.norm(self.nodes - p, axis=1))
        return self.nodes[node], node

    @cached_property
    def elems(self):
        """Element coordinate array of size ``(T,3,2)``."""
        return self.nodes[self.elems_to_nodes]

    @cached_property
    def edges_to_nodes(self):
        """Edge to node connection matrix. Array of size ``(E,2)``."""
        edges = np.concatenate([
            self.elems_to_nodes[:, [0, 1]],
            self.elems_to_nodes[:, [1, 2]],
            self.elems_to_nodes[:, [2, 0]]
        ])
        return np.unique(np.sort(edges), axis=0)

    @cached_property
    def edges(self):
        """Edge coordinate array of size ``(E,2,2)``."""
        return self.nodes[self.edges_to_nodes]

    def find_edge_by_nodes(self, n1: int, n2: int) -> int:
        """
        Finds the index of the edge from node ``n1`` to node ``n2``. Automatically sorts nodes in increasing order.
        :return: Index of the edge. If edge doesn't exist ``-1``.
        """

        edge = np.argwhere((sorted((n1, n2)) == self.edges_to_nodes).all(axis=1))
        return -1 if edge.size == 0 else edge[0, 0]

    def find_edges_by_elem(self, t: int):
        """
        Finds the edges of the element ``t``.
        :return: Indices of elements.
        """

        n1, n2, n3 = self.elems_to_nodes[t]
        return self.find_edge_by_nodes(n1, n2), self.find_edge_by_nodes(n2, n3), self.find_edge_by_nodes(n3, n1)


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
    T = int(elems_to_notes.size / 3)
    elems_to_notes = np.reshape(elems_to_notes, (T, 3))

    return Mesh2D(nodes, elems_to_notes)
