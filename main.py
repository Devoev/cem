import matplotlib.pyplot as plt
import numpy as np

from capacitor.generate_mesh import generate_mesh
from fem.basis.basis_node import basis_node_ref_grad
from fem.matrix.load_vector import load_node_local, load_node
from fem.matrix.mass_matrix import mass_node, mass_edge_local, mass_node_local, mass_vol_local, mass_edge
from fem.matrix.stiffness_matrix import stiffness_node
from fem.mesh.mesh_2d import make_mesh
from util.geo import jacobian

if __name__ == '__main__':
    generate_mesh(8, 10, 0, 15)
    msh = make_mesh()

    elems = msh.edges_to_elems
    bnd = msh.edges_bnd
    bnd_n = msh.nodes_bnd
