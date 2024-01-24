import matplotlib.pyplot as plt
import numpy as np

from capacitor.mesh import gen_capacitor
from fem.basis.basis_node import basis_node_ref_grad
from fem.matrix.load_vector import load_node_local, load_node
from fem.matrix.mass_matrix import mass_node, mass_edge_local, mass_node_local, mass_vol_local, mass_edge
from fem.matrix.stiffness_matrix import stiffness_node
from fem.mesh.mesh_2d import make_mesh
from util.geo import jacobian

if __name__ == '__main__':
    gen_capacitor(8, 10, 0, 15)
    msh = make_mesh()
