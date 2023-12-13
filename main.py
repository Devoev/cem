import matplotlib.pyplot as plt
import numpy as np

from capacitor.generate_mesh import generate_mesh
from fem.matrix.load_vector import load_node_local
from fem.matrix.mass_matrix import mass_node, mass_edge_local, mass_node_local, mass_vol_local
from fem.matrix.stiffness_matrix import stiffness_node
from fem.mesh.mesh_2d import make_mesh

if __name__ == '__main__':
    generate_mesh(8, 10, 0, 15)
    msh = make_mesh()

    nodes = np.array([[0,0], [1,0], [0,1]])
    # nodes = np.array([[0,0], [2,0], [2,6]]).T

    load_vec = load_node_local(lambda p: 1, nodes, 1)
    print(np.array([[-1, -1], [1, 0], [0, 1]]).T.shape)

    m_node = mass_node_local(nodes)
    m_edge = mass_edge_local(nodes)
