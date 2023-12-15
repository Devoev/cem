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

    p = msh.find_node(np.array([0,0]))
    print(p)

    # # nodes = np.array([[0,0], [1,0], [0,1]])
    # nodes = np.array([[0,0], [2,0], [2,6]])
    #
    # J = jacobian(nodes)
    # grad_b = basis_node_ref_grad()
    #
    # load_vec = load_node_local(lambda p: 1, nodes, 1)
    # m_node_l = mass_node_local(nodes)
    # m_edge_l = mass_edge_local(nodes)

    # m_node = mass_node(msh)
    # m_edge = mass_edge(msh)
    #
    # plt.figure()
    # plt.spy(m_node, markersize=0.1)
    # plt.show()
    # plt.figure()
    # plt.spy(m_edge, markersize=0.1)
    # plt.show()

    # f = load_node(msh, lambda p: p[0] + p[1], 1)
    # print(f.shape)
