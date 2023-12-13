import matplotlib.pyplot as plt
import numpy as np

from capacitor.generate_mesh import generate_mesh
from fem.matrix.mass_matrix import mass_node, mass_edge_local, mass_node_local, mass_vol_local
from fem.matrix.stiffness_matrix import stiffness_node
from fem.mesh.mesh_2d import make_mesh

if __name__ == '__main__':
    generate_mesh(8, 10, 0, 15)
    msh = make_mesh()

    # nodes = np.array([[0,1,0], [0,0,1]])
    nodes = np.array([[0,0], [2,0], [2,6]]).T
    mass_s1 = mass_node_local(nodes)
    mass_r1 = mass_edge_local(nodes.T)
    mass_q0 = mass_vol_local(nodes)

    # M = mass_node(msh).toarray()
    # K = stiffness_node(msh).toarray()
    # plt.spy(M, markersize=0.1)
    # plt.figure()
    # plt.spy(K, markersize=0.1)
    # plt.show()
    #
    # print(M.shape)
    # print(np.linalg.matrix_rank(M))
