import matplotlib.pyplot as plt
import numpy as np

from capacitor.generate_mesh import generate_mesh
from fem.matrix.mass_matrix import mass_node
from fem.matrix.stiffness_matrix import stiffness_node
from fem.mesh.mesh_2d import make_mesh

if __name__ == '__main__':
    generate_mesh(8, 10, 0, 15)
    msh = make_mesh()

    # M = mass_node(msh).toarray()
    # K = stiffness_node(msh).toarray()
    # plt.spy(M, markersize=0.1)
    # plt.figure()
    # plt.spy(K, markersize=0.1)
    # plt.show()
    #
    # print(M.shape)
    # print(np.linalg.matrix_rank(M))
