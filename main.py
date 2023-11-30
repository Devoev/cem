import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.csgraph import structural_rank

from capacitor.generate_mesh import generate_mesh
from fem.matrix.mass_matrix import mass_node
from fem.mesh.mesh import create_mesh

if __name__ == '__main__':
    generate_mesh(8, 10, 0, 15)
    msh = create_mesh()
    M = mass_node(msh).toarray()
    plt.spy(M, markersize=0.1)
    plt.show()

    print(M.shape)
    print(np.linalg.matrix_rank(M))
