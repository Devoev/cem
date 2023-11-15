import numpy as np

from capacitor.generate_mesh import generate_mesh
from fem.matrix import mass_matrix
from fem.matrix.mass_matrix import mass_node_local

if __name__ == '__main__':
    # generate_mesh(8, 10, 0, 15)

    nodes = np.array([[1, 0], [2, 2], [2, 6]]).T
    mat = mass_node_local(nodes)
