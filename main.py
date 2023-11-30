import gmsh
import numpy as np

from capacitor.generate_mesh import generate_mesh
from fem.matrix import mass_matrix
from fem.matrix.mass_matrix import mass_node_local, mass_node
from fem.matrix.stiffness_matrix import stiffness_node_local
from fem.mesh.mesh import create_mesh

if __name__ == '__main__':
    generate_mesh(8, 10, 0, 15)
    msh = create_mesh()
    mass_node(msh)
