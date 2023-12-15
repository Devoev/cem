from matplotlib import pyplot as plt

from fem.mesh.mesh_2d import Mesh2D


def plot_mesh(msh: Mesh2D, xlabel='$x$ (m)', ylabel='$y$ (m)', title=''):
    """Plots the given ``msh`` as a ``triplot``."""

    plt.triplot(msh.x, msh.y, msh.elems_to_nodes)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.axis('scaled')
    plt.show()
