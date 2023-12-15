import numpy as np
from matplotlib import pyplot as plt

from fem.mesh.mesh_2d import Mesh2D


def plot_pot(msh: Mesh2D, pot: np.ndarray, xlabel='$x$ (m)', ylabel='$y$ (m)', title='', **kwargs):
    """
    Plots the given scalar potential ``pot`` as a ``tripcolor`` plot.
    :param msh: Mesh struct
    :param pot: Potential array of size (msh.N).
    :param xlabel: Label of x-axis.
    :param ylabel: Label of y-axis.
    :param title: Title of the plot.
    """

    plt.tripcolor(msh.x, msh.y, msh.elems_to_nodes, pot, **kwargs)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.axis('scaled')
    plt.colorbar()
    plt.show()
