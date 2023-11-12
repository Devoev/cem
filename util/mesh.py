import sys
from functools import wraps
from typing import Dict, Final

import gmsh


def model(name: str, dim: int = 3, show_gui: bool = False, finalize: bool = False, options: Dict[str, float] = None):
    """Indicates that the function generates a gmsh model.

    :param name The name of the gmsh model.
    :param dim: Dimension of the mesh.
    :param show_gui: Whether to show the gui.
    :param finalize: Whether to finalize the gmsh API.
    :param options: Options that get passed to 'gmsh.option.set_number'.
    """

    if options is None:
        options = dict()

    def _model(func):

        @wraps(func)
        def wrapper(*args, **kwargs):
            gmsh.initialize()

            for option, value in options.items():
                gmsh.option.set_number(option, value)

            gmsh.model.add(name)

            res = func(*args, **kwargs)

            gmsh.model.occ.synchronize()
            gmsh.model.mesh.generate(dim)

            if '-nopopup' not in sys.argv and show_gui:
                gmsh.fltk.run()

            if finalize:
                gmsh.finalize()

            return res

        return wrapper

    return _model
