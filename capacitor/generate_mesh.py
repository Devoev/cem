import gmsh

from util.mesh import model

gm = gmsh.model.occ


@model("capacitor", dim=2, show_gui=True, finalize=True)
def generate_mesh(d: float, l: float, h: float, r: float):
    gm.add_rectangle(-l/2, -(d + h)/2, 0, l, h)
    gm.add_rectangle(-l/2, (d + h)/2, 0, l, h)
    gm.add_circle(0, 0, 0, r)
    gm.synchronize()
