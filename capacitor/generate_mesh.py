import gmsh

from util.mesh import model

cad = gmsh.model.occ
msh = gmsh.model.mesh


@model("capacitor", dim=2, show_gui=True, finalize=True, options={"Mesh.MeshSizeFactor": 0.4})
def generate_mesh(d: float, l: float, h: float, r: float):
    plate1 = add_plate(d, l, h)
    plate2 = add_plate(-d, l, -h)
    air = cad.add_circle(0, 0, 0, r)
    air = cad.add_curve_loop([air])
    cad.add_plane_surface([air, plate1, plate2])
    cad.synchronize()


def add_plate(d: float, l: float, h: float) -> int:
    """Adds a plate of the capacitor to the geometry.
    :return: The tag of the curve loop.
    """
    p1 = cad.add_point(l / 2, (d + h) / 2, 0)
    p2 = cad.add_point(-l / 2, (d + h) / 2, 0)
    p3 = cad.add_point(-l / 2, (d - h) / 2, 0)
    p4 = cad.add_point(l / 2, (d - h) / 2, 0)
    l1 = cad.add_line(p1, p2)
    l2 = cad.add_line(p2, p3)
    l3 = cad.add_line(p3, p4)
    l4 = cad.add_line(p4, p1)

    cad.synchronize()
    msh.set_transfinite_curve(l1, 30)
    msh.set_transfinite_curve(l3, 30)
    return cad.add_curve_loop([l1, l2, l3, l4])
