import gmsh

from util.mesh import model

gm = gmsh.model.occ


@model("capacitor", dim=2, show_gui=True, finalize=True)
def generate_mesh(d: float, l: float, h: float, r: float):
    plate1 = add_plate(d, l, h)
    plate2 = add_plate(-d, l, -h)
    air = gm.add_circle(0, 0, 0, r)
    air = gm.add_curve_loop([air])
    gm.add_plane_surface([air, plate1, plate2])
    gm.synchronize()


def add_plate(d: float, l: float, h: float) -> int:
    """Adds a plate of the capacitor to the geometry.
    :return: The tag of the curve loop.
    """
    p1 = gm.add_point(l/2, (d + h)/2, 0)
    p2 = gm.add_point(-l/2, (d + h)/2, 0)
    p3 = gm.add_point(-l/2, (d - h)/2, 0)
    p4 = gm.add_point(l/2, (d - h)/2, 0)
    l1 = gm.add_line(p1, p2)
    l2 = gm.add_line(p2, p3)
    l3 = gm.add_line(p3, p4)
    l4 = gm.add_line(p4, p1)
    return gm.add_curve_loop([l1, l2, l3, l4])
