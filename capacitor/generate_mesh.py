import gmsh

from capacitor.region import Region
from util.gmsh_model import gmsh_model

cad = gmsh.model.occ
msh = gmsh.model.mesh


@gmsh_model("capacitor", dim=2, finalize=False, options={"Mesh.MeshSizeFactor": 1})
def generate_mesh(d: float, l: float, h: float, r: float):
    """
    Generates the capacitor mesh.
    :param d: Distance of the capacitor plates.
    :param l: Length of the capacitor plate.
    :param h: Height of each capacitor plate.
    :param r: Radius of domain.
    """

    plate1 = add_thin_plate(d, l) if h == 0 else add_plate(d, l, h)
    plate2 = add_thin_plate(-d, l) if h == 0 else add_plate(-d, l, -h)

    cad.synchronize()
    gmsh.model.add_physical_group(1, [plate1], Region.PLATE_1, "Plate 1")
    gmsh.model.add_physical_group(1, [plate2], Region.PLATE_2, "Plate 2")

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


def add_thin_plate(d: float, l: float) -> int:
    """Adds an infinitely thin plate of the capacitor to the geometry.
    :return: The tag of the curve loop.
    """

    p1 = cad.add_point(-l / 2, d / 2, 0)
    p2 = cad.add_point(l / 2, d / 2, 0)
    l1 = cad.add_line(p1, p2)

    cad.synchronize()
    msh.set_transfinite_curve(l1, 30)
    return cad.add_curve_loop([l1, -l1])
