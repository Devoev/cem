import gmsh

from util.gmsh_model import gmsh_model

cad = gmsh.model.occ
msh = gmsh.model.mesh


@gmsh_model("fichera", dim=2, finalize=False, options={"Mesh.MeshSizeFactor": 0.3, "Mesh.MeshSizeMax": 0.3})
def gen_fichera():
    """Creates the fichera geometry."""

    p1 = cad.add_point(0, 1, 0)
    p2 = cad.add_point(1, 1, 0)
    p3 = cad.add_point(1, 0, 0)
    p4 = cad.add_point(0.5, 0, 0)
    p5 = cad.add_point(0.5, 0.5, 0)
    p6 = cad.add_point(0, 0.5, 0)
    l1 = cad.add_line(p1, p2)
    l2 = cad.add_line(p2, p3)
    l3 = cad.add_line(p3, p4)
    l4 = cad.add_line(p4, p5)
    l5 = cad.add_line(p5, p6)
    l6 = cad.add_line(p6, p1)
    boundary = cad.add_curve_loop([l1, l2, l3, l4, l5, l6])
    cad.add_plane_surface([boundary])
    cad.synchronize()
