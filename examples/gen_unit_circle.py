import gmsh

from util.gmsh_model import gmsh_model

cad = gmsh.model.occ


@gmsh_model("unit_circle", dim=2, finalize=False, options={"Mesh.MeshSizeFactor": 0.2})
def gen_unit_circle():
    c = cad.add_circle(0,0,0,1)
    boundary = cad.add_curve_loop([c])
    cad.add_plane_surface([boundary])
    cad.synchronize()
