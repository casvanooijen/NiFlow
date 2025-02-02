import ngsolve
from netgen.meshing import Element0D, Element1D, Element2D, MeshPoint, FaceDescriptor, Mesh
from netgen.csg import Pnt

from NiFlow.geometry.create_geometry import BOUNDARY_DICT, SEA, WALLUP, WALLDOWN, RIVER


def generate_mesh(geometry, method='unstructured', maxh_unstructured=None, num_cells = None, manual_mesh=None):
    """Generates an ngsolve Mesh from a geometry. Structured mesh generation currently only works for rectangular geometries (which **can** work in curvilinear coordinates eventually).
    
    Arguments:
    
    - geometry:             geometry to create mesh for;
    - method:               method to generate mesh; 'unstructured' (default) makes a Delaunay triangulation, 'structured_quads' makes a structured grid from rectangles,
                            'structured_tri_AQ', 'structured_tri_DQ', 'structured_tri_CCQ' make a structured grids from triangles using different methods,
                            and 'manual' lets the user input their own ngsolve mesh;
    - maxh_unstructured:    mesh size; only used if method == 'unstructured';
    - num_cells (tuple):    number of cells in each direction; only used if method != 'unstructured';
    - manual_mesh:          user-inputted mesh; only used if method == 'manual'.         
    """

    if method == 'unstructured':
        if isinstance(maxh_unstructured, float) or isinstance(maxh_unstructured, int):
            return ngsolve.Mesh(geometry.GenerateMesh(maxh=maxh_unstructured))
        else:
            raise ValueError("Could not generate unstructured mesh: mesh size not provided or invalid (maxh_unstructured)")
    elif method == 'structured_quads':
        # Meshing process based on https://docu.ngsolve.org/latest/netgen_tutorials/manual_mesh_generation.html
        # 
        if num_cells is None:
            raise ValueError("Could not generate structured mesh: number of elements in at least one direction not provided.")
        
        num_cells_x = num_cells[0]
        num_cells_y = num_cells[1]
        
        # initialisation
        mesh = Mesh(dim = 2)
        mesh.SetGeometry(geometry)

        min_x = min(geometry.PointData()[0])
        max_x = max(geometry.PointData()[0])
        min_y = min(geometry.PointData()[1])
        max_y = max(geometry.PointData()[1])

        # add mesh points
        meshpoints = []
        for i in range(num_cells_x + 1):
            for j in range(num_cells_y + 1):
                meshpoints.append(mesh.Add(MeshPoint(Pnt(min_x + i * (max_x-min_x) / num_cells_x, min_y + j * (max_y-min_y) / num_cells_y, 0))))

        # add interior elements
        interior = mesh.AddRegion('interior', 2)
        # surf = mesh.Add(FaceDescriptor(surfnr=1, domin=1, bc=0))
        for i in range(num_cells_y):
            for j in range(num_cells_x):
                mesh.Add(Element2D(interior, [meshpoints[i + j * (num_cells_y+1)], meshpoints[i + (j+1)*(num_cells_y+1)], meshpoints[i+1 + (j+1)*(num_cells_y+1)], meshpoints[i+1 + j*(num_cells_y+1)]]))

        # add boundary elements
        # seabc = mesh.Add(FaceDescriptor(surfnr=1, domin=1, bc=SEA))
        sea_bnd = mesh.AddRegion(BOUNDARY_DICT[SEA], 1)
        for j in range(num_cells_y):
            mesh.Add(Element1D([meshpoints[j], meshpoints[j+1]], index=sea_bnd))

        wallup_bnd = mesh.AddRegion(BOUNDARY_DICT[WALLUP], 1)
        for i in range(num_cells_x):
            mesh.Add(Element1D([meshpoints[num_cells_y + (num_cells_y + 1) * i], meshpoints[num_cells_y + (num_cells_y + 1) * (i+1)]], index=wallup_bnd))

        river_bnd = mesh.AddRegion(BOUNDARY_DICT[RIVER], 1)
        for j in range(num_cells_y):
            mesh.Add(Element1D([meshpoints[num_cells_x * (num_cells_y + 1) + j], meshpoints[num_cells_x * (num_cells_y + 1) + j + 1]], index=river_bnd))

        walldown_bnd = mesh.AddRegion(BOUNDARY_DICT[WALLDOWN], 1)
        for i in range(num_cells_x):
            mesh.Add(Element1D([meshpoints[(num_cells_y + 1) * i], meshpoints[((num_cells_y + 1) * (i+1))]], index=walldown_bnd))

        return mesh
    elif method == 'structured_tri_AQ': # Ascending quadrilateral bisection

        if num_cells is None:
            raise ValueError("Could not generate structured mesh: number of elements in at least one direction not provided.")
        
        num_cells_x = num_cells[0]
        num_cells_y = num_cells[1]
        # initialisation
        mesh = Mesh(dim = 2)
        mesh.SetGeometry(geometry)

        min_x = min(geometry.PointData()[0])
        max_x = max(geometry.PointData()[0])
        min_y = min(geometry.PointData()[1])
        max_y = max(geometry.PointData()[1])

        # add mesh points
        meshpoints = []
        for i in range(num_cells_x + 1):
            for j in range(num_cells_y + 1):
                meshpoints.append(mesh.Add(MeshPoint(Pnt(min_x + i * (max_x-min_x) / num_cells_x, min_y + j * (max_y-min_y) / num_cells_y, 0))))


        # add interior elements
        interior = mesh.AddRegion('interior', 2)
        for i in range(num_cells_y):
            for j in range(num_cells_x):
                mesh.Add(Element2D(interior, [meshpoints[i + j * (num_cells_y+1)], meshpoints[i + (j+1)*(num_cells_y+1)], meshpoints[i+1 + (j+1)*(num_cells_y+1)]])) # bottom right triangle in cell
                mesh.Add(Element2D(interior, [meshpoints[i + j * (num_cells_y+1)], meshpoints[i+1 + j*(num_cells_y+1)], meshpoints[i+1 + (j+1)*(num_cells_y+1)]])) # top left triangle in cell

        sea_bnd = mesh.AddRegion(BOUNDARY_DICT[SEA], 1)
        for j in range(num_cells_y):
            mesh.Add(Element1D([meshpoints[j], meshpoints[j+1]], index=sea_bnd))

        wallup_bnd = mesh.AddRegion(BOUNDARY_DICT[WALLUP], 1)
        for i in range(num_cells_x):
            mesh.Add(Element1D([meshpoints[num_cells_y + (num_cells_y + 1) * i], meshpoints[num_cells_y + (num_cells_y + 1) * (i+1)]], index=wallup_bnd))

        river_bnd = mesh.AddRegion(BOUNDARY_DICT[RIVER], 1)
        for j in range(num_cells_y):
            mesh.Add(Element1D([meshpoints[num_cells_x * (num_cells_y + 1) + j], meshpoints[num_cells_x * (num_cells_y + 1) + j + 1]], index=river_bnd))

        walldown_bnd = mesh.AddRegion(BOUNDARY_DICT[WALLDOWN], 1)
        for i in range(num_cells_x):
            mesh.Add(Element1D([meshpoints[(num_cells_y + 1) * i], meshpoints[((num_cells_y + 1) * (i+1))]], index=walldown_bnd))

        return mesh
    
    elif method == 'structured_tri_DQ': # Descending quadrilateral bisection 

        if num_cells is None:
            raise ValueError("Could not generate structured mesh: number of elements in at least one direction not provided.")
        
        num_cells_x = num_cells[0]
        num_cells_y = num_cells[1]
        
        # initialisation
        mesh = Mesh(dim = 2)
        mesh.SetGeometry(geometry)

        min_x = min(geometry.PointData()[0])
        max_x = max(geometry.PointData()[0])
        min_y = min(geometry.PointData()[1])
        max_y = max(geometry.PointData()[1])

        # add mesh points
        meshpoints = []
        for i in range(num_cells_x + 1):
            for j in range(num_cells_y + 1):
                meshpoints.append(mesh.Add(MeshPoint(Pnt(min_x + i * (max_x-min_x) / num_cells_x, min_y + j * (max_y-min_y) / num_cells_y, 0))))


        # add interior elements
        interior = mesh.AddRegion('interior', 2)
        for i in range(num_cells_y):
            for j in range(num_cells_x):
                mesh.Add(Element2D(interior, [meshpoints[i + j * (num_cells_y+1)], meshpoints[i+1 + j*(num_cells_y+1)], meshpoints[i + (j+1)*(num_cells_y+1)]])) # bottom left triangle in cell
                mesh.Add(Element2D(interior, [meshpoints[i+1 + (j+1) * (num_cells_y+1)], meshpoints[i+1 + j*(num_cells_y+1)], meshpoints[i + (j+1)*(num_cells_y+1)]])) # top right triangle in cell

        sea_bnd = mesh.AddRegion(BOUNDARY_DICT[SEA], 1)
        for j in range(num_cells_y):
            mesh.Add(Element1D([meshpoints[j], meshpoints[j+1]], index=sea_bnd))

        wallup_bnd = mesh.AddRegion(BOUNDARY_DICT[WALLUP], 1)
        for i in range(num_cells_x):
            mesh.Add(Element1D([meshpoints[num_cells_y + (num_cells_y + 1) * i], meshpoints[num_cells_y + (num_cells_y + 1) * (i+1)]], index=wallup_bnd))

        river_bnd = mesh.AddRegion(BOUNDARY_DICT[RIVER], 1)
        for j in range(num_cells_y):
            mesh.Add(Element1D([meshpoints[num_cells_x * (num_cells_y + 1) + j], meshpoints[num_cells_x * (num_cells_y + 1) + j + 1]], index=river_bnd))

        walldown_bnd = mesh.AddRegion(BOUNDARY_DICT[WALLDOWN], 1)
        for i in range(num_cells_x):
            mesh.Add(Element1D([meshpoints[(num_cells_y + 1) * i], meshpoints[((num_cells_y + 1) * (i+1))]], index=walldown_bnd))

        return mesh
    
    elif method == 'structured_tri_CCQ': # Criss-cross quadrilateral bisection
        if num_cells is None:
            raise ValueError("Could not generate structured mesh: number of elements in at least one direction not provided.")
        
        num_cells_x = num_cells[0]
        num_cells_y = num_cells[1]
        
        # initialisation
        mesh = Mesh(dim = 2)
        mesh.SetGeometry(geometry)

        min_x = min(geometry.PointData()[0])
        max_x = max(geometry.PointData()[0])
        min_y = min(geometry.PointData()[1])
        max_y = max(geometry.PointData()[1])

        # add mesh points
        meshpoints = []
        for i in range(num_cells_x + 1):
            for j in range(num_cells_y + 1):
                meshpoints.append(mesh.Add(MeshPoint(Pnt(min_x + i * (max_x-min_x) / num_cells_x, min_y + j * (max_y-min_y) / num_cells_y, 0))))

        cellcenters = []
        for i in range(num_cells_y):
            for j in range(num_cells_x):
                cellcenters.append(mesh.Add(MeshPoint(Pnt(i / num_cells_x + 1 / (2*num_cells_x), j / num_cells_y + 1 / (2*num_cells_y) - 0.5,0))))

        # print([f'({p.x}, {p.y})' for p in cellcenters])
        # Add elements
        interior = mesh.AddRegion('interior', 2)
        for i in range(num_cells_x):
            for j in range(num_cells_y):
                mesh.Add(Element2D(interior, [meshpoints[i + j*(num_cells_y+1)], meshpoints[i+1 + j*(num_cells_y+1)], cellcenters[i + j*num_cells_y]]))
                mesh.Add(Element2D(interior, [meshpoints[i + j*(num_cells_y+1)], meshpoints[i + (j+1)*(num_cells_y+1)], cellcenters[i + j*num_cells_y]]))
                mesh.Add(Element2D(interior, [meshpoints[i+1 + j*(num_cells_y+1)], meshpoints[i+1 + (j+1)*(num_cells_y+1)], cellcenters[i + j*num_cells_y]]))
                mesh.Add(Element2D(interior, [meshpoints[i+1 + (j+1)*(num_cells_y+1)], meshpoints[i + (j+1)*(num_cells_y+1)], cellcenters[i + j*num_cells_y]]))

        sea_bnd = mesh.AddRegion(BOUNDARY_DICT[SEA], 1)
        for j in range(num_cells_y):
            mesh.Add(Element1D([meshpoints[j], meshpoints[j+1]], index=sea_bnd))

        wallup_bnd = mesh.AddRegion(BOUNDARY_DICT[WALLUP], 1)
        for i in range(num_cells_x):
            mesh.Add(Element1D([meshpoints[num_cells_y + (num_cells_y + 1) * i], meshpoints[num_cells_y + (num_cells_y + 1) * (i+1)]], index=wallup_bnd))

        river_bnd = mesh.AddRegion(BOUNDARY_DICT[RIVER], 1)
        for j in range(num_cells_y):
            mesh.Add(Element1D([meshpoints[num_cells_x * (num_cells_y + 1) + j], meshpoints[num_cells_x * (num_cells_y + 1) + j + 1]], index=river_bnd))

        walldown_bnd = mesh.AddRegion(BOUNDARY_DICT[WALLDOWN], 1)
        for i in range(num_cells_x):
            mesh.Add(Element1D([meshpoints[(num_cells_y + 1) * i], meshpoints[((num_cells_y + 1) * (i+1))]], index=walldown_bnd))

        return mesh
    elif method == 'manual':
        return manual_mesh       
        
    else:
        raise ValueError(f"{method} is an invalid mesh generation method; please choose from 'unstructured', 'structured_quads', or 'structured_tri'.")

