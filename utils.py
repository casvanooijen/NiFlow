import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import ngsolve
import scipy.sparse as sp


def mesh_to_coordinate_array(mesh):
    """Generates a coordinate array from a netgen mesh.
    
    Args:

    - mesh:      netgen mesh object
    """

    coords = [[]]
    for p in mesh.Points():
        x, y, z = p.p
        coords[-1] += [x, y, z]
        coords.append([])

    coords = coords[:-1] # Delete last empty list        
    return np.array(coords)


def mesh2d_to_triangles(mesh):
    """Gives an array containing the indices of the triangle vertices in mesh.
    
    Args:
    
    - mesh:     netgen mesh object.
    """

    triangles = [[]]
    for el in mesh.Elements2D():
        # Netgen does not store integers in el.vertices, but netgen.libngpy._meshing.PointId objects; first we convert
        vertices = [v.nr - 1 for v in el.vertices] # PointId objects start counting at 1
        triangles[-1] += vertices
        triangles.append([])
    
    triangles = triangles[:-1] # Delete last empty list
    return np.array(triangles)


def get_triangulation(mesh):
    """Converts a netgen mesh to a matplotlib.tri-Triangularion object.
    
    Arguments:
    
        - mesh:     netgen mesh object.
    """
    coords = mesh_to_coordinate_array(mesh)
    triangles = mesh2d_to_triangles(mesh)
    triangulation = tri.Triangulation(coords[:,0], coords[:, 1], triangles)
    return triangulation


def refine_mesh_by_elemental_integration(mesh: ngsolve.Mesh, cf: ngsolve.CoefficientFunction, K: float, p=1):
    """
    Refines an ngsolve-Mesh by integrating a user-provided ngsolve.CoefficientFunction over each element.
    If the p-norm of the coefficient function in a particular element exceeds the average p-norm among all elements by 
    a factor of K, that element is marked for refinement. Returns how many elements were refined.

    Arguments:

    - mesh:     the mesh that will be refined;
    - cf:       the coefficient function that is used for the mesh refinement rule;
    - K:        threshold by which the p-norm in a particular element must exceed the average;
    - p:        indicates which L^p-norm is used for the rule.    
    
    """
    if K <= 1:
        print("Please enter K>1")
        return

    integralvals = ngsolve.Integrate(cf, mesh, ngsolve.VOL, element_wise=True)

    counter = 0

    # compute average integral val
    avg = (1/mesh.ne) * sum([integralvals[el.nr]**p for el in mesh.Elements()])


    # print(avg)

    for el in mesh.Elements():
        if integralvals[el.nr]**(p) > K * avg:
            counter += 1
        # print(integralvals[el.nr]**(1/p) / avg)
        mesh.SetRefinementFlag(el, integralvals[el.nr]**(p)  > K * avg)
    
    mesh.Refine()
    return counter


def fix_CF_before_xvalue(cf: ngsolve.CoefficientFunction, mesh: ngsolve.Mesh, x0: float, num_evaluation_points: int = 300):
    """Returns Coefficient Function that is defined by the piecewise definition new_cf(x)=cf(x0) if x <= x0 and new_cf(x)=cf(x) otherwise.
    
    Only works for rectangular meshes where y ranges from -0.5 to 0.5.
    
    Arguments:
    
        - cf: coefficient function to fix before x0.
        - mesh: mesh on which the coefficient function is defined.
        - x0: x-value before which the value of the coefficient function should be constant.
        - num_evaluation_points: number of points at which the y-structure of the cf is evaluated; by default 300. 
    
    """

    y = np.linspace(-0.5, 0.5, num_evaluation_points)
    eval_cf = evaluate_CF_range(cf, mesh, x0 * np.ones_like(y), y)

    # make a CF that is constant in x, whose y-profile is the same as eval_cf
    constant_cf_spline = ngsolve.BSpline(2, [-0.5] + list(y) + [0.5], list(eval_cf))
    constant_cf = ngsolve.CF(constant_cf_spline(ngsolve.y))

    # make a CF that is constant before the chosen value of x and returns to varying like cf after
    return ngsolve.IfPos(x0 - ngsolve.x, constant_cf, cf)


def evaluate_CF_point(cf, mesh, x, y):
    """
    Evaluates an ngsolve CoefficientFunction, of which the ngsolve GridFunction is a child class, at a point (x,y).
    Returns function value.

    Arguments:

        - cf:       Coefficient function to be evaluated;
        - mesh:     ngsolve mesh that contains (x,y);
        - x:        x-coordinate of evaluation point;
        - y:        y-coordinate of evaluation point;
    
    """
    return cf(mesh(x, y))


def evaluate_CF_range(cf, mesh, x, y):
    """
    Evaluates an ngsolve CoefficientFunction, of which the ngsolve GridFunction is a child class, at a range of points
    contained in two arrays x and y. Returns array of function values.

    Arguments:
        
        - cf:       Coefficient function to be evaluated;
        - mesh:     mesh that contains all points in (x,y);
        - x:        array of x-values;
        - y:        array of y-values;
    
    """
    return cf(mesh(x, y)).flatten()


def plot_CF_colormap(cf, mesh, refinement_level=1, show_mesh=False, title='Gridfunction', save=None, **kwargs):
    """
    Plots a simple colormap of a Coefficient function on a refined display mesh.

    Arguments:

        - cf:                   Coefficient function to be plotted;
        - mesh:                 computational mesh that is refined for the display mesh; 
        - refinement_level:     how many times the mesh is refined to get a display mesh on which the function is evaluated;
        - show_mesh:            if True, overlays the colormap with the mesh;
        - title:                title of the plot;
        - save:                 saves the figure to f'{save}.png',
        - **kwargs:             keyword arguments for matplotlib.pyplot.tripcolor
    
    """
    triangulation = get_triangulation(mesh.ngmesh)
    refiner = tri.UniformTriRefiner(triangulation)
    refined_triangulation = refiner.refine_triangulation(subdiv=refinement_level)
    
    eval_cf = evaluate_CF_range(cf, mesh, refined_triangulation.x, refined_triangulation.y)
    fig, ax = plt.subplots()
    if show_mesh:
        ax.triplot(triangulation, linewidth=0.5, color='k', zorder=2)
    colormesh = ax.tripcolor(refined_triangulation, eval_cf, **kwargs)

    ax.set_title(title)
    cbar = fig.colorbar(colormesh)
    if save is not None:
        fig.savefig(f'{save}.png')



def plot_mesh2d(mesh, title=None, color='k', linewidth=0.5):
    """
    Plots a wireframe of an ngsolve Mesh.

    Arguments:

        - mesh:         mesh to be plotted;
        - title:        title of the plot;
        - color:        color of the wireframe;
        - linewidth:    linewidth of the wireframe;
    
    """
    coords = mesh_to_coordinate_array(mesh.ngmesh)
    triangles = mesh2d_to_triangles(mesh.ngmesh)
    triangulation = tri.Triangulation(coords[:, 0], coords[:, 1], triangles)

    _, ax = plt.subplots()
    ax.triplot(triangulation, color=color, linewidth=linewidth)
    if title:
        ax.set_title(title)
    plt.show()


def get_boundaryelement_vertices(mesh: ngsolve.Mesh, bnd):
    """Returns a list of (x,y)-coordinates of all the gridpoints in a certain part of the mesh boundary.
    
    Arguments:
    
        - mesh:     the considered mesh;
        - bnd:      the name of the considered boundary;
        
        
    """

    IDlist = []
    for el in mesh.Boundaries(bnd).Elements():
        IDlist.append(el.vertices[0])
        IDlist.append(el.vertices[1])
    IDlist = list(set(IDlist)) # Remove duplicates

    plist = []
    for p in mesh.vertices:
        if p in IDlist:
            plist.append(p.point)

    return plist


def save_gridfunction(gf: ngsolve.GridFunction, filename, format='npy'):
    """
    Converts an ngsolve GridFunction to a numpy array and subsequently saves it.

    Arguments:

        - gf:           ngsolve GridFunction to be saved;
        - filename:     name of the file in which the GridFunction is saved;
        - format:       file format; choices are 'npy' for a .npy-file and 'txt' for a .txt-file. 
    
    
    """
    gf_vec_array = gf.vec.FV().NumPy()
    if format == 'npy':
        np.save(filename, gf_vec_array)
    elif format == 'txt':
        np.savetxt(filename, gf_vec_array)
    else:
        raise ValueError(f"Invalid format {format}: please choose npy or txt")


def load_basevector(vec: ngsolve.BaseVector, filename, format = 'npy'):
    """
    Sets the value of an ngsolve BaseVector using a file.

    Arguments:
        - vec:              ngsolve BaseVector to be set;
        - filename:         name of the file;
        - format:           format of the file; choices are 'npy' for .npy-files and 'txt' for .txt-files;
    
    
    """
    if format == 'npy':
        array = np.load(filename + '.npy')
    elif format == 'txt':
        array = np.genfromtxt(filename)
    else:
        raise ValueError(f"Invalid format {format}: please choose npy or txt")
    
    vec.FV().NumPy()[:] = array


def get_dirichletdof_indices(freedofs: ngsolve.BitArray):
    """Returns a list of indices corresponding to constrained (Dirichlet) degrees of freedom, based on a 
    bitarray where 0 denotes a constrained DOF.
    
    Arguments:
    
        - freedofs:     bitarray determining the free degrees of freedom, obtainable by using the ngsolve method femspace.FreeDofs()"""
    
    indices = []
    counter = 0
    for isFree in freedofs:
        if not isFree:
            indices.append(counter)
        counter += 1
    return indices
    

def get_freedof_list(freedof_bitarray):
    """Constructs a list containing the indices of the free degrees of freedom, generated from a bitarray from ngsolve.
    
    Arguments:
    
        - freedof_bitarray:     bitarray indicating which degree of freedom is free.
    """
    freedof_list = []
    for i, isFree in enumerate(freedof_bitarray):
        if isFree:
            freedof_list.append(i)
    return freedof_list


def basematrix_to_csr_matrix(mat: ngsolve.BaseMatrix):
    """Converts an ngsolve BaseMatrix to a scipy sparse matrix in CSR-format. Returns the CSR-matrix
    
    Arguments:

    - mat:      to-be-converted BaseMatrix      
        
    """
    rows, cols, vals = mat.COO()
    return sp.csr_matrix((vals, (rows, cols)))


def slice_csr_matrix(mat, slicelist0, slicelist1):
    """Returns a sliced sparse CSR matrix, which is row-sliced using slicelist0 and column-sliced using slicelist1.
    
    Arguments:
    
    - mat:          matrix to be sliced;
    - slicelist0:   rows to keep in the slice;
    - slicelist1:   columns to keep in the slice;
    """

    mat = mat[slicelist0, :]
    mat = mat[:, slicelist1]
    return mat


def remove_fixeddofs_from_csr(mat, freedof_list):
    """Removes all of the fixed degrees of freedom from a scipy sparse matrix in CSR-format.
    
    Arguments:

        - mat:              matrix to be sliced;
        - freedof_list:     list of degrees of freedom to be kept; obtainable via `get_freedof_list`.

    """

    return slice_csr_matrix(mat, freedof_list, freedof_list)


def get_component_length(gf: ngsolve.GridFunction):
    """Returns the length of a component of a gridfunction. In our model, this means how many FE basis functions are associated to each alpha_{m,i}, beta_{m,i} and gamma_{m,i}.
    
    Arguments:
    
    gf (ngsolve.Gridfunction)
    
    """
    vec = gf.components[0].vec
    return np.shape(vec.FV().NumPy())[0]


def get_num_free_basisfunctions_per_component(gf: ngsolve.GridFunction, fespace, component):
    """Returns the number of free basis functions for a component of a gridfunction. In our model, this means the size of each block in the system matrix.
    
    Arguments:
    
    - gf (ngsolve.GridFunction):      gridfunction
    - fespace:                        finite element the gridfunction is defined on
    - component (int):                which component should be evaluated
    
    """
    total_num_basisfunctions = get_component_length(gf)
    
    num_free_basisfunctions = 0
    for i in range(total_num_basisfunctions):
        if fespace.FreeDofs()[total_num_basisfunctions * component + i]:
            num_free_basisfunctions += 1
    
    return num_free_basisfunctions


def get_num_linear_basisfunctions(mesh, dirichlet=None):
    """Returns the number of first order basis functions for a certain mesh, order, and set of boundary conditions.
    
    
    Arguments:
    
    - mesh:                 NGSolve mesh
    - order:                order of the FESpace
    - dirichlet:            string indicating which of the boundaries have essential boundary conditions. For instance, for the water level, the argument would be dirichlet=BOUNDARY_DICT[SEA]
    
    """

    fes = ngsolve.H1(mesh, order=1, dirichlet=dirichlet)
    return len(list(fes.FreeDofs()))


def ngsolve_tanh(argument):
    """Returns ngsolve.CoefficientFunction-version of the hyperbolic tangent function, evaluated in argument."""
    return ngsolve.sinh(argument) / ngsolve.cosh(argument)


def count_free_dofs(fes):
    """
    Returns the number of free degrees of freedom in an ngsolve Finite Element space.

    Arguments:

        - fes:      ngsolve Finite element space.   
    
    """
    i = 0
    for isFree in fes.FreeDofs():
        i = i + isFree
    return i


def homogenise_essential_Dofs(vec: ngsolve.BaseVector, freedofs):
    """
    Sets the essential (non-free) degrees of freedom of an ngsolve BaseVector to zero.

    Arguments:
        
        - vec:          ngsolve BaseVector;
        - freedofs:     bitarray indicating the free degrees of freedom, obtainable via calling the method FiniteElementSpace.FreeDofs();
    
    
    """
    for i, free in enumerate(freedofs):
        if not free:
            vec[i] = 0.


def amp(gfu: ngsolve.GridFunction):
    """Returns amplitude of complex ngsolve.GridFunction"""
    return ngsolve.sqrt(gfu*ngsolve.Conj(gfu)).real


def minusonepower(n: int):
    """Fast way to evaluate (-1)^n."""
    if n % 2 == 0:
        return 1
    else:
        return -1















