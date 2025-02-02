"""In this file, a class is made that fits and projects point cloud data to ngsolve objects"""


# TODO
"""In this file a NGSolve gridfunction is generated from a point cloud

The point cloud input data is provided as
pointclouddata = np.array([[x1,y1,s1],[x2,y2,s2],[x3,y3,s3]...])
"""

import numpy as np
from scipy.interpolate import griddata, Rbf

import ngsolve

def fit_and_project(pointclouddata, fitting_method, projection_method, mesh):
    """Function used to both fit data and then project it to ngsolve
    Parameters:
            pointclouddata      - numpy array as [[x1, y1, v1], [x2, y2, v2], ...], where xi,yi are the spatial locations and vi the value at that location.
            fitting_method      - method to fit the point cloud data, options: "griddata"
            projection_method   - method used to project the fit into a NGSolve object (i.e. a grid function), options: "linear"
            mesh                - computational mesh
    Returns:
        - NGSolve gridfunction
    """
    # TODO test
    # Check if the fit is projected linearly on a NGSolve grid function:
    if projection_method == "linear":
        # For the linear fit, we project exactly on the mesh vertices.
        # 1) Get mesh vertices
        # 2) Use fitting algorithm to get pointclouddata on mesh vertices
        # 3) Create gridfunction with the coefficients set to these values

        # 1) Mesh vertices
        mesh_vertices = np.array([vertex.point for vertex in mesh.vertices])


        # 2) Values at mesh vertices
        if fitting_method == "griddata":
            # TODO make fill_value not hardcoded
            values_mesh_vertices = griddata(pointclouddata[:, :2], pointclouddata[:, 2],  mesh_vertices, fill_value=0.2)

        # 3) Generate linear grid function:
        fes_linear = ngsolve.H1(mesh, order=1)
        gf = ngsolve.GridFunction(fes_linear)

        # Set the coefficients of gf
        gf.vec.FV().NumPy()[:] = values_mesh_vertices

    return gf




def project(coefficient_function, projection_method, mesh):
    """Function used to both fit data and then project it to ngsolve
    Parameters:
            coefficient_function  - Function consisting of the composition of GridFunctions and CoefficientFunctions
            fitting_method      - method to fit the point cloud data, options: "griddata"
            projection_method   - method used to project the fit into a NGSolve object (i.e. a grid function), options: "linear"
            mesh                - computational mesh
    Returns:
        - NGSolve gridfunction
    """
    # TODO test
    # Check if the fit is projected linearly on a NGSolve grid function:
    if projection_method == "linear":
        # For the linear fit, we project exactly on the mesh vertices.
        # 1) Get mesh vertices
        # 2) Use fitting algorithm to get pointclouddata on mesh vertices
        # 3) Create gridfunction with the coefficients set to these values

        # 1) Mesh vertices
        mesh_vertices = np.array([vertex.point for vertex in mesh.vertices])


        # 2) Values at mesh vertices
        value_at_mesh_vertices = coefficient_function(mesh(mesh_vertices[:, 0], mesh_vertices[:, 1]))

        # 3) Generate linear grid function:
        fes_linear = ngsolve.H1(mesh, order=1, complex=True)
        gf = ngsolve.GridFunction(fes_linear)

        # Set the coefficients of gf
        gf.vec.FV().NumPy()[:] = value_at_mesh_vertices.flatten()

    return gf




# TODO remove
def generate_pointclouddata(eta, mesh):
    """
    Function used to generate sample point cloud data.
    Parameters:
    - eta : ngsolve grid function
    - mesh: computational mesh
    Returns:
    - pointclouddata: Pointclouddata in the form np.array([[x1,y1,s1],[x2,y2,s2],...])
    """
    # Evaluate the gridfunction xi in the mesh vertices
    pointclouddata = []
    for v in mesh.vertices:
        x,y = v.point
        s = eta(mesh(x,y))
        pointclouddata.append([x, y, s])
    return np.array(pointclouddata)

def rbf_fit(pointclouddata):
    """
       Function generates a radial basis function fit of the given point cloud data
       Parameters:
           - pointclouddata: Pointclouddata in the form np.array([[x1,y1,s1],[x2,y2,s2],...])
       Returns:
           - s_rbf_fit: fit object from numpy
       """
    s_rbf_fit = Rbf(pointclouddata[:, 0], pointclouddata[:, 1], pointclouddata[:, 2], smooth=0, function='gaussian')
    return s_rbf_fit


def creategridfunction(s_fit, mesh):
    """
    Function used to create a linear NGSolve gridfunction from a given fitting object
    Parameters:
        - s_fit : fit object from numpy
        - mesh: computational mesh
    Returns:
        - s_gf: NGSolve grid function that linearly approximates the fit
    """

    # Generate a linear approximation space on the mesh
    fes_approximation = ngsolve.H1(mesh, order=1)
    s_gf = ngsolve.GridFunction(fes_approximation)

    # Evaluate s_fg on the mesh vertices
    s_meshvertices = []
    for v in mesh.vertices:
        s_meshvertices.append(s_fit(*v.point))

    # Set the coefficients of s_gf equal to s_meshvertices at the mesh vertices
    s_gf.vec.FV().NumPy()[:] = s_meshvertices
    return s_gf
