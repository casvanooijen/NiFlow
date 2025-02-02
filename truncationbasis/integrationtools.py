import numpy as np
# import JacobiPolynomials
import scipy.special
import timeit


# To do: add regular Gauss quadrature; it is more accurate for the same computational cost; the benefit of GLL is that end points are included
# but in contrast to FEM, we don't really need them.

def get_GLL_points(num_points):
    """Returns a one-dimensional array of values corresponding to the Gauss-Lobatto-Legendre quadrature points in the interval
    [xl, xr]."""

    # Generate quadrature points on [-1,1]
    points = np.zeros(num_points)
    # points[1:-1] = JacobiPolynomials.Jacobi_zeros(1, 1, num_points - 2, 1e-6)
    points[1:-1], _ = scipy.special.roots_jacobi(num_points - 2, 1, 1)
    points[0] = -1
    points[-1] = 1
    return points


def map_GLL_points(points, xl, xr):
    """Map the GLL points from the interval [-1,1] to [xl, xr]."""
    return 0.5*xl*(np.ones_like(points)-points)+0.5*xr*(np.ones_like(points) + points)


def GLL_integrate(values, points, xl=-1, xr=1):
    """Numerically integrate a function over [xl, xr] with function values 'values' at GLL quadrature points mapped to [xl, xr], using
    GLL-quadrature.
    
    Arguments:
    
    - values:       values of the integrand at mapped GLL quadrature points;
    - points:       basic (non-mapped) GLL quadrature points;
    - xl:           left integration boundary;
    - xr:           right integration boundary;"""

    num_points = values.shape[0]
    #Legendre_pol = JacobiPolynomials.Jacobi_polynomial(0, 0, num_points - 1)

    weights = [2 / (num_points*(num_points-1)*scipy.special.eval_legendre(num_points - 1, points[i])*scipy.special.eval_legendre(num_points - 1, points[i])) \
               for i in range(num_points)]

    return sum([values[i]*weights[i]*(xr-xl)/2 for i in range(num_points)])


def get_mesh_GLL_points(num_elements, points, xl, xr):
    """For a one-dimensional mesh with [num_elements] elements, return an array of values corresponding to the GLL quadrature points
    for every element in the mesh. Will contain duplicate values at boundaries of elements, but the rest of the functions take this
    into account.
    
    Arguments:
    
    - num_elements:         number of elements in the mesh;
    - points:               array of basic GLL quadrature points;
    - xl:                   left boundary of mesh;
    - xr:                   right boundary of mesh;
    """
    vertices = np.linspace(xl, xr, num_elements+1) 
    num_GLL_points = points.shape[0]

    mesh_GLL_points = np.zeros((num_elements,num_GLL_points))

    for e in range(num_elements): # there is one less element than there are vertices
        elemental_xl = vertices[e]
        elemental_xr = vertices[e + 1]
        mesh_GLL_points[e, :] = map_GLL_points(points, elemental_xl, elemental_xr)

    return mesh_GLL_points


def GLL_integrate_mesh(values, points, xl=-1, xr=1):
    """Arguments:
    
    values:         values of integrand in matrix form; rows denote element index and columns denote quadrature point index; to
                    generate such values from a standard python function that takes as input the x-variable, input the result of
                    get_mesh_GLL_points(..) into it;
    points:         basic (non-mapped) GLL quadrature points;
    xl:             left integration boundary;
    xr:             right integration boundary;
    """
    total_integral = 0
    num_elements = values.shape[0]
    vertices = np.linspace(xl, xr, num_elements + 1)
    for e in range(num_elements):
        elemental_xl = vertices[e]
        elemental_xr = vertices[e + 1]
        total_integral += GLL_integrate(values[e,:], points, xl=elemental_xl, xr=elemental_xr)

    return total_integral



if __name__ == '__main__':

    points = get_GLL_points(3)

    mapped_points = map_GLL_points(points, 0, 1)
    values = np.sin(mapped_points)

    mesh_points = get_mesh_GLL_points(10, points, 0, 1)
    mesh_values = np.sin(mesh_points)

    # integrate identity function over [-1,0]
    print('One element: ', GLL_integrate(values, points, xl=0, xr=1))
    print('Two elements: ', GLL_integrate_mesh(mesh_values, points, xl=0, xr=1))
    



