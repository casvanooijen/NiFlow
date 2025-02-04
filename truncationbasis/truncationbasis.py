###############################################################
## THIS FILE CONTAINS CLASSES TruncationBasis AND Projection ##
###############################################################


import numpy as np
import matplotlib.pyplot as plt
import ngsolve
import scipy.special
import scipy.sparse
from scipy.sparse.linalg import spsolve

import NiFlow.truncationbasis.integrationtools as integrationtools



class TruncationBasis(object):
    """TruncationBasis object with evaluation_function attribute. This is an ngsolve-CoefficientFunction that takes as arguments 
    a coordinate z, and a non-negative integer n to denote the index of the basis function. Assumed to have domain [-1,0]. By means
    of the weight function, also has information about the underlying version of L^2([-1,0]).

    Attributes:

    - evaluation_function:              function of z and n that evaluates basis function n at z;
    - derivative_evaluation_function:   derivative w.r.t. z;
    - evaluation_cf:                    ngsolve CoefficientFunction version of evaluation_function with n as parameter;
    - inner_product:                    function of i and j that returns the value of the (weighted) inner product of basis functions i and j;
    - weight_function:                  function of z that specifies the weight function of this inner product and the underlying version of L^2. 
    
    Methods:
    
    - plot:                     plots a number of basis functions.
    - add_analytical_tensors    adds a number of functions that return analytical expressions for various inner product between basis functions and their derivatives/integrals
    
    """

    def __init__(self, evaluation_function, inner_product=None, weight_function=None, derivative_evaluation_function=None, second_derivative_evaluation_function=None):
        """Arguments:
        
        - evaluation_function:      function with input z and (integer) n;
        - inner_product:            function that takes input i and j and outputs the L^2([-1,0])-inner product of 
                                    basis functions i and j; could be a weighted inner product
        - weight_function:          weight function of the inner product above; if no weight is provided, a constant 1 one is used.
        
        """
        self.evaluation_function = evaluation_function
        self.derivative_evaluation_function = derivative_evaluation_function
        if inner_product:
            self.inner_product = inner_product

        # You can provide a weight function without providing an inner product
        if weight_function:
            self.weight_function = weight_function
        else:
        # Use identity weight function if no weight function is provided
            def wf(x):
                return 1
            
            self.weight_function = wf

        if derivative_evaluation_function is not None:
            self.derivative_evaluation_function = derivative_evaluation_function
        
        if second_derivative_evaluation_function is not None:
            self.second_derivative_evaluation_function = second_derivative_evaluation_function


    def plot(self, num_functions: int, num_points: int = 1000):
        """Plots a number of the basis functions.
        
        Arguments:
        
        - num_functions:        number of basis functions (starting at 0) that should be displayed;
        - num_points:           number of points the basis functions should be evaluated at;
        """
        _, ax = plt.subplots()
        z = np.linspace(-1, 0, num_points)
        for n in range(num_functions):
            ax.plot(z, self.evaluation_function(z, n))
        plt.show()


    def add_analytical_tensors(self, tensor_dict):
        self.tensor_dict = tensor_dict


##################################
## PREDEFINED EXAMPLES OF BASES ##
##################################
        
    # 1: vertical eigenfunction basis corresponding to constant eddy viscosity and no-slip bed boundary condition

def f(z, n):
    return  np.cos((n+0.5)*np.pi*z)


def fprime(z, n):
    return -(n+0.5)*np.pi*np.sin((n+0.5)*np.pi*z)


def fdoubleprime(z, n):
    return -(n+0.5)**2 * np.pi**2 * f(z,n)


def inner_prod(m, k):
    if m == k:
        return 0.5
    else:
        return 0
    
def fint(z, n):
    xn = (n+0.5) * np.pi
    return (1/xn) * (np.sin(xn * z) + np.sin(xn))
    
eigbasis_constantAv = TruncationBasis(f, inner_prod, derivative_evaluation_function=fprime, second_derivative_evaluation_function=fdoubleprime)
eigbasis_constantAv.integrated_evaluation_function = fint

def minusonepower(num):
    if num % 2 == 0:
        return 1
    else:
        return -1

def analytical_G1(m, n, k): # int f_m f_n f_k
    if isinstance(m, int):
        m = float(m)
    elif isinstance(m, np.ndarray):
        m = m.astype(float)
    if isinstance(n, int):
        n = float(n)
    elif isinstance(n, np.ndarray):
        n = n.astype(float)
    if isinstance(k, int):
        k = float(k)
    elif isinstance(k, np.ndarray):
        k = k.astype(float)
    return 0.25*(np.power(-1, m+n+k+1)/((m+n+k+1.5)*np.pi) + np.power(-1, m+n-k)/((m+n-k+0.5)*np.pi)
                    + np.power(-1, m-n+k)/((m-n+k+0.5)*np.pi) + np.power(-1, m-n-k+1)/((m-n-k-0.5)*np.pi))

def G1_iszero(m, n, k):
    return False

def analytical_G2(m, n, k): # int f_m' int[f_n] f_k
    if isinstance(m, int):
        m = float(m)
    elif isinstance(m, np.ndarray):
        m = m.astype(float)
    if isinstance(n, int):
        n = float(n)
    elif isinstance(n, np.ndarray):
        n = n.astype(float)
    if isinstance(k, int):
        k = float(k)
    elif isinstance(k, np.ndarray):
        k = k.astype(float)
    if m != k:
        term1 = np.power(-1,m-n+k+1)/((m-n+k+0.5)*np.pi) + np.power(-1,m+n+k+1)/((m+n+k+1.5)*np.pi) + \
                np.power(-1,m-n-k)/((m-n-k-0.5)*np.pi) + np.power(-1,m+n-k)/((m+n-k+0.5)*np.pi)
        term2 = (np.power(-1,m+n+k)-1)/((m+k+1)*np.pi) + (np.power(-1,m+n-k+1)-1)/((m-k)*np.pi)
    else:
        term1 = np.power(-1,m-n+k+1)/((m-n+k+0.5)*np.pi) + np.power(-1,m+n+k+1)/((m+n+k+1.5)*np.pi) + \
                np.power(-1,m-n-k)/((m-n-k-0.5)*np.pi) + np.power(-1,m+n-k)/((m+n-k+0.5)*np.pi)
        term2 = (np.power(-1,m+n+k)-1)/((m+k+1)*np.pi)
    
    return ((2*m+1)/(8*n+4))*term1 + ((2*m+1)/(4*n+2))*term2

def G2_iszero(m, n, k):
    return False

def analytical_G3(m, k): # int f_m'' f_k
    if m == k:
        return -0.5*(m+0.5)*(m+0.5)*np.pi*np.pi
    else:
        return 0
    
def G3_iszero(m, k):
    return m != k
    

def analytical_G4(k): # int f_k
    if isinstance(k, int):
        k = float(k)
    elif isinstance(k, np.ndarray):
        k = k.astype(float)
    return np.power(-1,k)/((k+0.5)*np.pi)


def G4_iszero(k):
    return False


def analytical_G5(k): # int zf_k
    if isinstance(k, int):
        k = float(k)
    elif isinstance(k, np.ndarray):
        k = k.astype(float)
    return 1/((k+0.5)*(k+0.5)*np.pi*np.pi) - np.power(-1,k)/((k+0.5)*np.pi) 

def G5_iszero(k):
    return False

eigbasis_constantAv.add_analytical_tensors({'G1': analytical_G1,
                                            'G1_iszero': G1_iszero,
                                            'G2': analytical_G2,
                                            'G2_iszero': G2_iszero,
                                            'G3': analytical_G3,
                                            'G3_iszero': G3_iszero,
                                            'G4': analytical_G4,
                                            'G4_iszero': G4_iszero,
                                            'G5': analytical_G5,
                                            'G5_iszero': G5_iszero})  

# 2: vertical basis corresponding to the partial-slip condition

def eigbasis_partialslip(M: int, sf: float, Av: float, tol=1e-12, maxits=10):

    """Generates an eigenfunction basis corresponding to a no-stress surface condition and a partial-slip bed condition.
    
    Arguments:
    
    - M:      number of eigenvalues and eigenfunctions to generate
    - sf:     partial-slip parameter
    - Av:     vertical eddy viscosity
    
    """

    # Solve the equation xtan(x)=sf/Av to find the eigenvalues using algebraic Newton-Raphson applied to the function 1/tan(x)-x*(Av/sf), which has the same solutions, but delivers a more stable Newton algorithm
    # first starting guess at pi/4 (otherwise division by zero); then at n*pi

    init_guess = np.array([np.pi / 4 + j * np.pi for j in range(M)])
    stopcriterion_vals = np.power(np.tan(init_guess), -1) - Av*init_guess/sf
    root_eigvals = init_guess[:]

    for _ in range(maxits): # stopping criterion: residual/initial_guess < epsilon
        root_eigvals += (sf*np.tan(root_eigvals) - Av*root_eigvals*np.tan(root_eigvals)*np.tan(root_eigvals)) / (sf + (sf+Av)*np.tan(root_eigvals)*np.tan(root_eigvals))
        stopcriterion_vals = np.power(np.tan(root_eigvals), -1) - Av*root_eigvals/sf
        if np.amax(np.absolute(stopcriterion_vals / init_guess)) < tol:
            break

    def f_ps(z, n):
        try:
            return np.cos(root_eigvals[n] * z)
        except IndexError:
            raise IndexError(f"Index {n} not available in this basis; the maximum index of this basis is {M-1}.")


    def fprime_ps(z, n):
        try:
            return -root_eigvals[n] * np.sin(root_eigvals[n] * z)
        except IndexError:
            raise IndexError(f"Index {n} not available in this basis; the maximum index of this basis is {M-1}.")


    def fdoubleprime_ps(z, n):
        try:
            return -root_eigvals[n]**2 * np.cos(root_eigvals[n] * z)
        except IndexError:
            raise IndexError(f"Index {n} not available in this basis; the maximum index of this basis is {M-1}.")
        

    def inner_prod_ps(m, k):
        return 0.5 + np.sin(2*root_eigvals[k])/(4*root_eigvals[k]) if m == k else 0
    

    def fint_ps(z, n):
        return (1/root_eigvals[n]) * (np.sin(root_eigvals[n] * z) + np.sin(root_eigvals[n]))

    
    basis = TruncationBasis(f_ps, inner_product=inner_prod_ps, derivative_evaluation_function=fprime_ps, second_derivative_evaluation_function=fdoubleprime_ps)
    basis.integrated_evaluation_function = fint_ps


    def analytical_G1(m, n, k):
        return 0.25 * (
            np.sin(root_eigvals[m] + root_eigvals[n] + root_eigvals[k]) / (root_eigvals[m] + root_eigvals[n] + root_eigvals[k]) + \
            np.sin(root_eigvals[m] + root_eigvals[n] - root_eigvals[k]) / (root_eigvals[m] + root_eigvals[n] - root_eigvals[k]) + \
            np.sin(root_eigvals[m] - root_eigvals[n] + root_eigvals[k]) / (root_eigvals[m] - root_eigvals[n] + root_eigvals[k]) + \
            np.sin(root_eigvals[m] - root_eigvals[n] - root_eigvals[k]) / (root_eigvals[m] - root_eigvals[n] - root_eigvals[k])
        )
    
    def analytical_G2(m, n, k):
        term1 = 0.25 * (
            np.sin(root_eigvals[m] - root_eigvals[n] + root_eigvals[k]) / (root_eigvals[m] - root_eigvals[n] + root_eigvals[k]) + \
            np.sin(root_eigvals[m] - root_eigvals[n] - root_eigvals[k]) / (root_eigvals[m] - root_eigvals[n] - root_eigvals[k]) - \
            np.sin(root_eigvals[m] + root_eigvals[n] + root_eigvals[k]) / (root_eigvals[m] + root_eigvals[n] + root_eigvals[k]) - \
            np.sin(root_eigvals[m] + root_eigvals[n] - root_eigvals[k]) / (root_eigvals[m] + root_eigvals[n] - root_eigvals[k])
        )
        if m != k:
            term2 = 0.5 * np.sin(root_eigvals[n]) * (
                np.cos(root_eigvals[m] + root_eigvals[k]) / (root_eigvals[m] + root_eigvals[k]) + \
                np.cos(root_eigvals[m] - root_eigvals[k]) / (root_eigvals[m] - root_eigvals[k]) - \
                1 / (root_eigvals[m] + root_eigvals[k]) - \
                1 / (root_eigvals[m] - root_eigvals[k])
            )
        else:
            term2 = 0.5 * np.sin(root_eigvals[n]) * (
                np.cos(root_eigvals[m] + root_eigvals[k]) / (root_eigvals[m] + root_eigvals[k]) - \
                1 / (root_eigvals[m] + root_eigvals[k])
            )
        return -(root_eigvals[m]/root_eigvals[n]) * (term1 + term2)

    def analytical_G3(m, k):
        return -root_eigvals[k]**2 * inner_prod_ps(m, k)

    def analytical_G4(m):
        return np.sin(root_eigvals[m]) / root_eigvals[m]
    
    def analytical_G5(m):
        return -np.sin(root_eigvals[m]) / root_eigvals[m] + (1 - np.cos(root_eigvals[m])) / (root_eigvals[m]**2)
    
    tensordict_ps = {
        'G1': analytical_G1,
        'G2': analytical_G2,
        'G3': analytical_G3,
        'G4': analytical_G4,
        'G5': analytical_G5
    }

    basis.add_analytical_tensors(tensordict_ps)
    return basis


# 3: temporal basis based on sines (negative index) and cosines (positive index)

def harmonic_time_basis(sigma):

    def h(t, q):
        if q < 0:
            return np.sin(2*np.pi*sigma*q*t)
        elif q == 0:
            return 0.5 * np.sqrt(2) * np.ones_like(t)
        elif q > 0:
            return np.cos(2*np.pi*sigma*q*t)
        
    def h_cf(q):
        if q < 0:
            return ngsolve.sin(2*ngsolve.pi*sigma*q*ngsolve.x)
        elif q == 0:
            return 0.5 * np.sqrt(2)
        elif q > 0:
            return ngsolve.cos(2*ngsolve.pi*sigma*q*ngsolve.x)

    def inner_prod_h(i, j):
            if i == j:
                return 0.5 / sigma
            else:
                return 0
            
    time_basis = TruncationBasis(h, h_cf, inner_prod_h)

    def analytical_H2(p, q):
        if p == -q:
            return np.pi*abs(p)
        else:
            return 0
        
    def H2_iszero(p,q):
        return p != -q
        
    def analytical_H3(p, q, r):
        # To calculate this, we systematically treat all cases for positivity/negativity starting with r, then q, and then p

        if r == 0:
            if p == q:
                return 0.25 * np.sqrt(2) / sigma
            else:
                return 0
        elif r > 0:
            if q == 0:
                if p == r:
                    return 0.25 * np.sqrt(2) / sigma
                else:
                    return 0
            elif q > 0:
                if q == r:
                    if p == 0:
                        return 0.25 * np.sqrt(2) / sigma
                    elif p == 2 * r:
                        return 0.25 / sigma
                    else:
                        return 0
                else:
                    if p == abs(r-q) or p == r+q:
                        return 0.25 / sigma
                    else:
                        return 0
            elif q < 0:
                if q == -r:
                    if p == -2 * r:
                        return 0.25 / sigma
                    else:
                        return 0
                else:
                    if p == -abs(r+q):
                        return 0.25 / sigma if q < -r else -0.25 / sigma
                    elif p == -abs(r-q):
                        return 0.25 / sigma
                    else:
                        return 0
            else:
                return 0
        elif r < 0:
            if q == 0:
                if p == r:
                    return 0.25 * np.sqrt(2) / sigma
                else:
                    return 0
            elif q > 0:
                if q == -r:
                    if p == -abs(r-q):
                        return 0.25 / sigma
                    else:
                        return 0
                else:
                    if p == -abs(r+q):
                        return 0.25 / sigma if q < -r else -0.25 / sigma
                    elif p == -abs(r-q):
                        return 0.25 / sigma
                    else:
                        return 0
            elif q < 0:
                if q == r:
                    if p == 0:
                        return 0.25 * np.sqrt(2) / sigma
                    elif p == abs(r+q):
                        return -0.25 / sigma
                    else:
                        return 0
                else:
                    if p == abs(r+q):
                        return -0.25 / sigma
                    elif p == abs(r-q):
                        return 0.25 / sigma
                    else:
                        return 0
            else:
                return 0
        else:
            return 0
        
    def H3_iszero(p, q, r):
        if r == 0:
            if p == q:
                return False
            else:
                return True
        elif r > 0:
            if q == 0:
                if p == r:
                    return False
                else:
                    return True
            elif q > 0:
                if q == r:
                    if p == 0:
                        return False
                    elif p == 2 * r:
                        return False
                    else:
                        return True
                else:
                    if p == abs(r-q) or p == r+q:
                        return False
                    else:
                        return True
            elif q < 0:
                if q == -r:
                    if p == -2 * r:
                        return False
                    else:
                        return True
                else:
                    if p == -abs(r+q):
                        return False
                    elif p == -abs(r-q):
                        return False
                    else:
                        return True
            else:
                return True
        elif r < 0:
            if q == 0:
                if p == r:
                    return False
                else:
                    return True
            elif q > 0:
                if q == -r:
                    if p == -abs(r-q):
                        return False
                    else:
                        return True
                else:
                    if p == -abs(r+q):
                        return False
                    elif p == -abs(r-q):
                        return False
                    else:
                        return True
            elif q < 0:
                if q == r:
                    if p == 0:
                        return False
                    elif p == abs(r+q):
                        return False
                    else:
                        return True
                else:
                    if p == abs(r+q):
                        return False
                    elif p == abs(r-q):
                        return False
                    else:
                        return False
            else:
                return True
        else:
            return True


        
    time_basis.add_analytical_tensors({'H2': analytical_H2,
                                       'H2_iszero': H2_iszero,
                                       'H3': analytical_H3,
                                       'H3_iszero': H3_iszero})
    
    return time_basis

unit_harmonic_time_basis = harmonic_time_basis(1)


class Projection(object):
    """Object that contains information about a function that was projected onto a finite-dimensional subspace of L^2([-1,0]).

    Attributes:

    - original_function:                function of z that returns value of the original function at z;
    - basis:                            TruncationBasis object; original function is projected on these basis functions;
    - dimension:                        number of basis functions used for the projection;
    - massmatrix:                       mass matrix associated to this projection; two-dimensional numpy array; generated by
                                        methods 'construct_numerical_massmatrix' or 'construct_analytical_massmatrix';
    - sparse_massmatrix:                matrix in CSC format for sparse inversion; generated by 'project_galerkin' in 'sparse'=True;
    - coefficients:                     array of coefficients for each basis function; generated by 'project_galerkin';
    - projected_function:               function of z that returns value of the projection at z; generated by 'project_galerkin';
    
    Methods:
    
    - project_galerkin:                 computes Galerkin projection w.r.t. the basis' weighted inner product;
    - construct_numerical_massmatrix:   constructs mass matrix for this projection using Gauss-Lobatto-Legendre quadrature;
    - construct_analytical_massmatrix:  constructs mass matrix for this projection using the basis' inner product function;
    
    """

    def __init__(self, original_function, basis: TruncationBasis, dimension: int):
        
        self.original_function = original_function
        self.basis = basis
        self.dimension = dimension


    def project_galerkin(self, num_quadrature_points: int, num_mesh_elements: int, sparse=True, minimum_coefficient = 0):
        """Performs a Galerkin projection of the original function onto [dimension] basis functions. Computation of the right-hand
        side vector is done using Gauss-Lobatto-Legendre integration with [num_quadrature_points] points.
        
        Arguments:
        
        - num_quadrature_points:            number of quadrature points for GLL-integration of right-hand side;
        - num_mesh_elements:                number of mesh elements used for the GLL-integration of right-hand side;
        - sparse:                           True or False; if True, mass matrix inversion will be done via scipy.sparse.linalg.spsolve;
        - minimum_coefficient:              minimal value of the coefficients to eliminate rounding errors;
        
        """

        # Construct right-hand side vector
        GLL_points = integrationtools.get_GLL_points(num_quadrature_points)
        mesh_GLL_points = integrationtools.get_mesh_GLL_points(num_mesh_elements, GLL_points, -1, 0)
        f = np.zeros(self.dimension)
        for j in range(self.dimension):
            f[j] = integrationtools.GLL_integrate_mesh(self.original_function(mesh_GLL_points)*self.basis.evaluation_function(mesh_GLL_points, j) * \
                                                  self.basis.weight_function(mesh_GLL_points), GLL_points, xl=-1, xr=0)
        if sparse:
            self.sparse_massmatrix = scipy.sparse.csc_matrix(self.massmatrix)
            self.coefficients = spsolve(self.sparse_massmatrix, f)
        else:
            self.coefficients = np.linalg.solve(self.massmatrix, f) # uses LU-decomposition

        self.coefficients = np.where(np.absolute(self.coefficients) >= minimum_coefficient, self.coefficients, 
                                     np.zeros_like(self.coefficients))

        # Define evaluation function for the Galerkin projection
        def evaluate_projection(z):
            return sum([self.coefficients[i] * self.basis.evaluation_function(z, i) for i in range(self.dimension)])
        
        self.projected_function = evaluate_projection


    def construct_numerical_massmatrix(self, num_quadrature_points: int, num_mesh_elements: int, use_weight_function: bool = False):
        """Constructs a mass matrix for the basis using numerical integration using num_quadrature_points points.
        If one desires an analytical mass matrix, one may assign it directly to the massmatrix attribute without constructing it via
        this method.
        
        Arguments:
        
        - num_quadrature_points:    number of quadrature points in case of numerical construction;
        - num_mesh_elements:        number of elements in the mesh of the numerical integration;
        - use_weight_function:      True or False; if True, then integration will be done using the weight function of the basis."""
        
        self.massmatrix = np.zeros((self.dimension, self.dimension))
        GLL_points = integrationtools.get_GLL_points(num_quadrature_points)
        mesh_GLL_points = integrationtools.get_mesh_GLL_points(num_mesh_elements, GLL_points, -1, 0)
        for i in range(self.dimension):
            for j in range(self.dimension):
                if use_weight_function:
                    values = self.basis.evaluation_function(mesh_GLL_points, i) * self.basis.evaluation_function(mesh_GLL_points, j) \
                             * self.basis.weight_function(mesh_GLL_points)
                else:
                    values = self.basis.evaluation_function(mesh_GLL_points, i) * self.basis.evaluation_function(mesh_GLL_points, j)
                self.massmatrix[i, j] = integrationtools.GLL_integrate_mesh(values, GLL_points, xl=-1, xr=0)
        return self.massmatrix # option to save it to another variable


    def construct_analytical_massmatrix(self):
        """Constructs a mass matrix using the weighted inner product of the associated TruncationBasis object."""
        
        self.massmatrix = np.zeros((self.dimension, self.dimension))
        for i in range(self.dimension):
            for j in range(self.dimension):
                self.massmatrix[i, j] = self.basis.inner_product(i, j)

        return self.massmatrix # option to save it to another variable
