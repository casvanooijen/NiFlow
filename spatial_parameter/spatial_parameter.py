"""
In this file, we create two standard objects for spatially varying parameters: a SpatialParameter class and a VectorSpatialParameter class.
Given a function handle depending on xi and eta
and the gridfunctions xi_gf and eta_gf.
This object generates the grid function variant of the function handle and its gradient automatically.

We also define the vector spatial parameter, for now we assume no derivatives of this class are needed.

sympy is used to symbolically compute the derivative with respectto xi and eta of the spatially variable
parameter

1. Added unittest rectangle and annulus section
2. Added CoefficientFunctions around cf and gradient_cf

#TODO ADDED new class
"""

import numpy as np
import sympy

import ngsolve

from NiFlow.spatial_parameter.boundary_fitted_coordinates import BoundaryFittedCoordinates
from NiFlow.spatial_parameter.fit_and_project import fit_and_project, project


def ngsolve_tanh(x):
    return (ngsolve.exp(2*x) - 1)/(ngsolve.exp(2*x) + 1)

ngsolve_functiondict = {'acos': ngsolve.acos, 'asin': ngsolve.asin, 'atan': ngsolve.atan, 'atan2': ngsolve.atan2,
                        'ceil': ngsolve.ceil, 'cos': ngsolve.cos, 'cosh': ngsolve.cosh, 'exp': ngsolve.exp,
                        'floor': ngsolve.floor, 'log': ngsolve.log, 'pow': ngsolve.pow, 'sin': ngsolve.sin,
                        'sinh': ngsolve.sinh, 'sqrt': ngsolve.sqrt, 'tan': ngsolve.tan, 'tanh': ngsolve_tanh}

class SpatialParameter:
    """
    Class for horizontally spatially varying parameters defined in terms of (xi, eta).
    This file maybe used by other functions throughout the model to use its spatially varying parameters
    """
    def __init__(self, fh, bfc: BoundaryFittedCoordinates):
        """
        Initialization
        Args:
            fh: function handle fh(xi,eta)
            bfc: BoundaryFittedCoordinates object
        """
        self.fh = fh
        self.xi_gf = bfc.xi_gf
        self.eta_gf = bfc.eta_gf

        self.cf = self._generate_coefficientfunction()
        self.gradient_cf = self._generate_gradient_coefficientfunction()


    # Private methods
    def _generate_coefficientfunction(self):
        # Check if function handle is callable
        if callable(self.fh):
            xi, eta = sympy.symbols("xi, eta")
            coefficientfunction = ngsolve.CoefficientFunction(sympy.lambdify([xi, eta], self.fh(xi, eta), ngsolve_functiondict)(self.xi_gf, self.eta_gf))
        else:
            # Constant coefficient function
            coefficientfunction = ngsolve.CoefficientFunction(self.fh)
        return coefficientfunction

    def _generate_gradient_coefficientfunction(self):
        """
           Computes the gradient using a sympy function handle
           Args:
               fh: symbolic sympy function of xi and eta

           Returns:
               gradient of fh: the symbolically computed gradient as a coefficient function
        """
        # Check if function handle is callable
        if callable(self.fh):
            xi, eta = sympy.symbols("xi, eta")
            dfhdxi = sympy.diff(self.fh(xi, eta), xi)
            dfhdxi_cf = ngsolve.CoefficientFunction(sympy.lambdify([xi, eta], dfhdxi, ngsolve_functiondict)(self.xi_gf, self.eta_gf))

            dfhdeta = sympy.diff(self.fh(xi, eta), eta)
            dfhdeta_cf = ngsolve.CoefficientFunction(sympy.lambdify([xi, eta], dfhdeta, ngsolve_functiondict)(self.xi_gf, self.eta_gf))
            gradient = dfhdxi_cf * ngsolve.Grad(self.xi_gf) + dfhdeta_cf * ngsolve.Grad(self.eta_gf)
        else:
            gradient = ngsolve.CoefficientFunction((0, 0))
        return gradient


    # Public methods
    def get_numpy_fh(self):
        # Check if function handle is callable
        if callable(self.fh):
            xi, eta = sympy.symbols("xi, eta")
            fh_numpy = sympy.lambdify([xi, eta], self.fh(xi, eta), "numpy")
        else:
            fh_numpy = lambda xi, eta: self.fh * np.ones_like(xi)
        return fh_numpy



# TODO test
class SpatialParameterFromData:
    """
    Class for horizontally spatially varying parameters that are based on point cloud data
    This file maybe used by other functions throughout the model to use the spatially varying parameters and gradient
    """
    def __init__(self, pointclouddata, fitting_method, projection_method, mesh):
        """
        Initialization
        Args:
            pointclouddata      - numpy array as [[x1, y1, v1], [x2, y2, v2], ...], where xi,yi are the spatial locations and vi the value at that location.
            fitting_method      - method to fit the point cloud data
            projection_method   - method used to project the fit into a NGSolve object (i.e. a grid function)
            mesh                - computational mesh
        Returns:
            SpatialParameter object with cf and gradient_cf
        """

        #TODO go to fitting routine class and projection class
        gf = fit_and_project(pointclouddata, fitting_method, projection_method, mesh)

        self.cf = gf
        self.gradient_cf = ngsolve.Grad(self.cf)



class SpatialParameterFromCoefficientFunction:
    """
    Class for horizontally spatially varying parameters that are based NGSolve CoefficientFunctions.
    This object maybe used by other functions throughout the model to use the spatially varying parameters and gradient
    """
    def __init__(self, coefficient_function, projection_method, mesh):
        """
        Initialization
        Args:
            coefficient_function - Function consisting of the composition of GridFunctions and CoefficientFunctions
            projection_method   - method used to project the fit into a NGSolve object (i.e. a grid function)
            mesh                - computational mesh
        Returns:
            SpatialParameter object with cf and gradient_cf
        """

        #TODO make a function that does this
        gf = project(coefficient_function, projection_method, mesh)

        self.cf = gf
        self.gradient_cf = ngsolve.Grad(self.cf)



class SpatialParameterProduct:
    """Class to generate a spatial parameter from a product of spatial parameters """

    def __init__(self, sp1, sp2, fac=1):
        """
        Initialization
         Create a spatial parameter class from the product of two spatial parameters and a factor
        Args:
            sp1 - spatial parameter class 1
            sp2 - spatial parameter class 2
            factor - constant factor
        Returns:
            SpatialParameter object
        """
        # Set values
        self.cf = fac * sp1.cf * sp2.cf
        # Set derivative using chain rule for the gradient
        self.gradient_cf = fac * (sp1.gradient_cf * sp2.cf + sp1.cf * sp2.gradient_cf)


class SpatialParameterInterpolation:
    """Class to generate a spatial parameter by linearly interpolating between two spatial parameters"""

    def __init__(self, sp1, sp2, alpha):
        """
        Initialization
        Create a spatial parameter by linearly interpolating between two spatial parameters based on the value of alpha
        Args:
            sp1 - spatial parameter class 1
            sp2 - spatial parameter class 2
            alpha - interpolation value, alpha=0 implies sp1 and alpha=1 implies sp2
        Returns:
            SpatialParameter object
        """
        # Set values
        self.cf = (1-alpha) * sp1.cf + alpha * sp2.cf
        # The derivative is linear over summation
        self.gradient_cf = (1-alpha) * sp1.gradient_cf + alpha * sp2.gradient_cf






