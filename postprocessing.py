import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.animation as anim
import ngsolve

from NiFlow.hydrodynamics import Hydrodynamics
from NiFlow.utils import *


## EVALUATE ON DIFFERENT TYPES OF DOMAINS ##

def evaluate_vertical_structure_at_point(mesh: ngsolve.Mesh, quantity_function, p, num_vertical_points: int):
    """Evaluates a variable (quantity_function) from the river bed, to the surface, at a horizontal point p=(p[0], p[1]), in num_vertical_points equally spaced sigma-layer steps.
    
    Arguments:

    - mesh:    mesh on which the variable is defined;
    - quantity_function: function (can be a lambda-function) of sigma that returns the horizontal field of that variable at a specific sigma layer.
    - p: horizontal point at which the function should be evaluated.
    - num_vertical_points: the number of sigma layers that should be evaluated.
    
    """
    sigma_range = np.linspace(-1, 0, num_vertical_points)
    Q = np.zeros_like(sigma_range)

    for i in range(num_vertical_points):
        Q[i] = evaluate_CF_point(quantity_function(sigma_range[i]), mesh, p[0], p[1])

    return Q


def evaluate_vertical_structure_at_cross_section(mesh: ngsolve.Mesh, quantity_function, p1, p2, num_horizontal_points: int, num_vertical_points: int):
    """Evaluates a variable (quantity_function) from the river bed to the surface in the linear cross-section from p1 to p2, at num_horizontal_points equally spaced horizontal points and
    num_vertical_points sigma layers.

    Arguments:

    - mesh: mesh on which the variable is defined.
    - quantity_function: function (can be lambda-function) that takes sigma as argument and returns the horizontal solution field of that variable as an ngsolve.GridFunction or ngsolve.CF.
    - p1: first point to span the cross-section.
    - p2: second point to span the cross-section.
    - num_horizontal_points: number of equally spaced horizontal locations at which the variable is evaluated.
    - num_vertical_points: number of equally spaced sigma layers at which the variable is evaluated.    
    
    """
    sigma_range = np.linspace(-1, 0, num_vertical_points)
    x_range = np.linspace(p1[0], p2[0], num_horizontal_points)
    y_range = np.linspace(p1[1], p2[1], num_horizontal_points)

    Q = np.zeros((num_horizontal_points, num_vertical_points))
    
    for i in range(num_vertical_points):
        Q[i, :] = evaluate_CF_range(quantity_function(sigma_range[i]), mesh, x_range, y_range)

    return Q


class PostProcessing(object):

    """Class that postprocesses the raw solution field in terms of vertical/Fourier-basis coefficients into intepretable solution fields for the velocities and water surface elevation.
    
    Attributes:


    Methods:
    
    """

    def __init__(self, hydro: Hydrodynamics):

        """
        Creates functions that represent the velocity fields in different ways (timed, basis coefficients, depth-averaged, ...).

        Arguments:

        - hydro: hydrodynamics-object of the simulation to postprocess
        """

        self.hydro = hydro

        M = hydro.numerical_information['M']
        imax = hydro.numerical_information['imax']

        x_scaling = hydro.geometric_information['x_scaling']
        y_scaling = hydro.geometric_information['y_scaling']

        # make the linear interpolant functions to treat computational boundary conditions
        if hydro.model_options['sea_boundary_treatment'] == 'exact':
            sea_interpolant = ((hydro.geometric_information['riverine_boundary_x'] / x_scaling) + (hydro.geometric_information['L_BL_river']/x_scaling) + (hydro.geometric_information['L_R_river']/x_scaling) + \
                            (hydro.geometric_information['L_RA_river']/x_scaling) - ngsolve.x) / \
                            ((hydro.geometric_information['riverine_boundary_x']+hydro.geometric_information['L_BL_river']+hydro.geometric_information['L_R_river']+hydro.geometric_information['L_RA_river'] +
                                hydro.geometric_information['L_BL_sea'] + hydro.geometric_information['L_R_sea'] + hydro.geometric_information['L_RA_sea']) / x_scaling)
            sea_interpolant_x = -1 / ((hydro.geometric_information['riverine_boundary_x']+hydro.geometric_information['L_BL_river']+hydro.geometric_information['L_R_river']+hydro.geometric_information['L_RA_river'] +
                                hydro.geometric_information['L_BL_sea'] + hydro.geometric_information['L_R_sea'] + hydro.geometric_information['L_RA_sea']) / x_scaling)
            self.sea_interpolant = sea_interpolant
        if hydro.model_options['river_boundary_treatment'] == 'exact':
            river_interpolant = (-(hydro.geometric_information['L_BL_sea']/x_scaling) - (hydro.geometric_information['L_R_sea']/x_scaling) - \
                                (hydro.geometric_information['L_RA_sea']/x_scaling) + ngsolve.x) / \
                                ((hydro.geometric_information['riverine_boundary_x']+hydro.geometric_information['L_BL_river']+hydro.geometric_information['L_R_river']+hydro.geometric_information['L_RA_river'] +
                                hydro.geometric_information['L_BL_sea'] + hydro.geometric_information['L_R_sea'] + hydro.geometric_information['L_RA_sea']) / x_scaling)
            river_interpolant_x = 1 / ((hydro.geometric_information['riverine_boundary_x']+hydro.geometric_information['L_BL_river']+hydro.geometric_information['L_R_river']+hydro.geometric_information['L_RA_river'] +
                                hydro.geometric_information['L_BL_sea'] + hydro.geometric_information['L_R_sea'] + hydro.geometric_information['L_RA_sea']) / x_scaling)
        

        if hydro.model_options['river_boundary_treatment'] != 'exact':
            self.u = lambda q, sigma : sum([hydro.alpha_solution[m][q] * hydro.vertical_basis.evaluation_function(sigma, m) for m in range(M)])
        else:
            self.u = lambda q, sigma : sum([(hydro.alpha_solution[m][q]+ hydro.Q_solution[q] * hydro.riverine_forcing.normal_alpha[m] * river_interpolant) * hydro.vertical_basis.evaluation_function(sigma, m) for m in range(M)])
            self.eval_Q = lambda q : evaluate_CF_point(hydro.Q_solution[q], hydro.mesh, 0, 0) 
        self.v = lambda q, sigma : sum([hydro.beta_solution[m][q] * hydro.vertical_basis.evaluation_function(sigma, m) for m in range(M)])

        if hydro.model_options['sea_boundary_treatment'] != 'exact':
            self.gamma = lambda q : hydro.gamma_solution[q]
        else:
            self.gamma = lambda q : hydro.gamma_solution[q] + hydro.A_solution[q] * sea_interpolant
            self.eval_A = lambda q : evaluate_CF_point(hydro.A_solution[q], hydro.mesh, 0, 0)

        self.gamma_abs = lambda q: ngsolve.sqrt(self.gamma(q)*self.gamma(q)) if q == 0 else ngsolve.sqrt(self.gamma(q)*self.gamma(q)+self.gamma(-q)*self.gamma(-q))


        if hydro.model_options['river_boundary_treatment'] != 'exact':
            self.u_DA = lambda q: sum([hydro.vertical_basis.tensor_dict['G4'](m) * hydro.alpha_solution[m][q] for m in range(M)])
            self.u_DA_x = lambda q: sum([hydro.vertical_basis.tensor_dict['G4'](m) * ngsolve.grad(hydro.alpha_solution[m][q])[0] for m in range(M)])
            self.u_DA_y = lambda q: sum([hydro.vertical_basis.tensor_dict['G4'](m) * ngsolve.grad(hydro.alpha_solution[m][q])[1] for m in range(M)])
        else:
            normalalphay = hydro.riverine_forcing.normal_alpha_y

            self.u_DA = lambda q: sum([hydro.vertical_basis.tensor_dict['G4'](m) * (hydro.alpha_solution[m][q] + hydro.Q_solution[q]*hydro.riverine_forcing.normal_alpha[m]*river_interpolant) for m in range(M)])
            self.u_DA_x = lambda q: sum([hydro.vertical_basis.tensor_dict['G4'](m) * (ngsolve.grad(hydro.alpha_solution[m][q])[0] + hydro.Q_solution[q]*hydro.riverine_forcing.normal_alpha[m]*river_interpolant_x) for m in range(M)])
            self.u_DA_y = lambda q: sum([hydro.vertical_basis.tensor_dict['G4'](m) * (ngsolve.grad(hydro.alpha_solution[m][q])[1] + hydro.Q_solution[q]*normalalphay[m]*river_interpolant) for m in range(M)])

        self.v_DA = lambda q: sum([hydro.vertical_basis.tensor_dict['G4'](m) * hydro.beta_solution[m][q] for m in range(M)])
        self.v_DA_x = lambda q: sum([hydro.vertical_basis.tensor_dict['G4'](m) * ngsolve.grad(hydro.beta_solution[m][q])[0] for m in range(M)])
        self.v_DA_y = lambda q: sum([hydro.vertical_basis.tensor_dict['G4'](m) * ngsolve.grad(hydro.beta_solution[m][q])[1] for m in range(M)])

        if hydro.model_options['river_boundary_treatment'] != 'exact':
            self.u_timed = lambda t, sigma: sum([sum([hydro.alpha_solution[m][q] * hydro.vertical_basis.evaluation_function(sigma, m) * hydro.time_basis.evaluation_function(t, q) for m in range(M)]) for q in range(-imax, imax+1)])
        else:
            self.u_timed = lambda t, sigma: sum([sum([(hydro.alpha_solution[m][q]+hydro.Q_solution[q]*hydro.riverine_forcing.normal_alpha[m]*river_interpolant) * hydro.vertical_basis.evaluation_function(sigma, m) * hydro.time_basis.evaluation_function(t, q) for m in range(M)]) for q in range(-imax, imax+1)])
        self.v_timed = lambda t, sigma: sum([sum([hydro.beta_solution[m][q] * hydro.vertical_basis.evaluation_function(sigma, m) * hydro.time_basis.evaluation_function(t, q) for m in range(M)]) for q in range(-imax, imax + 1)])

        
        H = hydro.spatial_parameters['H'].cf
        Hx = hydro.spatial_parameters['H'].gradient_cf[0]
        Hy = hydro.spatial_parameters['H'].gradient_cf[1]

        F = dict()
        if hydro.model_options['river_boundary_treatment'] != 'exact':
            F[0] = [H * (ngsolve.grad(hydro.alpha_solution[m][0])[0] / x_scaling + ngsolve.grad(hydro.beta_solution[m][0])[1] / y_scaling) + \
                    hydro.alpha_solution[m][0] * Hx / x_scaling + hydro.beta_solution[m][0] * Hy / y_scaling for m in range(M)]
            for q in range(1, imax + 1):
                F[-q] = [H * (ngsolve.grad(hydro.alpha_solution[m][-q])[0] / x_scaling + ngsolve.grad(hydro.beta_solution[m][-q])[1] / y_scaling) + \
                            hydro.alpha_solution[m][-q] * Hx / x_scaling + hydro.beta_solution[m][-q] * Hy / y_scaling for m in range(M)]
                F[q] = [H * (ngsolve.grad(hydro.alpha_solution[m][q])[0] / x_scaling + ngsolve.grad(hydro.beta_solution[m][q])[1] / y_scaling) + \
                            hydro.alpha_solution[m][q] * Hx / x_scaling + hydro.beta_solution[m][q] * Hy / y_scaling for m in range(M)]
        else:
            F[0] = [H * ((ngsolve.grad(hydro.alpha_solution[m][0])[0]+hydro.Q_solution[0]*hydro.riverine_forcing.normal_alpha[m]*river_interpolant_x) / x_scaling + ngsolve.grad(hydro.beta_solution[m][0])[1] / y_scaling) + \
                    (hydro.alpha_solution[m][0]+hydro.Q_solution[0]*hydro.riverine_forcing.normal_alpha[m]*river_interpolant) * Hx / x_scaling + hydro.beta_solution[m][0] * Hy / y_scaling for m in range(M)]
            for q in range(1, imax + 1):
                F[-q] = [H * ((ngsolve.grad(hydro.alpha_solution[m][-q])[0]+hydro.Q_solution[-q]*hydro.riverine_forcing.normal_alpha[m]*river_interpolant_x) / x_scaling + ngsolve.grad(hydro.beta_solution[m][-q])[1] / y_scaling) + \
                            (hydro.alpha_solution[m][-q]+hydro.Q_solution[-q]*hydro.riverine_forcing.normal_alpha[m]*river_interpolant) * Hx / x_scaling + hydro.beta_solution[m][-q] * Hy / y_scaling for m in range(M)]
                F[q] = [H * ((ngsolve.grad(hydro.alpha_solution[m][q])[0]+hydro.Q_solution[q]*hydro.riverine_forcing.normal_alpha[m]*river_interpolant_x) / x_scaling + ngsolve.grad(hydro.beta_solution[m][q])[1] / y_scaling) + \
                            (hydro.alpha_solution[m][q]+hydro.Q_solution[q]*hydro.riverine_forcing.normal_alpha[m]*river_interpolant) * Hx / x_scaling + hydro.beta_solution[m][q] * Hy / y_scaling for m in range(M)]
            
        self.w = lambda q, sigma : -sum([1/((m+0.5)*np.pi) * F[q][m] * (np.sin((m+0.5)*np.pi*sigma) + minusonepower(m) * np.ones_like(sigma)) for m in range(M)]) + \
                                   sigma * self.u(q, sigma) * Hx / x_scaling + sigma * self.v(q, sigma) * Hy / y_scaling
        self.w_timed = lambda t, sigma : sum([-sum([1/((m+0.5)*np.pi) * F[q][m] * (np.sin((m+0.5)*np.pi*sigma) + minusonepower(m) * np.ones_like(sigma)) for m in range(M)]) for q in range(-imax, imax)]) + \
                                         sigma * self.u_timed(t, sigma) * Hx / x_scaling + sigma * self.v_timed(t, sigma) * Hy / y_scaling


        self.u_abs = lambda q, sigma : ngsolve.sqrt(self.u(q,sigma)*self.u(q,sigma)) if q == 0 else ngsolve.sqrt(self.u(q,sigma)*self.u(q,sigma)+self.u(-q,sigma)*self.u(-q,sigma)) 
        self.v_abs = lambda q, sigma : ngsolve.sqrt(self.v(q,sigma)*self.v(q,sigma)) if q == 0 else ngsolve.sqrt(self.v(q,sigma)*self.v(q,sigma)+self.v(-q,sigma)*self.v(-q,sigma)) 
        self.w_abs = lambda q, sigma : ngsolve.sqrt(self.w(q,sigma)*self.w(q,sigma)) if q == 0 else ngsolve.sqrt(self.w(q,sigma)*self.w(q,sigma)+self.w(-q,sigma)*self.w(-q,sigma))
        self.gamma_abs = lambda q : ngsolve.sqrt(self.gamma(q)*self.gamma(q)) if q == 0 else ngsolve.sqrt(self.gamma(q)*self.gamma(q)+self.gamma(-q)*self.gamma(-q)) 

        self.u_phase = lambda q, sigma: ngsolve.atan2(-self.u(-q, sigma), self.u(q, sigma))
        self.v_phase = lambda q, sigma: ngsolve.atan2(-self.v(-q, sigma), self.v(q, sigma))
        self.w_phase = lambda q, sigma: ngsolve.atan2(-self.w(-q, sigma), self.w(q, sigma))
        self.gamma_phase = lambda q: ngsolve.atan2(-self.gamma(-q), self.gamma(q))

        
        self.u_DA_abs = lambda q: ngsolve.sqrt(self.u_DA(q)*self.u_DA(q)) if q == 0 else ngsolve.sqrt(self.u_DA(q)*self.u_DA(q) + self.u_DA(-q)*self.u_DA(-q))
        self.v_DA_abs = lambda q: ngsolve.sqrt(self.v_DA(q)*self.v_DA(q)) if q == 0 else ngsolve.sqrt(self.v_DA(q)*self.v_DA(q) + self.v_DA(-q)*self.v_DA(-q))

        self.u_DA_phase = lambda q: ngsolve.atan2(-self.u_DA(-q), self.u_DA(q))
        self.v_DA_phase = lambda q: ngsolve.atan2(-self.v_DA(-q), self.v_DA(q))

        # Get derivative gridfunctions

        if hydro.model_options['river_boundary_treatment'] != 'exact':
            self.ux = lambda q, sigma : sum([ngsolve.grad(hydro.alpha_solution[m][q])[0] * hydro.vertical_basis.evaluation_function(sigma, m) for m in range(M)]) / x_scaling
        else:
            self.ux = lambda q, sigma : sum([(ngsolve.grad(hydro.alpha_solution[m][q])[0] + hydro.Q_solution[q]*hydro.riverine_forcing.normal_alpha[m]*river_interpolant_x)* hydro.vertical_basis.evaluation_function(sigma, m) for m in range(M)]) / x_scaling
        self.vx = lambda q, sigma : sum([ngsolve.grad(hydro.beta_solution[m][q])[0] * hydro.vertical_basis.evaluation_function(sigma, m) for m in range(M)]) / x_scaling
        if hydro.model_options['sea_boundary_treatment'] != 'exact':
            self.gammax = lambda q: ngsolve.grad(hydro.gamma_solution[q])[0] / x_scaling
        else:
            self.gammax = lambda q: (ngsolve.grad(hydro.gamma_solution[q])[0] + hydro.A_solution[q] * sea_interpolant_x) / x_scaling
        if hydro.model_options['river_boundary_treatment'] != 'exact':
            self.uy = lambda q, sigma : sum([ngsolve.grad(hydro.alpha_solution[m][q])[1] * hydro.vertical_basis.evaluation_function(sigma, m) for m in range(M)]) / y_scaling 
        else:
            self.uy = lambda q, sigma : sum([(ngsolve.grad(hydro.alpha_solution[m][q])[1] + hydro.Q_solution[q]*normalalphay[m]*river_interpolant) * hydro.vertical_basis.evaluation_function(sigma, m) for m in range(M)]) / y_scaling 
        self.vy = lambda q, sigma : sum([ngsolve.grad(hydro.beta_solution[m][q])[1] * hydro.vertical_basis.evaluation_function(sigma, m) for m in range(M)]) / y_scaling
        self.gammay = lambda q: ngsolve.grad(hydro.gamma_solution[q])[1] / y_scaling

        if hydro.model_options['river_boundary_treatment'] != 'exact':
            self.usig = lambda q, sigma : sum([hydro.alpha_solution[m][q] * hydro.vertical_basis.derivative_evaluation_function(sigma, m) for m in range(M)]) #This is the derivative w.r.t. sigma, and not z. To transform this to the derivative w.r.t. z, divide by H
        else:
            self.usig = lambda q, sigma : sum([(hydro.alpha_solution[m][q] + hydro.Q_solution[q]*hydro.riverine_forcing.normal_alpha[m]*river_interpolant)* hydro.vertical_basis.derivative_evaluation_function(sigma, m) for m in range(M)])
        self.vsig = lambda q, sigma : sum([hydro.beta_solution[m][q] * hydro.vertical_basis.derivative_evaluation_function(sigma, m) for m in range(M)]) #same

        if hydro.model_options['river_boundary_treatment'] != 'exact':
            self.usigsig = lambda q, sigma: sum([hydro.alpha_solution[m][q] * hydro.vertical_basis.second_derivative_evaluation_function(sigma, m) for m in range(M)]) #This is the derivative w.r.t. sigma, and not z. To transform this to the derivative w.r.t. z, divide by H^2
        else:
            self.usigsig = lambda q, sigma: sum([(hydro.alpha_solution[m][q] + hydro.Q_solution[q]*hydro.riverine_forcing.normal_alpha[m]*river_interpolant) * hydro.vertical_basis.second_derivative_evaluation_function(sigma, m) for m in range(M)])
        self.vsigsig = lambda q, sigma: sum([hydro.beta_solution[m][q] * hydro.vertical_basis.second_derivative_evaluation_function(sigma, m) for m in range(M)]) #This is the derivative w.r.t. sigma, and not z. To transform this to the derivative w.r.t. z, divide by H^2

        if hydro.model_options['river_boundary_treatment'] != 'exact':
            self.ux_timed = lambda t, sigma: sum([sum([ngsolve.grad(hydro.alpha_solution[m][q])[0] * hydro.vertical_basis.evaluation_function(sigma, m) * hydro.time_basis.evaluation_function(t, q) for m in range(M)]) for q in range(-imax, imax+1)]) / x_scaling
        else:
            self.ux_timed = lambda t, sigma: sum([sum([(ngsolve.grad(hydro.alpha_solution[m][q])[0] + hydro.Q_solution[q]*hydro.riverine_forcing.normal_alpha[m]*river_interpolant_x) * hydro.vertical_basis.evaluation_function(sigma, m) * hydro.time_basis.evaluation_function(t, q) for m in range(M)]) for q in range(-imax, imax+1)]) / x_scaling
        self.vx_timed = lambda t, sigma: sum([sum([ngsolve.grad(hydro.beta_solution[m][q])[0] * hydro.vertical_basis.evaluation_function(sigma, m) * hydro.time_basis.evaluation_function(t, q) for m in range(M)]) for q in range(-imax, imax + 1)]) / x_scaling

        if hydro.model_options['river_boundary_treatment'] != 'exact':
            self.uy_timed = lambda t, sigma: sum([sum([ngsolve.grad(hydro.alpha_solution[m][q])[1] * hydro.vertical_basis.evaluation_function(sigma, m) * hydro.time_basis.evaluation_function(t, q) for m in range(M)]) for q in range(-imax, imax+1)]) / y_scaling
        else:
            self.uy_timed = lambda t, sigma: sum([sum([(ngsolve.grad(hydro.alpha_solution[m][q])[1] + hydro.Q_solution[q] * normalalphay[m] * river_interpolant) * hydro.vertical_basis.evaluation_function(sigma, m) * hydro.time_basis.evaluation_function(t, q) for m in range(M)]) for q in range(-imax, imax+1)]) / y_scaling
        
        self.vy_timed = lambda t, sigma: sum([sum([ngsolve.grad(hydro.beta_solution[m][q])[1] * hydro.vertical_basis.evaluation_function(sigma, m) * hydro.time_basis.evaluation_function(t, q) for m in range(M)]) for q in range(-imax, imax + 1)]) / y_scaling

        if hydro.model_options['river_boundary_treatment'] != 'exact':
            self.usig_timed = lambda t, sigma: sum([sum([hydro.alpha_solution[m][q] * hydro.vertical_basis.derivative_evaluation_function(sigma, m) * hydro.time_basis.evaluation_function(t, q) for m in range(M)]) for q in range(-imax, imax+1)])
        else:
            self.usig_timed = lambda t, sigma: sum([sum([(hydro.alpha_solution[m][q] + hydro.Q_solution[q] * hydro.riverine_forcing.normal_alpha[m] * river_interpolant) * hydro.vertical_basis.derivative_evaluation_function(sigma, m) * hydro.time_basis.evaluation_function(t, q) for m in range(M)]) for q in range(-imax, imax+1)])
        self.vsig_timed = lambda t, sigma: sum([sum([hydro.beta_solution[m][q] * hydro.vertical_basis.derivative_evaluation_function(sigma, m) * hydro.time_basis.evaluation_function(t, q) for m in range(M)]) for q in range(-imax, imax + 1)])

    ## PLOTTING ##
        
    def plot_mesh_wireframe(self, title: str=None, save: str=None, **kwargs):

        """
        Plots the computational mesh of the model simulation as a wireframe.

        Arguments:

        - title (str): title of the plot
        - save (str): name of the file this plot will be saved to; if None, then the plot is not saved.
        - **kwargs: keyword arguments for matplotlib triplot in case of triangular mesh, and matplotlib hlines/vlines in case of rectangular elements.
        
        """
        if self.hydro.numerical_information['mesh_generation_method'] != 'structured_quads':
            coords = mesh_to_coordinate_array(self.hydro.mesh.ngmesh)
            triangles = mesh2d_to_triangles(self.hydro.mesh.ngmesh)
            triangulation = tri.Triangulation(coords[:, 0], coords[:, 1], triangles)

            fig_mesh, ax_mesh = plt.subplots()
            ax_mesh.triplot(triangulation, **kwargs)
        else:
            x = np.linspace(0, 1, self.hydro.numerical_information['grid_size'][0] + 1)
            y = np.linspace(-0.5, 0.5, self.hydro.numerical_information['grid_size'][1] + 1)

            fig_mesh, ax_mesh = plt.subplots()
            ax_mesh.hlines(y, xmin=0, xmax=1, **kwargs)
            ax_mesh.vlines(x, ymin=-0.5, ymax=0.5, **kwargs)

        if title:
            ax_mesh.set_title(title)

        if save:
            fig_mesh.savefig(save)

        return fig_mesh, ax_mesh
    

    def plot_horizontal(self, quantity: ngsolve.CoefficientFunction, refinement_level: int=1, exclude_ramping_zone: bool=True, 
                        title: str='Colormap', clabel: str='Color', center_range: bool=False, contourlines: bool=True, num_levels: int=10,
                        subamplitude_lines: int=2, save: str=None, figsize: tuple=(7,4), show_mesh: bool =False, continuous: bool=True, **kwargs):
        """
        Plots a variable horizontally using colours; no visualisation of any vertical structure.
        
        Arguments:

        - quantity: variable to plot. Must be ngsolve.CoefficientFunction or ngsolve.GridFunction (which is a child class of ngsolve.CoefficientFunction).
        - refinement_level: this parameter indicates on how many points the variable is evaluated and hence controls the resolution; to achieve this, the mesh is refined refinement_level times.
        - exclude_ramping_zone: flag to indicate whether the variable should only be shown on the display mesh (which excludes the ramping zone).
        - title: title of the plot.
        - clabel: label of the colorbar next to the plot; should indicate the name of the variable and unit.
        - center_range: flag to indicate whether the range of the color bar should be symmetric (i.e. 0 corresponds to the middle of the color bar); strongly recommended for variables that can become negative.
        - contourlines: flag to indicate whether contour lines should be included in the plot; the labels at these contourlines are rounded to four decimals.
        - num_contourlines: number of contour lines if contourlines=True.
        - subamplitude_lines: number of sub-contour lines inbetween each contour line
        - save: name of the file this plot is saved to; if set to None, the plot is not saved.
        - figsize: size of the plot.
        - show_mesh: flag to indicate whether a wireframe of the mesh should be plotted over the colour.
        - continuous: if False, then constant values will be used inbetween the contours.
        - **kwargs: keyword arguments for matplotlib tripcolor or matplotlib pcolormesh
        
        """

        x_scaling = self.hydro.geometric_information['x_scaling']
        y_scaling = self.hydro.geometric_information['y_scaling']

        if self.hydro.numerical_information['mesh_generation_method'] != 'structured_quads':

            if exclude_ramping_zone:
                triangulation = get_triangulation(self.hydro.display_mesh.ngmesh)
            else:
                triangulation = get_triangulation(self.hydro.mesh.ngmesh)

            refiner = tri.UniformTriRefiner(triangulation)
            refined_triangulation = refiner.refine_triangulation(subdiv=refinement_level)
            
            eval_gfu = evaluate_CF_range(quantity, self.hydro.mesh, refined_triangulation.x, refined_triangulation.y)

            if center_range:
                maxamp = max(np.amax(eval_gfu), -np.amin(eval_gfu))

            fig_colormap, ax_colormap = plt.subplots(figsize=figsize)

            if show_mesh:
                ax_colormap.triplot(triangulation, linewidth=0.5, color='k', zorder=2)

            if center_range:
                if continuous:
                    colormesh = ax_colormap.tripcolor(refined_triangulation, eval_gfu, vmin=-maxamp, vmax=maxamp, cmap='RdBu', **kwargs)
                else:
                    colormesh = ax_colormap.tricontourf(refined_triangulation, eval_gfu, vmin=-maxamp, vmax=maxamp, cmap='RdBu', **kwargs)
            else:
                if continuous:
                    colormesh = ax_colormap.tripcolor(refined_triangulation, eval_gfu, **kwargs)
                else:
                    colormesh = ax_colormap.tricontourf(refined_triangulation, eval_gfu, **kwargs)

            if contourlines:
                try:
                    levels = np.linspace(np.min(eval_gfu), np.max(eval_gfu), num_levels*(subamplitude_lines+1))
                    contour = ax_colormap.tricontour(refined_triangulation, eval_gfu, levels, colors=['k'] + ["0.4"] * subamplitude_lines, linewidths=[.5] * (1+subamplitude_lines))
                    ax_colormap.clabel(contour, levels[0::subamplitude_lines+1], inline=1, fontsize=10, fmt='%1.4f')
                except ValueError:
                    print("Constant solution; plotting contour lines impossible")

        else:
            if exclude_ramping_zone:
                x = np.linspace(0, self.hydro.geometric_information['riverine_boundary_x'] / self.hydro.geometric_information['x_scaling'], self.hydro.numerical_information['grid_size'][0] * (refinement_level + 1) + 1) # This is a temporary solution; we assume now that if a structured grid is used, then the (scaled) domain is a unit square
            else:
                x = np.linspace((-self.hydro.geometric_information['L_BL_sea']-self.hydro.geometric_information['L_R_sea']-self.hydro.geometric_information['L_RA_sea'])/x_scaling,
                                (self.hydro.geometric_information['riverine_boundary_x']+self.hydro.geometric_information['L_RA_river']+self.hydro.geometric_information['L_R_river'] + self.hydro.geometric_information['L_BL_river'])/x_scaling,
                                self.hydro.numerical_information['grid_size'][0] * (refinement_level + 1) + 1)
            y = np.linspace(-0.5, 0.5, self.hydro.numerical_information['grid_size'][1] * (refinement_level + 1) + 1)
            X, Y = np.meshgrid(x, y, indexing='ij')
            Q = np.zeros_like(X)

            for i in range(Q.shape[1]):
                Q[:, i] = evaluate_CF_range(quantity, self.hydro.mesh, x, y[i] * np.ones_like(x))

            if center_range:
                maxamp = np.amax(np.absolute(Q.flatten()))
                
            fig_colormap, ax_colormap = plt.subplots(figsize=figsize)

            if show_mesh:
                if exclude_ramping_zone:
                    x = np.linspace(0, self.hydro.geometric_information['riverine_boundary_x'], self.hydro.numerical_information['grid_size'][0] + 1) # This is a temporary solution; we assume now that if a structured grid is used, then the (scaled) domain is a unit square
                else:
                    x = np.linspace((-self.hydro.geometric_information['L_BL_sea']-self.hydro.geometric_information['L_R_sea']-self.hydro.geometric_information['L_RA_sea'])/x_scaling,
                                    (self.hydro.geometric_information['riverine_boundary_x']+self.hydro.geometric_information['L_RA_river']+self.hydro.geometric_information['L_R_river'] + self.hydro.geometric_information['L_BL_river'])/x_scaling,
                                    self.hydro.numerical_information['grid_size'][0] + 1)
                y = np.linspace(-0.5, 0.5, self.hydro.numerical_information['grid_size'][1] + 1)
                if exclude_ramping_zone:
                    ax_colormap.hlines(y, xmin=0, xmax=1, color='k', linewidth=0.5, zorder=2)
                else:
                    ax_colormap.hlines(y, xmin=-self.hydro.geometric_information['L_BL_sea']-self.hydro.geometric_information['L_R_sea']-self.hydro.geometric_information['L_RA_sea'],
                                       xmax = self.hydro.geometric_information['riverine_boundary_x']+self.hydro.geometric_information['L_RA_river']+self.hydro.geometric_information['L_R_river'] + self.hydro.geometric_information['L_BL_river'],
                                       color='k', linewidth=0.5, zorder=2)
                ax_colormap.vlines(x, ymin=-0.5, ymax=0.5, color='k', linewidth=0.5, zorder=2)

            if center_range:
                if continuous:
                    colormesh = ax_colormap.pcolormesh(X, Y, Q, vmin=-maxamp, vmax=maxamp, cmap='RdBu', **kwargs)
                else:
                    colormesh = ax_colormap.contourf(X, Y, Q, vmin=-maxamp, vmax=maxamp, cmap='RdBu')
            else:
                if continuous:
                    colormesh = ax_colormap.pcolormesh(X, Y, Q, **kwargs)
                else:
                    colormesh = ax_colormap.contourf(X, Y, Q, **kwargs)
            
            if contourlines:
                try:
                    levels = np.linspace(np.min(Q.flatten()), np.max(Q.flatten()), num_levels*(subamplitude_lines+1))
                    contour = ax_colormap.contour(X, Y, Q, levels, colors=['k'] + ["0.4"] * subamplitude_lines, linewidths=[.5] * (1+subamplitude_lines))
                    ax_colormap.clabel(contour, levels[0::subamplitude_lines+1], inline=1, fontsize=10, fmt='%1.4f')
                except ValueError:
                    print("Constant solution; plotting contour lines impossible")

        ax_colormap.set_title(title)
        cbar = fig_colormap.colorbar(colormesh)
        cbar.ax.set_ylabel(clabel)
        
        if exclude_ramping_zone:
            x_ticks = list(np.linspace(0, 1, 11)) # Also a temporary solution; more domain types and compatibility with these functions will be added in the future
        else:
            x_ticks = list(np.linspace((-self.hydro.geometric_information['L_BL_sea']-self.hydro.geometric_information['L_R_sea']-self.hydro.geometric_information['L_RA_sea'])/x_scaling,
                                (self.hydro.geometric_information['riverine_boundary_x']+self.hydro.geometric_information['L_RA_river']+self.hydro.geometric_information['L_R_river'] + self.hydro.geometric_information['L_BL_river'])/x_scaling,
                                11))
        y_ticks = list(np.linspace(-0.5, 0.5, 5))
        ax_colormap.set_xticks(x_ticks)
        ax_colormap.set_yticks(y_ticks)
        ax_colormap.set_xticklabels(np.round(np.array(x_ticks) * self.hydro.geometric_information['x_scaling'] / 1e3, 1))
        ax_colormap.set_yticklabels(np.round(np.array(y_ticks) * self.hydro.geometric_information['y_scaling'] / 1e3, 1))

        ax_colormap.set_xlabel('x [km]')
        ax_colormap.set_ylabel('y [km]')

        if save is not None:
            fig_colormap.savefig(save)

        plt.tight_layout()
        return fig_colormap, ax_colormap


    def plot_horizontal_vectorfield(self, x_field, y_field, background_colorfield=None, num_x:int=40, num_y:int=40, arrow_color='white', title: str='Vector Field', clabel:str='Colour',
                                    save: str=None, exclude_ramping_zone: bool=True, center_range:bool=False, **kwargs):
        """
        Plots a vector field, where the transparency of the arrows denote the magnitude of the vector. A background colour field can also be provided.
        Works only for rectangular domains.

        Arguments:

        - x_field: CoefficientFunction (or GridFunction) of the x-component of the vector field.
        - y_field: CoefficientFunction (or GridFunction) of the y-component of the vector field.
        - background_colorfield: CoefficientFunction (or GridFunction) of the background colour field.
        - num_x: number of arrows in x-direction
        - num_y: number of arrows in y-direction
        - arrow_color: colour of the arrows.
        - title: title of the plot.
        - clabel: label next to the colour bar of the background colour field.
        - save: name of the file this plot is saved to; if set to None, the plot is not saved.
        - exclude_ramping_zone: flag indicating whether the ramping zone should be excluded from the plot.
        - center_range: flag indicating whether the background colourfield's colour bar should be symmetric around 0.
        - **kwargs: keyword arguments for pcolormesh of the background colourfield.

        """

        if exclude_ramping_zone:
            xquiv = np.linspace(0, self.hydro.geometric_information['riverine_boundary_x'], num_x) # This is a temporary solution; we assume now that if a structured grid is used, then the (scaled) domain is a unit square
            xbackground = np.linspace(0, self.hydro.geometric_information['riverine_boundary_x'], 500) # This is a temporary solution; we assume now that if a structured grid is used, then the (scaled) domain is a unit square
        else:
            xquiv = np.linspace(-self.hydro.geometric_information['L_BL_sea']-self.hydro.geometric_information['L_R_sea']-self.hydro.geometric_information['L_RA_sea'],
                            self.hydro.geometric_information['riverine_boundary_x']+self.hydro.geometric_information['L_RA_river']+self.hydro.geometric_information['L_R_river'] + self.hydro.geometric_information['L_BL_river'],
                            num_x)
            xbackground = np.linspace(-self.hydro.geometric_information['L_BL_sea']-self.hydro.geometric_information['L_R_sea']-self.hydro.geometric_information['L_RA_sea'],
                            self.hydro.geometric_information['riverine_boundary_x']+self.hydro.geometric_information['L_RA_river']+self.hydro.geometric_information['L_R_river'] + self.hydro.geometric_information['L_BL_river'],
                            500)
        yquiv = np.linspace(-0.5, 0.5, num_y)
        ybackground = np.linspace(-0.5, 0.5, 500)
        X, Y = np.meshgrid(xquiv, yquiv, indexing='ij')
        Xbackground, Ybackground = np.meshgrid(xbackground, ybackground, indexing='ij')
        Xquiv = np.zeros_like(X)
        Yquiv = np.zeros_like(Y)
        C = np.zeros_like(Xbackground)

        for i in range(Xquiv.shape[1]):
            Xquiv[:, i] = evaluate_CF_range(x_field, self.hydro.mesh, xquiv, yquiv[i] * np.ones_like(xquiv))
            Yquiv[:, i] = evaluate_CF_range(y_field, self.hydro.mesh, xquiv, yquiv[i] * np.ones_like(xquiv))
        if background_colorfield is not None:
            for i in range(Xbackground.shape[1]):
                C[:, i] = evaluate_CF_range(background_colorfield, self.hydro.mesh, xbackground, ybackground[i] * np.ones_like(xbackground))

            fig, ax = plt.subplots()

            if center_range:
                maxamp = np.amax(np.absolute(C))
                background = ax.pcolormesh(Xbackground, Ybackground, C, vmin=-maxamp, vmax=maxamp, cmap='RdBu', **kwargs)
            else:
                background = ax.pcolormesh(Xbackground, Ybackground, C, **kwargs)

            cbar = fig.colorbar(background)
            cbar.ax.set_ylabel(clabel)

        norms = np.sqrt(Xquiv**2 + Yquiv**2)

        arrows = ax.quiver(X, Y, Xquiv/norms, Yquiv/norms, color=arrow_color, pivot='mid', alpha=norms)

        ax.set_title(title + f'\nMaximum magnitude of arrows: {np.amax(norms)}')
        
        if exclude_ramping_zone:
            x_ticks = list(np.linspace(0, 1, 11)) # Also a temporary solution; more domain types and compatibility with these functions will be added in the future
        else:
            x_ticks = list(np.linspace(-self.hydro.geometric_information['L_BL_sea']-self.hydro.geometric_information['L_R_sea']-self.hydro.geometric_information['L_RA_sea'],
                                self.hydro.geometric_information['riverine_boundary_x']+self.hydro.geometric_information['L_RA_river']+self.hydro.geometric_information['L_R_river'] + self.hydro.geometric_information['L_BL_river'],
                                11))
        y_ticks = list(np.linspace(-0.5, 0.5, 5))
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
        ax.set_xticklabels(np.round(np.array(x_ticks) * self.hydro.geometric_information['x_scaling'] / 1e3, 1))
        ax.set_yticklabels(np.round(np.array(y_ticks) * self.hydro.geometric_information['y_scaling'] / 1e3, 1))

        ax.set_xlabel('x [km]')
        ax.set_ylabel('y [km]')

        if save is not None:
            fig.savefig(save)

        plt.tight_layout()
        return fig, ax
    


    def plot_vertical_profile_at_point(self, p, num_vertical_points, constituent_index, **kwargs):
        
        H = self.hydro.spatial_parameters['H'].cf
        depth = evaluate_CF_point(H, self.hydro.mesh, p[0], p[1])
        z_range = np.linspace(-depth, 0, num_vertical_points)

        if constituent_index == 0:
            u_discrete = evaluate_vertical_structure_at_point(self.hydro.mesh, lambda sigma: self.u(0, sigma), p, num_vertical_points)
            v_discrete = evaluate_vertical_structure_at_point(self.hydro.mesh, lambda sigma: self.v(0, sigma), p, num_vertical_points)
            w_discrete = evaluate_vertical_structure_at_point(self.hydro.mesh, lambda sigma: self.w(0, sigma), p, num_vertical_points)
        elif constituent_index > 0:            
            u_discrete = evaluate_vertical_structure_at_point(self.hydro.mesh, lambda sigma: self.u_abs(constituent_index, sigma), p, num_vertical_points)
            v_discrete = evaluate_vertical_structure_at_point(self.hydro.mesh, lambda sigma: self.v_abs(constituent_index, sigma), p, num_vertical_points)
            w_discrete = evaluate_vertical_structure_at_point(self.hydro.mesh, lambda sigma: self.w_abs(constituent_index, sigma), p, num_vertical_points)

        fig_vertical_profile_point, ax_vertical_profile_point = plt.subplots(1,3)
        ax_vertical_profile_point[0].plot(u_discrete, z_range, label='u', **kwargs)
        ax_vertical_profile_point[0].set_title('u')
        ax_vertical_profile_point[0].axvline(x=0, color='k', linewidth=1.5)
        ax_vertical_profile_point[1].plot(v_discrete, z_range, label='v', **kwargs)
        ax_vertical_profile_point[1].set_title('v')
        ax_vertical_profile_point[1].axvline(x=0, color='k', linewidth=1.5)
        ax_vertical_profile_point[2].plot(w_discrete, z_range, label='w', **kwargs)
        ax_vertical_profile_point[2].set_title('w')
        ax_vertical_profile_point[2].axvline(x=0, color='k', linewidth=1.5)
        for i in range(3):
            ax_vertical_profile_point[i].set_ylabel('Depth [m]')
            ax_vertical_profile_point[i].set_xlabel('Velocity [m/s]')

        constituent_string = f'M{2*constituent_index}'

        plt.suptitle(f'Vertical structure of {constituent_string} velocities at x={p[0]}, y={p[1]}')
        plt.tight_layout()


    def plot_vertical_cross_section(self, quantity_function, p1, p2, num_horizontal_points=500, num_vertical_points=500, title='Cross-section', clabel='Color', center_range=False, save=None, contourlines=True, num_levels=None, figsize=(12,6), **kwargs):
        
        scaling_vec = np.array([self.hydro.model_options['x_scaling'], self.hydro.model_options['y_scaling']])
        width = np.linalg.norm((p1-p2) * scaling_vec, 2) / 1e3
        s_range = np.linspace(width/2, -width/2, num_horizontal_points)
        x_range = np.linspace(p1[0], p2[0], num_horizontal_points)
        y_range = np.linspace(p1[1], p2[1], num_horizontal_points)

        H = self.hydro.spatial_parameters['H'].cf

        depth = evaluate_CF_range(H, self.hydro.mesh, x_range, y_range)

        s_grid = np.tile(s_range, (num_vertical_points, 1))
        z_grid = np.array([np.linspace(-depth[i], 0, num_vertical_points) for i in range(num_horizontal_points)]).T

        Q = evaluate_vertical_structure_at_cross_section(self.hydro.mesh, quantity_function, p1, p2, num_horizontal_points, num_vertical_points)

        if center_range:
            maxamp = max(np.amax(Q), -np.amin(Q))


        fig_crosssection, ax_crosssection = plt.subplots(figsize=figsize)
        if center_range:
            color_crosssection = ax_crosssection.pcolormesh(s_grid, z_grid, Q, vmin=-maxamp, vmax=maxamp, **kwargs)
        else:
            color_crosssection = ax_crosssection.pcolormesh(s_grid, z_grid, Q, **kwargs)
        cbar_crosssection = plt.colorbar(color_crosssection)
        cbar_crosssection.ax.set_ylabel(clabel)

        ax_crosssection.plot(s_range, -depth, linewidth=1, color='k', zorder=3)
        ax_crosssection.fill_between(s_range, -np.amax(depth), -depth, color='silver')

        if contourlines:

            if num_levels is None:
                num_levels = 8
            subamplitude_lines = 2

            levels = np.linspace(np.min(Q), np.max(Q), num_levels*(subamplitude_lines+1))
            contour = ax_crosssection.contour(s_grid, z_grid, Q, levels, colors=['k'] + ["0.4"] * subamplitude_lines, linewidths=[.5] * (1+subamplitude_lines))
            ax_crosssection.clabel(contour, levels[0::subamplitude_lines+1], inline=1, fontsize=10, fmt='%1.4f')
            
        ax_crosssection.set_title(title)
        ax_crosssection.set_xlabel('Distance along cross-section [km]')
        ax_crosssection.set_ylabel('-Depth [m]')

        plt.tight_layout()

        if save is not None:
            fig_crosssection.savefig(save)


    def plot_cross_section_circulation(self, p1: np.ndarray, p2: np.ndarray, num_horizontal_points: int, num_vertical_points: int, stride: int, phase: float = 0, constituent='all', flowrange: tuple=None):
        width = np.linalg.norm(p1-p2, 2)
        s_range = np.linspace(-width/2, width/2, num_horizontal_points)
        x_range = np.linspace(p1[0], p2[0], num_horizontal_points)
        y_range = np.linspace(p1[1], p2[1], num_horizontal_points)

        H = self.hydro.spatial_physical_parameters['H'].cf

        depth = evaluate_CF_range(H, self.hydro.mesh, x_range, y_range)

        s_grid = np.tile(s_range, (num_vertical_points, 1))
        z_grid = np.array([np.linspace(-depth[i], 0, num_vertical_points) for i in range(num_horizontal_points)]).T

        if constituent == 'all':
            Q = evaluate_vertical_structure_at_cross_section(self.hydro.mesh, lambda sig: self.u_timed(phase / self.hydro.constant_physical_parameters['sigma'], sig), p1, p2, num_horizontal_points, num_vertical_points)
            V = evaluate_vertical_structure_at_cross_section(self.hydro.mesh, lambda sig: self.v_timed(phase/ self.hydro.constant_physical_parameters['sigma'], sig), p1, p2, num_horizontal_points, num_vertical_points)
            W = evaluate_vertical_structure_at_cross_section(self.hydro.mesh, lambda sig: self.w_timed(phase/ self.hydro.constant_physical_parameters['sigma'], sig), p1, p2, num_horizontal_points, num_vertical_points)
        elif constituent == 0:
            Q = evaluate_vertical_structure_at_cross_section(self.hydro.mesh, lambda sig: self.u(constituent, sig) * self.hydro.time_basis.evaluation_function(phase / self.hydro.constant_physical_parameters['sigma'], constituent), p1, p2,
                                                             num_horizontal_points, num_vertical_points)
            V = evaluate_vertical_structure_at_cross_section(self.hydro.mesh, lambda sig: self.v(constituent, sig) * self.hydro.time_basis.evaluation_function(phase / self.hydro.constant_physical_parameters['sigma'], constituent), p1, p2,
                                                             num_horizontal_points, num_vertical_points)
            W = evaluate_vertical_structure_at_cross_section(self.hydro.mesh, lambda sig: self.w(constituent, sig) * self.hydro.time_basis.evaluation_function(phase / self.hydro.constant_physical_parameters['sigma'], constituent), p1, p2,
                                                             num_horizontal_points, num_vertical_points)
        else:
            Q = evaluate_vertical_structure_at_cross_section(self.hydro.mesh, lambda sig: self.u(constituent, sig) * self.hydro.time_basis.evaluation_function(phase / self.hydro.constant_physical_parameters['sigma'], constituent) + \
                                                             self.u(-constituent, sig) * self.hydro.time_basis.evaluation_function(phase/self.hydro.constant_physical_parameters['sigma'], -constituent), p1, p2,
                                                             num_horizontal_points, num_vertical_points)
            V = evaluate_vertical_structure_at_cross_section(self.hydro.mesh, lambda sig: self.v(constituent, sig) * self.hydro.time_basis.evaluation_function(phase / self.hydro.constant_physical_parameters['sigma'], constituent) + \
                                                             self.v(-constituent, sig) * self.hydro.time_basis.evaluation_function(phase/self.hydro.constant_physical_parameters['sigma'], -constituent), p1, p2,
                                                             num_horizontal_points, num_vertical_points)
            W = evaluate_vertical_structure_at_cross_section(self.hydro.mesh, lambda sig: self.w(constituent, sig) * self.hydro.time_basis.evaluation_function(phase / self.hydro.constant_physical_parameters['sigma'], constituent) + \
                                                             self.w(-constituent, sig) * self.hydro.time_basis.evaluation_function(phase/self.hydro.constant_physical_parameters['sigma'], -constituent), p1, p2,
                                                             num_horizontal_points, num_vertical_points)
        
        if flowrange is None:
            maxamp = max(np.amax(Q), -np.amin(Q))


        fig_crosssection, ax_crosssection = plt.subplots()
        if flowrange is None:
            color_crosssection = ax_crosssection.pcolormesh(s_grid, z_grid, Q, vmin=-maxamp, vmax=maxamp, cmap='bwr')
        cbar_crosssection = plt.colorbar(color_crosssection)
        cbar_crosssection.ax.set_ylabel('Longitudinal velocity [m/s]')

        ax_crosssection.plot(s_range, -depth, linewidth=1, color='k', zorder=3)
        ax_crosssection.fill_between(s_range, -np.amax(depth), -depth, color='silver')

        

        visual_norms = np.sqrt((V[::stride,::stride] / width)**2 + (W[::stride,::stride] / np.amax(depth))**2) # norm of the vector that we plot
        physical_norms = np.sqrt((V[::stride,::stride])**2 + (W[::stride,::stride])**2)

        # ax_crosssection.quiver(s_grid[::stride,::stride], z_grid[::stride,::stride], V[::stride,::stride] / width / 10, W[::stride,::stride] / np.amax(depth) / 10, color='k')
        quiv = ax_crosssection.quiver(s_grid[::stride,::stride], z_grid[::stride,::stride], V[::stride,::stride] / (width*visual_norms), W[::stride,::stride] / (np.amax(depth)*visual_norms), color='k', alpha= physical_norms / np.amax(physical_norms))


        ax_crosssection.set_title(f'Lateral flow at t = {phase}' + r'$\sigma^{-1}$' f' s\nMaximum lateral velocity = {np.round(np.amax(physical_norms),5)}')


    def plot_cross_section_residual_forcing_mechanisms(self, p1: np.ndarray, p2: np.ndarray, num_horizontal_points, num_vertical_points, figsize=(12,6), cmap='RdBu', savename=None, component='u', **kwargs):
        """Plots all of the different forcing mechanisms for along-channel residual currents, along with the total forcing and the resulting residual flow."""

        width = np.linalg.norm(p1-p2, 2)
        s_range = np.linspace(width/2, -width/2, num_horizontal_points)
        x_range = np.linspace(p1[0], p2[0], num_horizontal_points)
        y_range = np.linspace(p1[1], p2[1], num_horizontal_points)
        sigma_range = np.linspace(-1, 0, num_vertical_points)

        H = self.hydro.spatial_physical_parameters['H'].cf
        epsilon = self.hydro.model_options['advection_epsilon']

        depth = evaluate_CF_range(H, self.hydro.mesh, x_range, y_range)

        s_grid = np.tile(s_range, (num_vertical_points, 1))
        z_grid = np.array([np.linspace(-depth[i], 0, num_vertical_points) for i in range(num_horizontal_points)]).T

        if component == 'u':
            along_advection = lambda sig: 0.5 * epsilon * (self.u(1, sig) * self.ux(1, sig) + self.u(-1, sig) * self.ux(-1, sig))
            lat_advection = lambda sig: 0.5 * epsilon * (self.v(1, sig) * self.uy(1, sig) + self.v(-1, sig) * self.uy(-1, sig))
            vert_advection = lambda sig: 0.5 / H * epsilon * (self.w(1, sig) * self.usig(1, sig) + self.w(-1, sig) * self.usig(-1, sig))
            pressure_gradient = lambda sig: self.hydro.constant_physical_parameters['g'] * self.gammax(0)
            coriolis = lambda sig: -self.hydro.constant_physical_parameters['f'] * self.v(0, sig)
            total_flow = lambda sig: self.u(0, sig)
        elif component == 'v':
            along_advection = lambda sig: 0.5 * epsilon * (self.u(1, sig) * self.vx(1, sig) + self.u(-1, sig) * self.vx(-1, sig))
            lat_advection = lambda sig: 0.5 * epsilon * (self.v(1, sig) * self.vy(1, sig) + self.v(-1, sig) * self.vy(-1, sig))
            vert_advection = lambda sig: 0.5 / H * epsilon * (self.w(1, sig) * self.vsig(1, sig) + self.w(-1, sig)*self.vsig(-1, sig))
            pressure_gradient = lambda sig: self.hydro.constant_physical_parameters['g'] * self.gammay(0)
            coriolis = lambda sig: self.hydro.constant_physical_parameters['f'] * self.u(0, sig)
            total_flow = lambda sig: self.v(0, sig)


        along_advection_eval = evaluate_vertical_structure_at_cross_section(self.hydro.mesh, along_advection, p1, p2, num_horizontal_points, num_vertical_points)
        lat_advection_eval = evaluate_vertical_structure_at_cross_section(self.hydro.mesh, lat_advection, p1, p2, num_horizontal_points, num_vertical_points)
        vert_advection_eval = evaluate_vertical_structure_at_cross_section(self.hydro.mesh, vert_advection, p1, p2, num_horizontal_points, num_vertical_points)
        pressure_gradient_eval = evaluate_vertical_structure_at_cross_section(self.hydro.mesh, pressure_gradient, p1, p2, num_horizontal_points, num_vertical_points)
        coriolis_eval = evaluate_vertical_structure_at_cross_section(self.hydro.mesh, coriolis, p1, p2, num_horizontal_points, num_vertical_points)
        # F = evaluate_vertical_structure_at_cross_section(self.hydro, total_forcing, p1, p2, num_horizontal_points, num_vertical_points)
        total_eval = along_advection_eval + lat_advection_eval + vert_advection_eval + pressure_gradient_eval + coriolis_eval
        flow_eval = evaluate_vertical_structure_at_cross_section(self.hydro.mesh, total_flow, p1, p2, num_horizontal_points, num_vertical_points)
        

        max_along_advection = np.amax(np.absolute(along_advection_eval))
        max_lat_advection = np.amax(np.absolute(lat_advection_eval))
        max_vert_advection = np.amax(np.absolute(vert_advection_eval))
        max_pressure_gradient = np.amax(np.absolute(pressure_gradient_eval))
        max_coriolis = np.amax(np.absolute(coriolis_eval))
        max_totalforcing = np.amax(np.absolute(total_eval))
        max_flow = np.amax(np.absolute(flow_eval))

        fig, ax = plt.subplots(3, 2, figsize=figsize)
        
        #   along_advection     lat_advection   
        #   vert_advection     pressure_gradient
        #   F       U
        
        along_advection_color = ax[0,0].pcolormesh(s_grid, z_grid, along_advection_eval, vmin=-max_along_advection, vmax=max_along_advection, cmap=cmap, **kwargs)
        lat_advection_color = ax[0,1].pcolormesh(s_grid, z_grid, lat_advection_eval, vmin=-max_lat_advection, vmax=max_lat_advection, cmap=cmap, **kwargs)
        vert_advection_color = ax[1,0].pcolormesh(s_grid, z_grid, vert_advection_eval, vmin=-max_vert_advection, vmax=max_vert_advection, cmap=cmap, **kwargs)
        pressure_gradient_color = ax[1,1].pcolormesh(s_grid, z_grid, pressure_gradient_eval, vmin=-max_pressure_gradient, vmax=max_pressure_gradient, cmap=cmap, **kwargs)
        F_color = ax[2,1].pcolormesh(s_grid, z_grid, total_eval, vmin=-max_totalforcing, vmax=max_totalforcing, cmap=cmap, **kwargs)
        # U_color = ax[2,1].pcolormesh(s_grid, z_grid, U, vmin=-maxU, vmax=maxU, cmap=cmap, **kwargs)
        coriolis_color = ax[2,0].pcolormesh(s_grid, z_grid, coriolis_eval, vmin=-max_coriolis, vmax=max_coriolis, cmap=cmap, **kwargs)

        along_advection_cbar = plt.colorbar(along_advection_color, ax=ax[0,0])
        lat_advection_cbar = plt.colorbar(lat_advection_color, ax=ax[0,1])
        vert_advection_cbar = plt.colorbar(vert_advection_color, ax=ax[1,0])
        pressure_gradient_cbar = plt.colorbar(pressure_gradient_color, ax=ax[1,1])
        F_cbar = plt.colorbar(F_color, ax=ax[2,1])
        coriolis_cbar = plt.colorbar(coriolis_color, ax=ax[2,0])

        ax[0,0].set_ylabel('-Depth [m]')
        ax[1,0].set_ylabel('-Depth [m]')
        ax[2,0].set_ylabel('-Depth [m]')
        ax[2,0].set_xlabel('y [m]')
        ax[2,1].set_xlabel('y [m]')

        ax[0,0].plot(s_range, -depth, linewidth=1, color='k', zorder=3)
        ax[1,0].plot(s_range, -depth, linewidth=1, color='k', zorder=3)
        ax[2,0].plot(s_range, -depth, linewidth=1, color='k', zorder=3)
        ax[2,0].plot(s_range, -depth, linewidth=1, color='k', zorder=3)
        ax[2,1].plot(s_range, -depth, linewidth=1, color='k', zorder=3)
        ax[1,1].plot(s_range, -depth, linewidth=1, color='k', zorder=3)
        ax[0,1].plot(s_range, -depth, linewidth=1, color='k', zorder=3)

        ax[0,0].fill_between(s_range, -np.amax(depth), -depth, color='silver')
        ax[1,0].fill_between(s_range, -np.amax(depth), -depth, color='silver')
        ax[2,0].fill_between(s_range, -np.amax(depth), -depth, color='silver')
        ax[2,0].fill_between(s_range, -np.amax(depth), -depth, color='silver')
        ax[2,1].fill_between(s_range, -np.amax(depth), -depth, color='silver')
        ax[1,1].fill_between(s_range, -np.amax(depth), -depth, color='silver')
        ax[0,1].fill_between(s_range, -np.amax(depth), -depth, color='silver')

        if component == 'u':
            ax[0,0].set_title(r'$\overline{uu_x}$')
            ax[0,1].set_title(r'$\overline{vu_y}$')
            ax[1,0].set_title(r'$\overline{wu_z}$')
            ax[1,1].set_title(r'$g\overline{\zeta_x}$')
            ax[2,0].set_title(r'$-f\overline{v}$')
            ax[2,1].set_title('All forcing')
        elif component == 'v':
            ax[0,0].set_title(r'$\overline{uv_x}$')
            ax[0,1].set_title(r'$\overline{vv_y}$')
            ax[1,0].set_title(r'$\overline{wv_z}$')
            ax[1,1].set_title(r'$g\overline{\zeta_y}$')
            ax[2,0].set_title(r'$f\overline{u}$')
            ax[2,1].set_title('All forcing')

        plt.suptitle(f'Residual forcing mechanisms for {component}')
        plt.tight_layout()

        if savename is not None:
            fig.savefig(savename)
    







    def animate_cross_section(self, p1, p2, num_horizontal_points, num_vertical_points, stride, num_frames, constituent='all', mode='savefigs', basename=None, variable='u'):
        
        # Initialize coordinate grids
        
        phase = np.linspace(0, 1, num_frames, endpoint=True)

        width = np.linalg.norm(p1-p2, 2)
        s_range = np.linspace(-width/2, width/2, num_horizontal_points)
        x_range = np.linspace(p1[0], p2[0], num_horizontal_points)
        y_range = np.linspace(p1[1], p2[1], num_horizontal_points)

        H = self.hydro.spatial_physical_parameters['H'].cf

        depth = evaluate_CF_range(H, self.hydro.mesh, x_range, y_range)

        s_grid = np.tile(s_range, (num_vertical_points, 1))
        z_grid = np.array([np.linspace(-depth[i], 0, num_vertical_points) for i in range(num_horizontal_points)]).T

        # Get flow variables

        if constituent == 'all':
            Q = np.zeros((num_frames, s_grid.shape[0], s_grid.shape[1]))
            V = np.zeros((num_frames, s_grid.shape[0], s_grid.shape[1]))
            W = np.zeros((num_frames, s_grid.shape[0], s_grid.shape[1]))
            for i in range(num_frames):
                Q[i,:,:] = evaluate_vertical_structure_at_cross_section(self.hydro.mesh, lambda sig: self.u_timed(phase[i] / self.hydro.constant_physical_parameters['sigma'], sig), p1, p2, num_horizontal_points, num_vertical_points)
                V[i,:,:] = evaluate_vertical_structure_at_cross_section(self.hydro.mesh, lambda sig: self.v_timed(phase[i]/ self.hydro.constant_physical_parameters['sigma'], sig), p1, p2, num_horizontal_points, num_vertical_points)
                W[i,:,:] = evaluate_vertical_structure_at_cross_section(self.hydro.mesh, lambda sig: self.w_timed(phase[i]/ self.hydro.constant_physical_parameters['sigma'], sig), p1, p2, num_horizontal_points, num_vertical_points)
        elif constituent == 0:
            Q = np.zeros((num_frames, s_grid.shape[0], s_grid.shape[1]))
            V = np.zeros((num_frames, s_grid.shape[0], s_grid.shape[1]))
            W = np.zeros((num_frames, s_grid.shape[0], s_grid.shape[1]))
            for i in range(num_frames):
                Q[i,:,:] = evaluate_vertical_structure_at_cross_section(self.hydro.mesh, lambda sig: self.u(constituent, sig) * self.hydro.time_basis.evaluation_function(phase[i] / self.hydro.constant_physical_parameters['sigma'], constituent), p1, p2,
                                                                num_horizontal_points, num_vertical_points)
                V[i,:,:] = evaluate_vertical_structure_at_cross_section(self.hydro.mesh, lambda sig: self.v(constituent, sig) * self.hydro.time_basis.evaluation_function(phase[i] / self.hydro.constant_physical_parameters['sigma'], constituent), p1, p2,
                                                                num_horizontal_points, num_vertical_points)
                W[i,:,:] = evaluate_vertical_structure_at_cross_section(self.hydro.mesh, lambda sig: self.w(constituent, sig) * self.hydro.time_basis.evaluation_function(phase[i] / self.hydro.constant_physical_parameters['sigma'], constituent), p1, p2,
                                                                num_horizontal_points, num_vertical_points)
        else:
            Q = np.zeros((num_frames, s_grid.shape[0], s_grid.shape[1]))
            V = np.zeros((num_frames, s_grid.shape[0], s_grid.shape[1]))
            W = np.zeros((num_frames, s_grid.shape[0], s_grid.shape[1]))
            for i in range(num_frames):
                Q[i,:,:] = evaluate_vertical_structure_at_cross_section(self.hydro.mesh, lambda sig: self.u(constituent, sig) * self.hydro.time_basis.evaluation_function(phase[i] / self.hydro.constant_physical_parameters['sigma'], constituent) + \
                                                                self.u(-constituent, sig) * self.hydro.time_basis.evaluation_function(phase[i]/self.hydro.constant_physical_parameters['sigma'], -constituent), p1, p2,
                                                                num_horizontal_points, num_vertical_points)
                V[i,:,:] = evaluate_vertical_structure_at_cross_section(self.hydro.mesh, lambda sig: self.v(constituent, sig) * self.hydro.time_basis.evaluation_function(phase[i] / self.hydro.constant_physical_parameters['sigma'], constituent) + \
                                                                self.v(-constituent, sig) * self.hydro.time_basis.evaluation_function(phase[i]/self.hydro.constant_physical_parameters['sigma'], -constituent), p1, p2,
                                                                num_horizontal_points, num_vertical_points)
                W[i,:,:] = evaluate_vertical_structure_at_cross_section(self.hydro.mesh, lambda sig: self.w(constituent, sig) * self.hydro.time_basis.evaluation_function(phase[i] / self.hydro.constant_physical_parameters['sigma'], constituent) + \
                                                                self.w(-constituent, sig) * self.hydro.time_basis.evaluation_function(phase[i]/self.hydro.constant_physical_parameters['sigma'], -constituent), p1, p2,
                                                                num_horizontal_points, num_vertical_points)
                
        # Get maxima

        maxlongi = max(np.amax(Q), np.amax(-Q))
        maxlati = np.amax(np.absolute(V))

        if variable == 'circulation':

            visual_norms = np.sqrt((V[:,::stride,::stride] / width)**2 + (W[:,::stride,::stride] / np.amax(depth))**2) # norm of the vector that we plot
            physical_norms = np.sqrt((V[:,::stride,::stride])**2 + (W[:,::stride,::stride])**2)

            maxlati = np.amax(physical_norms)

            if mode == 'savefigs':
                for i in range(num_frames):
                    fig_crosssection, ax_crosssection = plt.subplots()
            
                    color_crosssection = ax_crosssection.pcolormesh(s_grid, z_grid, Q[i,:,:], vmin=-maxlongi, vmax=maxlongi, cmap='bwr')
                    cbar_crosssection = plt.colorbar(color_crosssection)
                    cbar_crosssection.ax.set_ylabel('Longitudinal velocity [m/s]')

                    ax_crosssection.plot(s_range, -depth, linewidth=1, color='k', zorder=3)
                    ax_crosssection.fill_between(s_range, -np.amax(depth), -depth, color='silver')

                    quiv = ax_crosssection.quiver(s_grid[::stride,::stride], z_grid[::stride,::stride], V[i, ::stride,::stride] / (width*visual_norms[i,:,:]), W[i, ::stride,::stride] / (np.amax(depth)*visual_norms[i,:,:]), color='k', alpha= physical_norms[i,:,:] / np.amax(physical_norms))

                    phasestring = str(np.round(phase[i], 5))
                    phasestring += '0' * (7-len(phasestring))

                    ax_crosssection.set_title(f'Lateral flow at t = {phasestring}' + r'$\sigma^{-1}$' f' s\nMaximum lateral velocity = {np.round(np.amax(physical_norms),5)}')

                    fig_crosssection.savefig(f'{basename}_phase{i}.png')

        elif variable == 'u':

            if mode == 'savefigs':
                for i in range(num_frames):
                    fig_crosssection, ax_crosssection = plt.subplots()
            
                    color_crosssection = ax_crosssection.pcolormesh(s_grid, z_grid, Q[i,:,:], vmin=-maxlongi, vmax=maxlongi, cmap='bwr')
                    cbar_crosssection = plt.colorbar(color_crosssection)
                    cbar_crosssection.ax.set_ylabel('Along-channel velocity [m/s]')

                    ax_crosssection.plot(s_range, -depth, linewidth=1, color='k', zorder=3)
                    ax_crosssection.fill_between(s_range, -np.amax(depth), -depth, color='silver')

                    # quiv = ax_crosssection.quiver(s_grid[::stride,::stride], z_grid[::stride,::stride], V[i, ::stride,::stride] / (width*visual_norms[i,:,:]), W[i, ::stride,::stride] / (np.amax(depth)*visual_norms[i,:,:]), color='k', alpha= physical_norms[i,:,:] / np.amax(physical_norms))

                    phasestring = str(np.round(phase[i], 5))
                    phasestring += '0' * (7-len(phasestring))

                    ax_crosssection.set_title(f'Along-channel velocity at t = {phasestring}' + r'$\sigma^{-1}$')

                    fig_crosssection.savefig(f'{basename}_phase{i}.png')
        
        elif variable == 'v':

            if mode == 'savefigs':
                for i in range(num_frames):
                    fig_crosssection, ax_crosssection = plt.subplots()
            
                    color_crosssection = ax_crosssection.pcolormesh(s_grid, z_grid, V[i,:,:], vmin=-maxlati, vmax=maxlati, cmap='bwr')
                    cbar_crosssection = plt.colorbar(color_crosssection)
                    cbar_crosssection.ax.set_ylabel('Cross-channel velocity [m/s]')

                    ax_crosssection.plot(s_range, -depth, linewidth=1, color='k', zorder=3)
                    ax_crosssection.fill_between(s_range, -np.amax(depth), -depth, color='silver')

                    # quiv = ax_crosssection.quiver(s_grid[::stride,::stride], z_grid[::stride,::stride], V[i, ::stride,::stride] / (width*visual_norms[i,:,:]), W[i, ::stride,::stride] / (np.amax(depth)*visual_norms[i,:,:]), color='k', alpha= physical_norms[i,:,:] / np.amax(physical_norms))

                    phasestring = str(np.round(phase[i], 5))
                    phasestring += '0' * (7-len(phasestring))

                    ax_crosssection.set_title(f'Cross-channel velocity at t = {phasestring}' + r'$\sigma^{-1}$')

                    fig_crosssection.savefig(f'{basename}_phase{i}.png')


    def plot_cross_section_contours(self, quantity_function, quantity_string, num_levels, p1, p2, num_horizontal_points, num_vertical_points, subamplitude_lines=2,**kwargs):
        width = np.linalg.norm(p1-p2, 2)
        s_range = np.linspace(-width/2, width/2, num_horizontal_points)
        x_range = np.linspace(p1[0], p2[0], num_horizontal_points)
        y_range = np.linspace(p1[1], p2[1], num_horizontal_points)
        sigma_range = np.linspace(-1, 0, num_vertical_points)

        H = self.hydro.spatial_physical_parameters['H'].cf

        depth = evaluate_CF_range(H, self.hydro.mesh, x_range, y_range)

        s_grid = np.tile(s_range, (num_vertical_points, 1))
        z_grid = np.array([np.linspace(-depth[i], 0, num_vertical_points) for i in range(num_horizontal_points)]).T

        Q = evaluate_vertical_structure_at_cross_section(self.hydro.mesh, quantity_function, p1, p2, num_horizontal_points, num_vertical_points)

        levels = np.linspace(np.min(Q), np.max(Q), num_levels*(subamplitude_lines+1))

        fig_crosscontour, ax_crosscontour = plt.subplots()
        contourf = ax_crosscontour.contourf(s_grid, z_grid, Q, levels, **kwargs)
        contour = ax_crosscontour.contour(s_grid, z_grid, Q, levels, colors=['k'] + ["0.4"] * subamplitude_lines, linewidths=[0.7]+[0.1]*subamplitude_lines)

        ax_crosscontour.clabel(contour, levels[0::subamplitude_lines+1], inline=1, fontsize=10, fmt='%1.4f')
        ax_crosscontour.set_title(f'Cross section {quantity_string} from ({p1[0], p1[1]}) to ({p2[0], p2[1]})')

        ax_crosscontour.plot(s_range, -depth, linewidth=1, color='k', zorder=3)
        ax_crosscontour.fill_between(s_range, -np.amax(depth), -depth, color='silver')

        cbar = fig_crosscontour.colorbar(contourf)




    

        