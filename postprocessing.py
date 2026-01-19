import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import ngsolve
import os

from NiFlow.hydrodynamics import Hydrodynamics
from NiFlow.utils import *
from NiFlow.solve import solve


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
    - quantity_function: function (can be lambda-function) that takes sigma as argument and returns the horizontal solution field of that variable as an ngsolve.GridFunction or ngsolve.
    - p1: first point to span the cross-section.
    - p2: second point to span the cross-section.
    - num_horizontal_points: number of equally spaced horizontal locations at which the variable is evaluated.
    - num_vertical_points: number of equally spaced sigma layers at which the variable is evaluated.    
    
    """
    sigma_range = np.linspace(-1, 0, num_vertical_points)
    x_range = np.linspace(p1[0], p2[0], num_horizontal_points)
    y_range = np.linspace(p1[1], p2[1], num_horizontal_points)

    Q = np.zeros((num_vertical_points, num_horizontal_points))
    
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
        # if hydro.model_options['sea_boundary_treatment'] == 'exact':
        #     sea_interpolant = ((hydro.geometric_information['riverine_boundary_x'] / x_scaling) + (hydro.geometric_information['L_BL_river']/x_scaling) + (hydro.geometric_information['L_R_river']/x_scaling) + \
        #                     (hydro.geometric_information['L_RA_river']/x_scaling) - ngsolve.x) / \
        #                     ((hydro.geometric_information['riverine_boundary_x']+hydro.geometric_information['L_BL_river']+hydro.geometric_information['L_R_river']+hydro.geometric_information['L_RA_river'] +
        #                         hydro.geometric_information['L_BL_sea'] + hydro.geometric_information['L_R_sea'] + hydro.geometric_information['L_RA_sea']) / x_scaling)
        #     sea_interpolant_x = -1 / ((hydro.geometric_information['riverine_boundary_x']+hydro.geometric_information['L_BL_river']+hydro.geometric_information['L_R_river']+hydro.geometric_information['L_RA_river'] +
        #                         hydro.geometric_information['L_BL_sea'] + hydro.geometric_information['L_R_sea'] + hydro.geometric_information['L_RA_sea']) / x_scaling)
        #     self.sea_interpolant = sea_interpolant
        #     self.sea_interpolant_x = sea_interpolant_x
        if hydro.model_options['river_boundary_treatment'] == 'exact':
            river_interpolant = (-(hydro.geometric_information['L_BL_sea']/x_scaling) - (hydro.geometric_information['L_R_sea']/x_scaling) - \
                                (hydro.geometric_information['L_RA_sea']/x_scaling) + ngsolve.x) / \
                                ((hydro.geometric_information['riverine_boundary_x']+hydro.geometric_information['L_BL_river']+hydro.geometric_information['L_R_river']+hydro.geometric_information['L_RA_river'] +
                                hydro.geometric_information['L_BL_sea'] + hydro.geometric_information['L_R_sea'] + hydro.geometric_information['L_RA_sea']) / x_scaling)
            river_interpolant_x = 1 / ((hydro.geometric_information['riverine_boundary_x']+hydro.geometric_information['L_BL_river']+hydro.geometric_information['L_R_river']+hydro.geometric_information['L_RA_river'] +
                                hydro.geometric_information['L_BL_sea'] + hydro.geometric_information['L_R_sea'] + hydro.geometric_information['L_RA_sea']) / x_scaling)
            self.river_interpolant = river_interpolant
            self.river_interpolant_x = river_interpolant_x
        

        if hydro.model_options['river_boundary_treatment'] != 'exact':
            self.u = lambda q, sigma : sum([hydro.alpha_solution[m][q] * hydro.vertical_basis.evaluation_function(sigma, m) for m in range(M)])
        else:
            self.u = lambda q, sigma : sum([(hydro.alpha_solution[m][q]+ hydro.Q_solution[q] * hydro.riverine_forcing.normal_alpha[m] * river_interpolant) * hydro.vertical_basis.evaluation_function(sigma, m) for m in range(M)])
            self.eval_Q = lambda q : evaluate_CF_point(hydro.Q_solution[q], hydro.mesh, 0, 0) 
        self.v = lambda q, sigma : sum([hydro.beta_solution[m][q] * hydro.vertical_basis.evaluation_function(sigma, m) for m in range(M)])

        self.gamma = lambda q : hydro.gamma_solution[q]

        self.gamma_abs = lambda q: ngsolve.sqrt(self.hydro.gamma_solution[q]*self.hydro.gamma_solution[q]) if q == 0 else ngsolve.sqrt(self.hydro.gamma_solution[q]*self.hydro.gamma_solution[q]+self.hydro.gamma_solution[-q]*self.hydro.gamma_solution[-q])
        self.zeta_timed = lambda t: sum([self.hydro.gamma_solution[l] * self.hydro.time_basis.evaluation_function(t, l) for l in range(-imax, imax + 1)])
        self.zetat_timed = lambda t: sum([self.hydro.gamma_solution[l] * self.hydro.time_basis.derivative_evaluation_function(t, l) for l in range(-imax, imax + 1)])

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

        
        # H = hydro.spatial_parameters['H']
        # Hx = hydro.spatial_parameters_grad['H'][0]
        # Hy = hydro.spatial_parameters_grad['H'][1]


        ## Construct vertical velocity ######
        # Determine whether the free surface was included in the sigma-coordinates

        if np.any(hydro.model_options['surface_interaction_influence_matrix']) and hydro.constant_physical_parameters['surface_epsilon'] > 0:
            self.surface_in_sigma = True
        else:
            self.surface_in_sigma = False
        

        self._construct_w_timed()

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
        # if hydro.model_options['sea_boundary_treatment'] != 'exact':
        self.gammax = lambda q: ngsolve.grad(hydro.gamma_solution[q])[0] / x_scaling
        # else:
        #     self.gammax = lambda q: (ngsolve.grad(hydro.gamma_solution[q])[0] + hydro.A_solution[q] * sea_interpolant_x) / x_scaling
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


        self._construct_transport()
        self._construct_TWA_velocities()


    def _construct_transport(self):
        """Constructs depth-integrated velocity as a function of (x,y,t) as well as the time-averaged variant of that. Assumes that surface interactions have been
        taken into account."""

        M = self.hydro.numerical_information['M']
        imax = self.hydro.numerical_information['imax']

        G4 = self.hydro.vertical_basis.tensor_dict['G4']
        H3 = self.hydro.time_basis.tensor_dict['H3'] 

        H = self.hydro.spatial_parameters['H']
        alpha = self.hydro.alpha_solution
        beta = self.hydro.beta_solution
        gamma = self.hydro.gamma_solution

        # if self.hydro.model_options['sea_boundary_treatment'] == 'exact':
        #     A = self.hydro.A_solution
        if self.hydro.model_options['river_boundary_treatment'] == 'exact':
            Q = self.hydro.Q_solution
            normalalpha = self.hydro.riverine_forcing.normal_alpha

        # if self.surface_in_sigma:
        # if self.hydro.model_options['sea_boundary_treatment'] == 'exact' and self.hydro.model_options['river_boundary_treatment'] == 'exact':
        #     self.transport_vector = []

        #     self.transport_vector.append(
        #         lambda q: sum([0.5 * H * (alpha[m][q]+Q[q]*normalalpha[m]*self.river_interpolant) * G4(m) + \
        #                     sum([sum([(gamma[i]+A[i]*self.sea_interpolant) * (alpha[m][j]+Q[j]*normalalpha[m]*self.river_interpolant) * H3(i,j,q) * G4(m)
        #                     for i in range(-imax, imax + 1)]) for j in range(-imax, imax + 1)]) for m in range(M)])
        #     )
        #     self.transport_vector.append(
        #         lambda q: sum([0.5 * H * beta[m][q] * G4(m) + \
        #                     sum([sum([(gamma[i]+A[i]*self.sea_interpolant) * beta[m][j] * H3(i,j,q) * G4(m)
        #                     for i in range(-imax, imax + 1)]) for j in range(-imax, imax + 1)]) for m in range(M)])
        #     )

        #     self.TA_transport = []
        #     self.TA_transport.append(
        #         0.5 * np.sqrt(2) * H * sum([(alpha[m][0]+Q[0]*normalalpha[m]*self.river_interpolant) * G4(m) for m in range(M)]) + \
        #         sum([sum([0.5 * (gamma[q]+A[q]*self.sea_interpolant) * (alpha[m][q]+Q[q]*normalalpha[m]*self.river_interpolant) * G4(m) for q in range(-imax, imax+1)]) for m in range(M)])
        #     )
        #     self.TA_transport.append(
        #         0.5 * np.sqrt(2) * H * sum([beta[m][0] * G4(m) for m in range(M)]) + \
        #         sum([sum([0.5 * (gamma[q]+A[q]*self.sea_interpolant) * beta[m][q] * G4(m) for q in range(-imax, imax+1)]) for m in range(M)])
        #     )

        #     self.stokes_drift = [
        #         self.TA_transport[0] - 0.5 * np.sqrt(2) * H * sum([(alpha[m][0]+Q[0]*normalalpha[m]*self.river_interpolant) * G4(m) for m in range(M)]),
        #         self.TA_transport[1] - 0.5 * np.sqrt(2) * H * sum([beta[m][0] * G4(m) for m in range(M)])
        #     ]

        #     self.timed_transport = []
        #     self.timed_transport.append(
        #         lambda t: sum([sum([H * (alpha[m][i]+Q[i]*normalalpha[m]*self.river_interpolant) * G4(m) * self.hydro.time_basis.evaluation_function(t, i) for i in range(-imax, imax + 1)]) for m in range(M)]) + \
        #         sum([sum([sum([(gamma[i]+A[i]*self.sea_interpolant) * (alpha[m][j] + Q[j]*normalalpha[m]*self.river_interpolant) * G4(m) * self.hydro.time_basis.evaluation_function(t, i) * self.hydro.time_basis.evaluation_function(t, j) for i in range(-imax, imax + 1)]) for j in range(-imax, imax + 1)]) for m in range(M)])
        #     )
        #     self.timed_transport.append(
        #         lambda t: sum([sum([H * beta[m][i] * G4(m) * self.hydro.time_basis.evaluation_function(t, i) for i in range(-imax, imax + 1)]) for m in range(M)]) + \
        #         sum([sum([sum([(gamma[i]+A[i]*self.sea_interpolant) * beta[m][j] * G4(m) * self.hydro.time_basis.evaluation_function(t, i) * self.hydro.time_basis.evaluation_function(t, j) for i in range(-imax, imax + 1)]) for j in range(-imax, imax + 1)]) for m in range(M)])
        #     )

        if self.hydro.model_options['river_boundary_treatment'] == 'exact':
            self.transport_vector = []

            self.transport_vector.append(
                lambda q: sum([0.5 * H * (alpha[m][q]+Q[q]*normalalpha[m]*self.river_interpolant) * G4(m) + \
                            sum([sum([gamma[i] * alpha[m][j] * H3(i,j,q) * G4(m)
                            for i in range(-imax, imax + 1)]) for j in range(-imax, imax + 1)]) for m in range(M)])
            )
            self.transport_vector.append(
                lambda q: sum([0.5 * H * beta[m][q] * G4(m) + \
                            sum([sum([gamma[i] * beta[m][j] * H3(i,j,q) * G4(m)
                            for i in range(-imax, imax + 1)]) for j in range(-imax, imax + 1)]) for m in range(M)])
            )

            self.TA_transport = []
            self.TA_transport.append(
                0.5 * np.sqrt(2) * H * sum([(alpha[m][0]+Q[0]*normalalpha[m]*self.river_interpolant) * G4(m) for m in range(M)]) + \
                sum([sum([0.5 * gamma[q] * alpha[m][q] * G4(m) for q in range(-imax, imax+1)]) for m in range(M)])
            )
            self.TA_transport[-1] += sum(0.5 * gamma[0] * Q[0]*normalalpha[m]*self.river_interpolant*G4(m) for m in range(M))
            self.TA_transport.append(
                0.5 * np.sqrt(2) * H * sum([beta[m][0] * G4(m) for m in range(M)]) + \
                sum([sum([0.5 * gamma[q] * beta[m][q] * G4(m) for q in range(-imax, imax+1)]) for m in range(M)])
            )

            self.stokes_drift = [
                self.TA_transport[0] - 0.5 * np.sqrt(2) * H * sum([(alpha[m][0]+Q[0]*normalalpha[m]*self.river_interpolant) * G4(m) for m in range(M)]),
                self.TA_transport[1] - 0.5 * np.sqrt(2) * H * sum([beta[m][0] * G4(m) for m in range(M)])
            ]

            self.timed_transport = []
            self.timed_transport.append(
                lambda t: sum([sum([H * (alpha[m][i]+Q[i]*normalalpha[m]*self.river_interpolant) * G4(m) * self.hydro.time_basis.evaluation_function(t, i) for i in range(-imax, imax + 1)]) for m in range(M)]) + \
                sum([sum([sum([gamma[i] * (alpha[m][j]+Q[j]*normalalpha[m]*self.river_interpolant) * G4(m) * self.hydro.time_basis.evaluation_function(t, i) * self.hydro.time_basis.evaluation_function(t, j) for i in range(-imax, imax + 1)]) for j in range(-imax, imax + 1)]) for m in range(M)])
            )
            self.timed_transport.append(
                lambda t: sum([sum([H * beta[m][i] * G4(m) * self.hydro.time_basis.evaluation_function(t, i) for i in range(-imax, imax + 1)]) for m in range(M)]) + \
                sum([sum([sum([gamma[i] * beta[m][j] * G4(m) * self.hydro.time_basis.evaluation_function(t, i) * self.hydro.time_basis.evaluation_function(t, j) for i in range(-imax, imax + 1)]) for j in range(-imax, imax + 1)]) for m in range(M)])
            )
        else:
            self.transport_vector = []

            self.transport_vector.append(
                lambda q: sum([0.5 * H * alpha[m][q] * G4(m) + \
                            sum([sum([(gamma[i]) * alpha[m][j] * H3(i,j,q) * G4(m)
                            for i in range(-imax, imax + 1)]) for j in range(-imax, imax + 1)]) for m in range(M)])
            )
            self.transport_vector.append(
                lambda q: sum([0.5 * H * beta[m][q] * G4(m) + \
                            sum([sum([(gamma[i]) * beta[m][j] * H3(i,j,q) * G4(m)
                            for i in range(-imax, imax + 1)]) for j in range(-imax, imax + 1)]) for m in range(M)])
            )

            self.TA_transport = []
            self.TA_transport.append(
                0.5 * np.sqrt(2) * H * sum([alpha[m][0] * G4(m) for m in range(M)]) + \
                sum([sum([0.5 * (gamma[q]) * alpha[m][q] * G4(m) for q in range(-imax, imax+1)]) for m in range(M)])
            )
            self.TA_transport.append(
                0.5 * np.sqrt(2) * H * sum([beta[m][0] * G4(m) for m in range(M)]) + \
                sum([sum([0.5 * (gamma[q]) * beta[m][q] * G4(m) for q in range(-imax, imax+1)]) for m in range(M)])
            )

            self.stokes_drift = [
                self.TA_transport[0] - 0.5 * np.sqrt(2) * H * sum([alpha[m][0] * G4(m) for m in range(M)]),
                self.TA_transport[1] - 0.5 * np.sqrt(2) * H * sum([beta[m][0] * G4(m) for m in range(M)])
            ]

            self.timed_transport = []
            self.timed_transport.append(
                lambda t: sum([sum([H * alpha[m][i] * G4(m) * self.hydro.time_basis.evaluation_function(t, i) for i in range(-imax, imax + 1)]) for m in range(M)]) + \
                sum([sum([sum([(gamma[i]) * alpha[m][j] * G4(m) * self.hydro.time_basis.evaluation_function(t, i) * self.hydro.time_basis.evaluation_function(t, j) for i in range(-imax, imax + 1)]) for j in range(-imax, imax + 1)]) for m in range(M)])
            )
            self.timed_transport.append(
                lambda t: sum([sum([H * beta[m][i] * G4(m) * self.hydro.time_basis.evaluation_function(t, i) for i in range(-imax, imax + 1)]) for m in range(M)]) + \
                sum([sum([sum([(gamma[i]) * beta[m][j] * G4(m) * self.hydro.time_basis.evaluation_function(t, i) * self.hydro.time_basis.evaluation_function(t, j) for i in range(-imax, imax + 1)]) for j in range(-imax, imax + 1)]) for m in range(M)])
            )
        #     else:
        #         self.transport_vector = []

        #         self.transport_vector.append(
        #             lambda q: sum([0.5 * H * alpha[m][q] * G4(m) + \
        #                         sum([sum([gamma[i] * alpha[m][j] * H3(i,j,q) * G4(m)
        #                         for i in range(-imax, imax + 1)]) for j in range(-imax, imax + 1)]) for m in range(M)])
        #         )
        #         self.transport_vector.append(
        #             lambda q: sum([0.5 * H * beta[m][q] * G4(m) + \
        #                         sum([sum([gamma[i] * beta[m][j] * H3(i,j,q) * G4(m)
        #                         for i in range(-imax, imax + 1)]) for j in range(-imax, imax + 1)]) for m in range(M)])
        #         )

        #         self.TA_transport = []
        #         self.TA_transport.append(
        #             0.5 * np.sqrt(2) * H * sum([alpha[m][0] * G4(m) for m in range(M)]) + \
        #             sum([sum([0.5 * gamma[q] * alpha[m][q] * G4(m) for q in range(-imax, imax+1)]) for m in range(M)])
        #         )
        #         self.TA_transport.append(
        #             0.5 * np.sqrt(2) * H * sum([beta[m][0] * G4(m) for m in range(M)]) + \
        #             sum([sum([0.5 * gamma[q] * beta[m][q] * G4(m) for q in range(-imax, imax+1)]) for m in range(M)])
        #         )

        #         self.stokes_drift = [
        #             self.TA_transport[0] - 0.5 * np.sqrt(2) * H * sum([alpha[m][0] * G4(m) for m in range(M)]),
        #             self.TA_transport[1] - 0.5 * np.sqrt(2) * H * sum([beta[m][0] * G4(m) for m in range(M)])
        #         ]

        #         self.timed_transport = []
        #         self.timed_transport.append(
        #             lambda t: sum([sum([H * alpha[m][i] * G4(m) * self.hydro.time_basis.evaluation_function(t, i) for i in range(-imax, imax + 1)]) for m in range(M)]) + \
        #             sum([sum([sum([gamma[i] * alpha[m][j] * G4(m) * self.hydro.time_basis.evaluation_function(t, i) * self.hydro.time_basis.evaluation_function(t, j) for i in range(-imax, imax + 1)]) for j in range(-imax, imax + 1)]) for m in range(M)])
        #         )
        #         self.timed_transport.append(
        #             lambda t: sum([sum([H * beta[m][i] * G4(m) * self.hydro.time_basis.evaluation_function(t, i) for i in range(-imax, imax + 1)]) for m in range(M)]) + \
        #             sum([sum([sum([gamma[i] * beta[m][j] * G4(m) * self.hydro.time_basis.evaluation_function(t, i) * self.hydro.time_basis.evaluation_function(t, j) for i in range(-imax, imax + 1)]) for j in range(-imax, imax + 1)]) for m in range(M)])
        #         )
        # else:
        #     if self.hydro.model_options['river_boundary_treatment'] == 'exact':
        #         self.transport_vector = []

        #         self.transport_vector.append(
        #             lambda q: sum([0.5 * H * (alpha[m][q]+Q[q]*normalalpha[m]*self.river_interpolant) * G4(m) for m in range(M)])
        #         )
        #         self.transport_vector.append(
        #             lambda q: sum([0.5 * H * beta[m][q] * G4(m) for m in range(M)])
        #         )

        #         self.TA_transport = []
        #         self.TA_transport.append(
        #             0.5 * np.sqrt(2) * H * sum([(alpha[m][0]+Q[0]*normalalpha[m]*self.river_interpolant) * G4(m) for m in range(M)])
        #         )
        #         self.TA_transport.append(
        #             0.5 * np.sqrt(2) * H * sum([beta[m][0] * G4(m) for m in range(M)])
        #         )

        #         self.timed_transport = []
        #         self.timed_transport.append(
        #             lambda t: sum([sum([H * (alpha[m][i]+Q[i]*normalalpha[m]*self.river_interpolant) * G4(m) * self.hydro.time_basis.evaluation_function(t, i) for i in range(-imax, imax + 1)]) for m in range(M)])
        #         )
        #         self.timed_transport.append(
        #             lambda t: sum([sum([H * beta[m][i] * G4(m) * self.hydro.time_basis.evaluation_function(t, i) for i in range(-imax, imax + 1)]) for m in range(M)])
        #         )
        #     else:
        #         self.transport_vector = []

        #         self.transport_vector.append(
        #             lambda q: sum([0.5 * H * alpha[m][q] * G4(m) for m in range(M)])
        #         )
        #         self.transport_vector.append(
        #             lambda q: sum([0.5 * H * beta[m][q] * G4(m) for m in range(M)])
        #         )

        #         self.TA_transport = []
        #         self.TA_transport.append(
        #             0.5 * np.sqrt(2) * H * sum([alpha[m][0] * G4(m) for m in range(M)])
        #         )
        #         self.TA_transport.append(
        #             0.5 * np.sqrt(2) * H * sum([beta[m][0] * G4(m) for m in range(M)])
        #         )

        #         self.timed_transport = []
        #         self.timed_transport.append(
        #             lambda t: sum([sum([H * alpha[m][i] * G4(m) * self.hydro.time_basis.evaluation_function(t, i) for i in range(-imax, imax + 1)]) for m in range(M)])
        #         )
        #         self.timed_transport.append(
        #             lambda t: sum([sum([H * beta[m][i] * G4(m) * self.hydro.time_basis.evaluation_function(t, i) for i in range(-imax, imax + 1)]) for m in range(M)])
        #         )

        
    def _construct_w_timed(self):
        H = self.hydro.spatial_parameters['H']
        Hx = self.hydro.spatial_parameters_grad['H'][0]
        Hy = self.hydro.spatial_parameters_grad['H'][1]

        L = self.hydro.geometric_information['x_scaling']
        B = self.hydro.geometric_information['y_scaling']

        M = self.hydro.numerical_information['M']
        imax = self.hydro.numerical_information['imax']

        h = self.hydro.time_basis.evaluation_function
        f = self.hydro.vertical_basis.evaluation_function
        F = self.hydro.vertical_basis.integrated_evaluation_function

        sig_h_f_term = lambda t, sig: sum(sum(
            (Hx * self.hydro.alpha_solution[m][q] / L + Hy * self.hydro.beta_solution[m][q] / B) * sig * f(sig, m) * h(t, q)
            for q in range(-imax, imax + 1)) for m in range(M))
        
        onepsig_hh_f_term = lambda t, sig: sum(sum(sum(
            (self.hydro.gamma_grad[i][0] * self.hydro.alpha_solution[m][j] / L + self.hydro.gamma_grad[i][1] * self.hydro.beta_solution[m][j] / B) *
            (1 + sig) * h(t, i) * h(t, j) * f(sig, m)
        for i in range(-imax, imax + 1)) for j in range(-imax, imax + 1)) for m in range(M))

        h_F_term = lambda t, sig: sum(sum(
            (H * self.hydro.alpha_grad[m][i][0] / L + Hx * self.hydro.alpha_solution[m][i] / L + 
            H * self.hydro.beta_grad[m][i][1] / B + Hy * self.hydro.beta_solution[m][i] / B) * h(t, i) * F(sig, m)
        for i in range(-imax, imax + 1)) for m in range(M))

        hh_F_term = lambda t, sig: sum(sum(sum(
            (self.hydro.gamma_solution[i] * self.hydro.alpha_grad[m][j][0] / L + self.hydro.gamma_grad[i][0] * self.hydro.alpha_solution[m][j] / L +
             self.hydro.gamma_solution[i] * self.hydro.beta_grad[m][j][1] / B + self.hydro.gamma_grad[i][1] * self.hydro.beta_solution[m][j] / B) *
             h(t, i) * h(t, j) * F(sig, m)
        for i in range(-imax, imax + 1)) for j in range(-imax, imax + 1)) for m in range(M))

        self.w_timed = lambda t, sig: sig_h_f_term(t, sig) + onepsig_hh_f_term(t, sig) - h_F_term(t, sig) - hh_F_term(t, sig)



    # def _construct_w_timed(self):
    #     H = self.hydro.spatial_parameters['H']
    #     Hx = self.hydro.spatial_parameters_grad['H'][0]
    #     Hy = self.hydro.spatial_parameters_grad['H'][1]
    #     sigma_freq = self.hydro.constant_physical_parameters['sigma']

    #     scaling_vec = np.array([self.hydro.geometric_information['x_scaling'], self.hydro.geometric_information['y_scaling']])
    #     M = self.hydro.numerical_information['M']
    #     imax = self.hydro.numerical_information['imax']

    #     H_contribution_term1 = lambda t, sig: (1 / scaling_vec[0]) * sum(sum((Hx * self.hydro.alpha_solution[m][i] + H * ngsolve.grad(self.hydro.alpha_solution[m][i])[0]) * 
    #                                                                          self.hydro.time_basis.evaluation_function(t, i) * self.hydro.vertical_basis.integrated_evaluation_function(sig, m) for m in range(M)) for i in range(-imax, imax + 1))
    #     H_contribution_term2 = lambda t, sig: (1 / scaling_vec[1]) * sum(sum((Hy * self.hydro.beta_solution[m][i] + H * ngsolve.grad(self.hydro.beta_solution[m][i])[1]) * 
    #                                                                          self.hydro.time_basis.evaluation_function(t, i) * self.hydro.vertical_basis.integrated_evaluation_function(sig, m) for m in range(M)) for i in range(-imax, imax + 1))
    #     H_contribution_term3 = lambda t, sig: (sig / scaling_vec[0]) * sum(sum(Hx * self.hydro.alpha_solution[m][i] * self.hydro.time_basis.evaluation_function(t, i) * self.hydro.vertical_basis.evaluation_function(sig, m) for m in range(M)) for i in range(-imax, imax + 1))
    #     H_contribution_term4 = lambda t, sig: (sig / scaling_vec[1]) * sum(sum(Hy * self.hydro.beta_solution[m][i] * self.hydro.time_basis.evaluation_function(t, i) * self.hydro.vertical_basis.evaluation_function(sig, m) for m in range(M)) for i in range(-imax, imax + 1))

    #     zeta_contribution_term1 = lambda t, sig: self.surface_in_sigma * (1 / scaling_vec[0]) * \
    #                               sum(sum(sum((ngsolve.grad(self.hydro.gamma_solution[j])[0] * self.hydro.alpha_solution[m][i] + self.hydro.gamma_solution[j] * ngsolve.grad(self.hydro.alpha_solution[m][i])[0]) * 
    #                                            self.hydro.time_basis.evaluation_function(t, i) * self.hydro.vertical_basis.integrated_evaluation_function(sig, m) for m in range(M)) for i in range(-imax, imax + 1)) for j in range(-imax, imax + 1))
    #     zeta_contribution_term2 = lambda t, sig: self.surface_in_sigma * (1 / scaling_vec[1]) * \
    #                               sum(sum(sum((ngsolve.grad(self.hydro.gamma_solution[j])[1] * self.hydro.beta_solution[m][i] + self.hydro.gamma_solution[j] * ngsolve.grad(self.hydro.beta_solution[m][i])[1]) * 
    #                                            self.hydro.time_basis.evaluation_function(t, i) * self.hydro.vertical_basis.integrated_evaluation_function(sig, m) for m in range(M)) for i in range(-imax, imax + 1)) for j in range(-imax, imax + 1))
    #     zeta_contribution_term3 = lambda t, sig: self.surface_in_sigma * ((1+sig) / scaling_vec[0]) * \
    #                               sum(sum(sum(ngsolve.grad(self.hydro.gamma_solution[j])[0] * self.hydro.alpha_solution[m][i] * 
    #                                           self.hydro.time_basis.evaluation_function(t, i) * self.hydro.vertical_basis.evaluation_function(sig, m) for m in range(M)) for i in range(-imax, imax + 1)) for j in range(-imax, imax + 1))
    #     zeta_contribution_term4 = lambda t, sig: self.surface_in_sigma * ((1+sig) / scaling_vec[1]) * \
    #                               sum(sum(sum(ngsolve.grad(self.hydro.gamma_solution[j])[1] * self.hydro.beta_solution[m][i] * 
    #                                           self.hydro.time_basis.evaluation_function(t, i) * self.hydro.vertical_basis.evaluation_function(sig, m) for m in range(M)) for i in range(-imax, imax + 1)) for j in range(-imax, imax + 1))
        
    #     zeta_contribution_term5 = lambda t, sig: self.surface_in_sigma * sigma_freq * (1 + sig) * sum(self.hydro.gamma_solution[i] * self.hydro.time_basis.derivative_evaluation_function(t, i) for i in range(-imax, imax + 1))

    #     return lambda t, sig: H_contribution_term1(t, sig) + H_contribution_term2(t, sig) + H_contribution_term3(t, sig) + H_contribution_term4(t, sig) + zeta_contribution_term1(t, sig) + zeta_contribution_term2(t, sig) + \
    #                           zeta_contribution_term3(t, sig) + zeta_contribution_term4(t, sig) + zeta_contribution_term5(t, sig)
    

    def _construct_w_TA(self):
        
        M = self.hydro.numerical_information['M']
        imax = self.hydro.numerical_information['imax']

        L = self.hydro.geometric_information['x_scaling']
        B = self.hydro.geometric_information['y_scaling']

        H = self.hydro.spatial_parameters['H']
        Hx = self.hydro.spatial_parameters_grad['H'][0] / L
        Hy = self.hydro.spatial_parameters_grad['H'][1] / B        

        self.w_TA_sig_fm_coefficients = [0.5 * np.sqrt(2) * (self.hydro.alpha_solution[m][0] * Hx + self.hydro.beta_solution[m][0] * Hy) for m in range(M)]
        self.w_TA_onepsig_fm_coefficients = [sum(0.5 * self.hydro.gamma_grad[q][0] / L * self.hydro.alpha_solution[m][q] + 
                                                 0.5 * self.hydro.gamma_grad[q][1] / B * self.hydro.beta_solution[m][q] for q in range(-imax, imax + 1)) for m in range(M)]
        self.w_TA_FM_coefficients = [
            self.hydro.alpha_grad[m][0][0] * H / L + self.hydro.beta_grad[m][0][1] * H / B +
            self.hydro.alpha_solution[m][0] * Hx + self.hydro.beta_solution[m][0] * Hy + 
            0.5 * sum(
                self.hydro.gamma_solution[q] * self.hydro.alpha_grad[m][q][0] / L + self.hydro.gamma_solution[q] * self.hydro.beta_grad[m][q][1] / B +
                self.hydro.gamma_grad[q][0] * self.hydro.alpha_solution[m][q] / L + self.hydro.gamma_grad[q][1] * self.hydro.beta_solution[m][q] / B
                for q in range(-imax, imax + 1)
            )
            for m in range(M)
        ]

        vb = self.hydro.vertical_basis
        self.w_TA = lambda sig: sum(
            self.w_TA_sig_fm_coefficients[m] * sig * vb.evaluation_function(sig, m) + 
            self.w_TA_onepsig_fm_coefficients[m] * (1 + sig) * vb.evaluation_function(sig, m) + 
            self.w_TA_FM_coefficients[m] * vb.integrated_evaluation_function(sig, m)
            for m in range(M))


    def _construct_TWA_velocities(self):
        """Constructs thickness-weighted-averaged velocity functions, see Klingbeil et al. (2019)."""

        if self.surface_in_sigma:
            u_depth_correlation = lambda sig: self.hydro.spatial_parameters['H'] * np.sqrt(2) / 2 * sum(self.hydro.alpha_solution[m][0] * self.hydro.vertical_basis.evaluation_function(sig, m) for m in range(self.hydro.numerical_information['M'])) + \
                                0.5 * sum(sum(self.hydro.gamma_solution[i] * self.hydro.alpha_solution[m][i] * self.hydro.vertical_basis.evaluation_function(sig, m) for i in range(-self.hydro.numerical_information['imax'], self.hydro.numerical_information['imax'] + 1)) for m in range(self.hydro.numerical_information['M']))
            v_depth_correlation = lambda sig: self.hydro.spatial_parameters['H'] * np.sqrt(2) / 2 * sum(self.hydro.beta_solution[m][0] * self.hydro.vertical_basis.evaluation_function(sig, m) for m in range(self.hydro.numerical_information['M'])) + \
                                0.5 * sum(sum(self.hydro.gamma_solution[i] * self.hydro.beta_solution[m][i] * self.hydro.vertical_basis.evaluation_function(sig, m) for i in range(-self.hydro.numerical_information['imax'], self.hydro.numerical_information['imax'] + 1)) for m in range(self.hydro.numerical_information['M']))

            D_averaged = self.hydro.spatial_parameters['H'] + self.hydro.gamma_solution[0] * np.sqrt(2)/2
        else:
            u_depth_correlation = lambda sig: self.hydro.spatial_parameters['H'] * np.sqrt(2) / 2 * sum(self.hydro.alpha_solution[m][0] * self.hydro.vertical_basis.evaluation_function(sig, m) for m in range(self.hydro.numerical_information['M'])) 
            v_depth_correlation = lambda sig: self.hydro.spatial_parameters['H'] * np.sqrt(2) / 2 * sum(self.hydro.beta_solution[m][0] * self.hydro.vertical_basis.evaluation_function(sig, m) for m in range(self.hydro.numerical_information['M']))

            D_averaged = self.hydro.spatial_parameters['H']


        self.u_TWA = lambda sig: u_depth_correlation(sig) / D_averaged
        self.v_TWA = lambda sig: v_depth_correlation(sig) / D_averaged
        
        


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


    def animate_horizontal(self, quantity_func, savename: str, refinement_level: int=1, num_frames:int=128, exclude_ramping_zone: bool = True,
                           title: str='Colormap', clabel: str='Color', center_range: bool=False, contourlines: bool=False, num_levels: int=10,
                           subamplitude_lines: int=2, figsize: tuple=(7,4), show_mesh: bool =False, **kwargs):
        
        x_scaling = self.hydro.geometric_information['x_scaling']
        y_scaling = self.hydro.geometric_information['y_scaling']
        
        if self.hydro.numerical_information['mesh_generation_method'] != 'structured_quads':

            if exclude_ramping_zone:
                triangulation = get_triangulation(self.hydro.display_mesh.ngmesh)
            else:
                triangulation = get_triangulation(self.hydro.mesh.ngmesh)

            refiner = tri.UniformTriRefiner(triangulation)
            refined_triangulation = refiner.refine_triangulation(subdiv=refinement_level)
            
            eval_gfu = np.zeros((num_frames, refined_triangulation.x.shape[0]))
            for i in range(num_frames):
                eval_gfu[i, :] = evaluate_CF_range(quantity_func(i/num_frames), self.hydro.mesh, refined_triangulation.x, refined_triangulation.y)

            if center_range:
                cmap = 'RdBu'
                maxamp = np.amax(np.absolute(eval_gfu.flatten()))
                minamp = -maxamp
            else:
                cmap = 'viridis'
                maxamp = np.amax(eval_gfu.flatten())
                minamp = np.amin(eval_gfu.flatten())

            for i in range(num_frames):
                fig_colormap, ax_colormap = plt.subplots(figsize=figsize)

                if show_mesh:
                    ax_colormap.triplot(triangulation, linewidth=0.5, color='k', zorder=2)

                colormesh = ax_colormap.tripcolor(refined_triangulation, eval_gfu[i, :], vmin=minamp, vmax=maxamp, cmap=cmap, **kwargs)

                if contourlines:
                    try:
                        levels = np.linspace(np.min(eval_gfu), np.max(eval_gfu), num_levels*(subamplitude_lines+1))
                        contour = ax_colormap.tricontour(refined_triangulation, eval_gfu, levels, colors=['k'] + ["0.4"] * subamplitude_lines, linewidths=[.5] * (1+subamplitude_lines))
                        ax_colormap.clabel(contour, levels[0::subamplitude_lines+1], inline=1, fontsize=10, fmt='%1.4f')
                    except ValueError:
                        print("Constant solution; plotting contour lines impossible")

                phase_string = str(np.round(i/num_frames, 4))
                phase_string += '0' * (6 - len(phase_string))
                ax_colormap.set_title(f"{title}\n" + r"$t=$" + phase_string + r"$\sigma^{-1}$ s")

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

                if i == 0:
                    os.makedirs(savename)
                fig_colormap.savefig(f'{savename}/frame{i}')

        else:
            if exclude_ramping_zone:
                x = np.linspace(0, self.hydro.geometric_information['riverine_boundary_x'] / self.hydro.geometric_information['x_scaling'], self.hydro.numerical_information['grid_size'][0] * (refinement_level + 1) + 1) # This is a temporary solution; we assume now that if a structured grid is used, then the (scaled) domain is a unit square
            else:
                x = np.linspace((-self.hydro.geometric_information['L_BL_sea']-self.hydro.geometric_information['L_R_sea']-self.hydro.geometric_information['L_RA_sea'])/x_scaling,
                                (self.hydro.geometric_information['riverine_boundary_x']+self.hydro.geometric_information['L_RA_river']+self.hydro.geometric_information['L_R_river'] + self.hydro.geometric_information['L_BL_river'])/x_scaling,
                                self.hydro.numerical_information['grid_size'][0] * (refinement_level + 1) + 1)
            y = np.linspace(-0.5, 0.5, self.hydro.numerical_information['grid_size'][1] * (refinement_level + 1) + 1)
            X, Y = np.meshgrid(x, y, indexing='ij')
            Q = np.array([np.zeros_like(X) for _ in range(num_frames)])
            
            for j in range(num_frames):
                for i in range(Q.shape[2]):
                    Q[j, :, i] = evaluate_CF_range(quantity_func(j/num_frames), self.hydro.mesh, x, y[i] * np.ones_like(x))

            if center_range:
                cmap = 'RdBu'
                maxamp = np.amax(np.absolute(Q.flatten()))
                minamp = -maxamp
            else:
                cmap = 'viridis'
                maxamp = np.amax(Q.flatten())
                minamp = np.amin(Q.flatten())

            for i in range(num_frames):
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

                colormesh = ax_colormap.pcolormesh(X, Y, Q[i, :, :], vmin=minamp, vmax=maxamp, cmap='RdBu', **kwargs)

                if contourlines:
                    try:
                        levels = np.linspace(np.min(Q[i, :, :].flatten()), np.max(Q[i, :, :].flatten()), num_levels*(subamplitude_lines+1))
                        contour = ax_colormap.contour(X, Y, Q[i, :, :], levels, colors=['k'] + ["0.4"] * subamplitude_lines, linewidths=[.5] * (1+subamplitude_lines))
                        ax_colormap.clabel(contour, levels[0::subamplitude_lines+1], inline=1, fontsize=10, fmt='%1.4f')
                    except ValueError:
                        print("Constant solution; plotting contour lines impossible")

                phase_string = str(np.round(i/num_frames, 4))
                phase_string += '0' * (6 - len(phase_string))
                ax_colormap.set_title(f"{title}\n" + r"$t=$" + phase_string + r"$\sigma^{-1}$ s")

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
                if i == 0:
                    os.makedirs(savename)
                fig_colormap.savefig(f'{savename}/frame{i}')


    def plot_horizontal_vectorfield(self, x_field, y_field, background_colorfield=None, num_x:int=40, num_y:int=40, arrow_color='white', title: str='Vector Field', clabel:str='Colour',
                                    save: str=None, exclude_ramping_zone: bool=True, center_range:bool=False, figsize=(7,4), length_indication='alpha', **kwargs):
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
            xquiv = np.linspace(0, self.hydro.geometric_information['riverine_boundary_x'] / self.hydro.geometric_information['x_scaling'], num_x) # This is a temporary solution; we assume now that if a structured grid is used, then the (scaled) domain is a unit square
            xbackground = np.linspace(0, self.hydro.geometric_information['riverine_boundary_x'] / self.hydro.geometric_information['x_scaling'], 300) # This is a temporary solution; we assume now that if a structured grid is used, then the (scaled) domain is a unit square
        else:
            xquiv = np.linspace(-(self.hydro.geometric_information['L_BL_sea']-self.hydro.geometric_information['L_R_sea']-self.hydro.geometric_information['L_RA_sea']) / self.hydro.geometric_information['x_scaling'],
                            (self.hydro.geometric_information['riverine_boundary_x']+self.hydro.geometric_information['L_RA_river']+self.hydro.geometric_information['L_R_river'] + self.hydro.geometric_information['L_BL_river'])/self.hydro.geometric_information['x_scaling'],
                            num_x)
            xbackground = np.linspace(-(self.hydro.geometric_information['L_BL_sea']-self.hydro.geometric_information['L_R_sea']-self.hydro.geometric_information['L_RA_sea']) / self.hydro.geometric_information['x_scaling'],
                            (self.hydro.geometric_information['riverine_boundary_x']+self.hydro.geometric_information['L_RA_river']+self.hydro.geometric_information['L_R_river'] + self.hydro.geometric_information['L_BL_river']) / self.hydro.geometric_information['x_scaling'],
                            300)
        yquiv = np.linspace(-0.5, 0.5, num_y)
        ybackground = np.linspace(-0.5, 0.5, 300)
        X, Y = np.meshgrid(xquiv, yquiv, indexing='ij')
        Xbackground, Ybackground = np.meshgrid(xbackground, ybackground, indexing='ij')
        Xquiv = np.zeros_like(X)
        Yquiv = np.zeros_like(Y)
        C = np.zeros_like(Xbackground)

        for i in range(Xquiv.shape[1]):
            Xquiv[:, i] = evaluate_CF_range(x_field, self.hydro.mesh, xquiv, yquiv[i] * np.ones_like(xquiv))
            Yquiv[:, i] = evaluate_CF_range(y_field, self.hydro.mesh, xquiv, yquiv[i] * np.ones_like(xquiv))

        fig, ax = plt.subplots(figsize=figsize)

        if background_colorfield is not None:
            for i in range(Xbackground.shape[1]):
                C[:, i] = evaluate_CF_range(background_colorfield, self.hydro.mesh, xbackground, ybackground[i] * np.ones_like(xbackground))

            if center_range:
                maxamp = np.amax(np.absolute(C))
                background = ax.pcolormesh(Xbackground, Ybackground, C, vmin=-maxamp, vmax=maxamp, cmap='RdBu', **kwargs)
            else:
                background = ax.pcolormesh(Xbackground, Ybackground, C, **kwargs)

            cbar = fig.colorbar(background)
            cbar.ax.set_ylabel(clabel)

        visual_norms = np.sqrt((Xquiv/self.hydro.geometric_information['x_scaling'])**2 + (Yquiv/self.hydro.geometric_information['y_scaling'])**2)
        norms = np.sqrt(Xquiv**2 + Yquiv**2)

        if length_indication == 'alpha':
            arrows = ax.quiver(X, Y, (Xquiv/self.hydro.geometric_information['x_scaling'])/visual_norms, (Yquiv/self.hydro.geometric_information['y_scaling'])/visual_norms, color=arrow_color, pivot='mid', alpha=norms / np.amax(norms))
        elif length_indication == 'length':
            arrows = ax.quiver(X, Y, (Xquiv/self.hydro.geometric_information['x_scaling']), (Yquiv/self.hydro.geometric_information['y_scaling']), color=arrow_color, pivot='mid')
        elif length_indication == 'none':
            arrows = ax.quiver(X, Y, (Xquiv/self.hydro.geometric_information['x_scaling'])/visual_norms, (Yquiv/self.hydro.geometric_information['y_scaling'])/visual_norms, color=arrow_color, pivot='mid')

        ax.set_title(title + f'\nMaximum magnitude of arrows: {np.round(np.amax(norms), 8)}')
        
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


    def animate_horizontal_vectorfield(self, x_field_func, y_field_func, savename:str, background_colorfield_func=None, num_frames:int=128, num_x:int=40, num_y:int=40, arrow_color='k', title='Vector Field', clabel:str='Colour',
                                       exclude_ramping_zone: bool=True, center_range:bool=False, figsize=(7,4), **kwargs):
        
        if exclude_ramping_zone:
            xquiv = np.linspace(0, self.hydro.geometric_information['riverine_boundary_x'] / self.hydro.geometric_information['x_scaling'], num_x) # This is a temporary solution; we assume now that if a structured grid is used, then the (scaled) domain is a unit square
            xbackground = np.linspace(0, self.hydro.geometric_information['riverine_boundary_x'] / self.hydro.geometric_information['x_scaling'], 300) # This is a temporary solution; we assume now that if a structured grid is used, then the (scaled) domain is a unit square
        else:
            xquiv = np.linspace(-(self.hydro.geometric_information['L_BL_sea']-self.hydro.geometric_information['L_R_sea']-self.hydro.geometric_information['L_RA_sea']) / self.hydro.geometric_information['x_scaling'],
                            (self.hydro.geometric_information['riverine_boundary_x']+self.hydro.geometric_information['L_RA_river']+self.hydro.geometric_information['L_R_river'] + self.hydro.geometric_information['L_BL_river'])/self.hydro.geometric_information['x_scaling'],
                            num_x)
            xbackground = np.linspace(-(self.hydro.geometric_information['L_BL_sea']-self.hydro.geometric_information['L_R_sea']-self.hydro.geometric_information['L_RA_sea']) / self.hydro.geometric_information['x_scaling'],
                            (self.hydro.geometric_information['riverine_boundary_x']+self.hydro.geometric_information['L_RA_river']+self.hydro.geometric_information['L_R_river'] + self.hydro.geometric_information['L_BL_river']) / self.hydro.geometric_information['x_scaling'],
                            300)
        yquiv = np.linspace(-0.5, 0.5, num_y)
        ybackground = np.linspace(-0.5, 0.5, 300)
        X, Y = np.meshgrid(xquiv, yquiv, indexing='ij')
        Xbackground, Ybackground = np.meshgrid(xbackground, ybackground, indexing='ij')
        Xquiv = np.array([np.zeros_like(X) for _ in range(num_frames)])
        Yquiv = np.array([np.zeros_like(Y) for _ in range(num_frames)])
        C = np.array([np.zeros_like(Xbackground) for _ in range(num_frames)])

        for j in range(num_frames):
            for i in range(Xquiv.shape[1]):
                Xquiv[j, :, i] = evaluate_CF_range(x_field_func(j/num_frames), self.hydro.mesh, xquiv, yquiv[i] * np.ones_like(xquiv))
                Yquiv[j, :, i] = evaluate_CF_range(y_field_func(j/num_frames), self.hydro.mesh, xquiv, yquiv[i] * np.ones_like(xquiv))


        if background_colorfield_func is not None:
            for j in range(num_frames):
                for i in range(Xbackground.shape[1]):
                    C[j, :, i] = evaluate_CF_range(background_colorfield_func(j/num_frames), self.hydro.mesh, xbackground, ybackground[i] * np.ones_like(xbackground))

            if center_range:
                maxamp = np.amax(np.absolute(C.flatten()))
                minamp = -maxamp
                cmap = 'RdBu'
            else:
                maxamp = np.amax(C.flatten())
                minamp = np.amin(C.flatten())

        visual_norms = np.sqrt((Xquiv/self.hydro.geometric_information['x_scaling'])**2 + (Yquiv/self.hydro.geometric_information['y_scaling'])**2)
        norms = np.sqrt(Xquiv**2 + Yquiv**2)

        for i in range(num_frames):
            
            fig, ax = plt.subplots(figsize=figsize)
            if background_colorfield_func is not None:
                background = ax.pcolormesh(Xbackground, Ybackground, C[i, :, :], vmin=minamp, vmax=maxamp, cmap=cmap, **kwargs)

                cbar = fig.colorbar(background)
                cbar.ax.set_ylabel(clabel)


            arrows = ax.quiver(X, Y, (Xquiv[i,:,:]/self.hydro.geometric_information['x_scaling'])/visual_norms[i,:,:], (Yquiv[i,:,:]/self.hydro.geometric_information['y_scaling'])/visual_norms[i,:,:], color=arrow_color, pivot='mid', alpha=norms[i,:,:] / np.amax(norms.flatten()))

            phase_string = str(np.round(i/num_frames, 4))
            phase_string += '0' * (6 - len(phase_string))
            ax.set_title(f"{title}\n" + r"$t=$" + phase_string + r"$\sigma^{-1}$ s" + f'\nMaximum magnitude of arrows: {np.round(np.amax(norms.flatten()), 8)}')
            
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

            plt.tight_layout() 
            if i == 0:
                os.makedirs(savename)
            fig.savefig(f"{savename}/frame{i}")


    def plot_vertical_profile_at_point(self, p, num_vertical_points, constituent_index, **kwargs):
        
        H = self.hydro.spatial_parameters['H']
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


    def plot_vertical_cross_section(self, quantity_function, p1, p2, num_horizontal_points=500, num_vertical_points=500, title='Cross-section', clabel='Color', center_range=False, save=None, contourlines=True, num_levels=None, figsize=(7,4), **kwargs):
        
        scaling_vec = np.array([self.hydro.geometric_information['x_scaling'], self.hydro.geometric_information['y_scaling']])
        width = np.linalg.norm((p1-p2) * scaling_vec, 2) / 1e3
        s_range = np.linspace(width/2, -width/2, num_horizontal_points)
        x_range = np.linspace(p1[0], p2[0], num_horizontal_points)
        y_range = np.linspace(p1[1], p2[1], num_horizontal_points)

        H = self.hydro.spatial_parameters['H']

        depth = evaluate_CF_range(H, self.hydro.mesh, x_range, y_range)

        s_grid = -np.tile(s_range, (num_vertical_points, 1))
        z_grid = np.array([np.linspace(-depth[i], 0, num_vertical_points) for i in range(num_horizontal_points)]).T

        if isinstance(quantity_function, np.ndarray):
            Q = quantity_function
        else:
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

    
    def plot_sigma_vertical_cross_section(self, quantity_function, p1, p2, num_horizontal_points=500, num_vertical_points=500, title=r'Cross-section in $\varsigma$-coordinates',
                                          clabel='Quantity', center_range=False, save=None, contourlines=True, num_levels=None, figsize=(7,4), **kwargs):
        
        scaling_vec = np.array([self.hydro.geometric_information['x_scaling'], self.hydro.geometric_information['y_scaling']])
        width = np.linalg.norm((p1-p2) * scaling_vec, 2) / 1e3
        s_range = np.linspace(width/2, -width/2, num_horizontal_points)
        x_range = np.linspace(p1[0], p2[0], num_horizontal_points)
        y_range = np.linspace(p1[1], p2[1], num_horizontal_points)

        s_grid = -np.tile(s_range, (num_vertical_points, 1))
        sig_grid = np.array([np.linspace(-1, 0, num_vertical_points) for _ in range(num_horizontal_points)]).T

        if isinstance(quantity_function, np.ndarray):
            Q = quantity_function
        else:
            Q = evaluate_vertical_structure_at_cross_section(self.hydro.mesh, quantity_function, p1, p2, num_horizontal_points, num_vertical_points)

        if center_range:
            maxamp = max(np.amax(Q), -np.amin(Q))

        fig_crosssection, ax_crosssection = plt.subplots(figsize=figsize)
        if center_range:
            color_crosssection = ax_crosssection.pcolormesh(s_grid, sig_grid, Q, vmin=-maxamp, vmax=maxamp, **kwargs)
        else:
            color_crosssection = ax_crosssection.pcolormesh(s_grid, sig_grid, Q, **kwargs)
        cbar_crosssection = plt.colorbar(color_crosssection)
        cbar_crosssection.ax.set_ylabel(clabel)

        if contourlines:
            if num_levels is None:
                num_levels = 8
            subamplitude_lines = 2

            levels = np.linspace(np.min(Q), np.max(Q), num_levels*(subamplitude_lines+1))
            contour = ax_crosssection.contour(s_grid, sig_grid, Q, levels, colors=['k'] + ["0.4"] * subamplitude_lines, linewidths=[.5] * (1+subamplitude_lines))
            ax_crosssection.clabel(contour, levels[0::subamplitude_lines+1], inline=1, fontsize=10, fmt='%1.4f')
            
        ax_crosssection.set_title(title)
        ax_crosssection.set_xlabel('Distance along cross-section [km]')
        ax_crosssection.set_ylabel(r'$\varsigma$ [-]')

        plt.tight_layout()

        if save is not None:
            fig_crosssection.savefig(save)

        
        

    # def animate_vertical_cross_section(self): TO DO
    #     pass

    
    def get_w_cross_section(self, x, constituent=0, num_horizontal_points=500, num_vertical_points=500, dx=0.05):
        """Currently only works for R=0. And we need the fact that free surface effects are not yet taken into account (so continuity is a linear equation)."""
        p1 = np.array([x, -0.5])
        p2 = np.array([x, 0.5])
        scaling_vec = np.array([self.hydro.geometric_information['x_scaling'], self.hydro.geometric_information['y_scaling']])

        H = self.hydro.spatial_parameters['H']
        Hx = self.hydro.spatial_parameters_grad['H'][0]
        Hy = self.hydro.spatial_parameters_grad['H'][1]

        sig_grid = np.array([np.linspace(-1, 0, num_vertical_points) for i in range(num_horizontal_points)]).T
        depth_grid = evaluate_vertical_structure_at_cross_section(self.hydro.mesh, lambda sig: H, p1, p2, num_horizontal_points, num_vertical_points)
        Hx_grid = evaluate_vertical_structure_at_cross_section(self.hydro.mesh, lambda sig: Hx, p1, p2, num_horizontal_points, num_vertical_points)
        Hy_grid = evaluate_vertical_structure_at_cross_section(self.hydro.mesh, lambda sig: Hy, p1, p2, num_horizontal_points, num_vertical_points)

        DUx_plus = (1/scaling_vec[0]) * evaluate_vertical_structure_at_cross_section(self.hydro.mesh, lambda sig: self.u(constituent, sig) * H, np.array([x+dx/2, -0.5]), np.array([x+dx/2, 0.5]), num_horizontal_points, num_vertical_points)
        DUx_minus = (1/scaling_vec[0]) * evaluate_vertical_structure_at_cross_section(self.hydro.mesh, lambda sig: self.u(constituent, sig) * H, np.array([x-dx/2, -0.5]), np.array([x-dx/2, 0.5]), num_horizontal_points, num_vertical_points)
        DUx = (DUx_plus - DUx_minus) / dx

        DV = evaluate_vertical_structure_at_cross_section(self.hydro.mesh, lambda sig: self.v(constituent, sig) * H, p1, p2, num_horizontal_points, num_vertical_points)
        DVy = np.zeros_like(DV)
        DVy[:, 1:-1] = (DV[:, 2:] - DV[:, :-2]) * num_horizontal_points / 2
        DVy[:, 0] = (DV[:, 1] - DV[:, 0]) * num_horizontal_points
        DVy[:, -1] = (DV[:, -1] - DV[:, -2]) * num_horizontal_points
        DVy *= (1/scaling_vec[1])

        Wtilde = -np.power(depth_grid, -1) / num_vertical_points * np.cumsum(DUx + DVy, axis=0)
        W = depth_grid * Wtilde + (1/scaling_vec[0]) * (sig_grid * Hx_grid) * evaluate_vertical_structure_at_cross_section(self.hydro.mesh, lambda sig: self.u(constituent, sig), p1, p2, num_horizontal_points, num_vertical_points) + \
            (1/scaling_vec[1]) * (sig_grid * Hy_grid) * evaluate_vertical_structure_at_cross_section(self.hydro.mesh, lambda sig: self.v(constituent, sig), p1, p2, num_horizontal_points, num_vertical_points)
        
        return W
    

    # def get_w_timed_cross_section(self, x, num_horizontal_points=500, num_vertical_points=500, dx=0.05):

    #     p1 = np.array([x, -0.5])
    #     p2 = np.array([x, 0.5])
    #     scaling_vec = np.array([self.hydro.geometric_information['x_scaling'], self.hydro.geometric_information['y_scaling']])

    #     sigma_freq = self.hydro.constant_physical_parameters['sigma']

    #     H = self.hydro.spatial_parameters['H']
    #     Hx = self.hydro.spatial_parameters_grad['H'][0]
    #     Hy = self.hydro.spatial_parameters_grad['H'][1]

    #     sig_grid = np.array([np.linspace(-1, 0, num_vertical_points) for i in range(num_horizontal_points)]).T
    #     Hx_grid = evaluate_vertical_structure_at_cross_section(self.hydro.mesh, lambda sig: Hx, p1, p2, num_horizontal_points, num_vertical_points)
    #     Hy_grid = evaluate_vertical_structure_at_cross_section(self.hydro.mesh, lambda sig: Hy, p1, p2, num_horizontal_points, num_vertical_points)

    #     if self.surface_in_sigma:
    #         zeta_grid = lambda t: evaluate_vertical_structure_at_cross_section(self.hydro.mesh, lambda sig: self.zeta_timed(t), p1, p2, num_horizontal_points, num_vertical_points)
    #         zetat_grid = lambda t: evaluate_vertical_structure_at_cross_section(self.hydro.mesh, lambda sig: self.zetat_timed(t), p1, p2, num_horizontal_points, num_vertical_points)

    #         zetax_plus = lambda t: evaluate_vertical_structure_at_cross_section(self.hydro.mesh, lambda sig: self.zeta_timed(t), np.array([x+dx/2, -0.5]), np.array([x+dx/2, 0.5]), num_horizontal_points, 1)
    #         zetax_minus = lambda t: evaluate_vertical_structure_at_cross_section(self.hydro.mesh, lambda sig: self.zeta_timed(t), np.array([x-dx/2, -0.5]), np.array([x-dx/2, 0.5]), num_horizontal_points, 1)
    #         zetax_grid = lambda t: np.tile((1/dx) * (zetax_plus(t) - zetax_minus(t)), (1, sig_grid.shape[1]))

    #     DUx_plus = lambda t: (1/scaling_vec[0]) * evaluate_vertical_structure_at_cross_section(self.hydro.mesh, lambda sig: self.u_timed(t, sig) * (H + self.surface_in_sigma * self.zeta_timed(t)), np.array([x+dx/2, -0.5]), np.array([x+dx/2, 0.5]), num_horizontal_points, num_vertical_points)
    #     DUx_minus = lambda t: (1/scaling_vec[0]) * evaluate_vertical_structure_at_cross_section(self.hydro.mesh, lambda sig: self.u_timed(t, sig) * (H + self.surface_in_sigma * self.zeta_timed(t)), np.array([x-dx/2, -0.5]), np.array([x-dx/2, 0.5]), num_horizontal_points, num_vertical_points)
    #     DUx = lambda t: (DUx_plus(t) - DUx_minus(t)) / dx

    #     DV = lambda t: evaluate_vertical_structure_at_cross_section(self.hydro.mesh, lambda sig: self.v_timed(t, sig) * (H + self.surface_in_sigma * self.zeta_timed(t)), p1, p2, num_horizontal_points, num_vertical_points)
        
    #     def DVy(t):
    #         DV_eval = DV(t)
    #         DVy_eval = np.zeros_like(DV_eval)
    #         DVy_eval[:, 1:-1] = (DV_eval[:, 2:] - DV_eval[:, :-2]) * num_horizontal_points / 2
    #         DVy_eval[:, 0] = (DV_eval[:, 1] - DV_eval[:, 0]) * num_horizontal_points
    #         DVy_eval[:, -1] = (DV_eval[:, -1] - DV_eval[:, -2]) * num_horizontal_points
    #         DVy_eval *= (1/scaling_vec[1])

    #     Wtilde_contribution = lambda t: 1 / num_vertical_points * np.cumsum(DUx(t) + DVy(t), axis=0)
    #     zetat_contribution = lambda t: sigma_freq * (np.ones_like(sig_grid) + sig_grid) * self.zetat_timed(t) * self.surface_in_sigma
    #     u_contribution = lambda t: (1/scaling_vec[0]) * (self.surface_in_sigma)

    #     W = lambda tWtilde + self.surface_in_sigma * sigma_freq * (1 + sig_grid) * zetat_grid + (1/scaling_vec[0]) * (sig_grid * Hx_grid) * evaluate_vertical_structure_at_cross_section(self.hydro.mesh, lambda sig: self.u(constituent, sig), p1, p2, num_horizontal_points, num_vertical_points) + \
    #         (1/scaling_vec[1]) * (sig_grid * Hy_grid) * evaluate_vertical_structure_at_cross_section(self.hydro.mesh, lambda sig: self.v(constituent, sig), p1, p2, num_horizontal_points, num_vertical_points)



    def plot_cross_section_circulation(self, x, stride: int, num_horizontal_points: int = 500, num_vertical_points: int = 500, phase: float = 0, constituent=0, flowrange: tuple=None, figsize=(7,4), spacing='equal', alpha='equal', save=None):
        p1 = np.array([x, -0.5])
        p2 = np.array([x, 0.5])
        width = np.linalg.norm(p1-p2, 2)
        s_range = np.linspace(-width/2, width/2, num_horizontal_points)
        x_range = np.linspace(p1[0], p2[0], num_horizontal_points)
        y_range = np.linspace(p1[1], p2[1], num_horizontal_points)

        H = self.hydro.spatial_parameters['H']

        depth = evaluate_CF_range(H, self.hydro.mesh, x_range, y_range)

        s_grid = np.tile(s_range, (num_vertical_points, 1))
        z_grid = np.array([np.linspace(-depth[i], 0, num_vertical_points) for i in range(num_horizontal_points)]).T

        # if constituent == 'all':
        #     Q = evaluate_vertical_structure_at_cross_section(self.hydro.mesh, lambda sig: self.u_timed(phase / self.hydro.constant_physical_parameters['sigma'], sig), p1, p2, num_horizontal_points, num_vertical_points)
        #     V = evaluate_vertical_structure_at_cross_section(self.hydro.mesh, lambda sig: self.v_timed(phase/ self.hydro.constant_physical_parameters['sigma'], sig), p1, p2, num_horizontal_points, num_vertical_points)
        #     W = evaluate_vertical_structure_at_cross_section(self.hydro.mesh, lambda sig: self.w_timed(phase/ self.hydro.constant_physical_parameters['sigma'], sig), p1, p2, num_horizontal_points, num_vertical_points)
        if constituent == 0:
            Q = evaluate_vertical_structure_at_cross_section(self.hydro.mesh, lambda sig: self.u(constituent, sig) * self.hydro.time_basis.evaluation_function(phase / self.hydro.constant_physical_parameters['sigma'], constituent), p1, p2,
                                                             num_horizontal_points, num_vertical_points)
            V = evaluate_vertical_structure_at_cross_section(self.hydro.mesh, lambda sig: self.v(constituent, sig) * self.hydro.time_basis.evaluation_function(phase / self.hydro.constant_physical_parameters['sigma'], constituent), p1, p2,
                                                             num_horizontal_points, num_vertical_points)
            W = self.get_w_cross_section(x, constituent=constituent, dx=0.01) * self.hydro.time_basis.evaluation_function(phase, constituent)
        else:
            Q = evaluate_vertical_structure_at_cross_section(self.hydro.mesh, lambda sig: self.u(constituent, sig) * self.hydro.time_basis.evaluation_function(phase, constituent) + \
                                                             self.u(-constituent, sig) * self.hydro.time_basis.evaluation_function(phase, -constituent), p1, p2,
                                                             num_horizontal_points, num_vertical_points)
            V = evaluate_vertical_structure_at_cross_section(self.hydro.mesh, lambda sig: self.v(constituent, sig) * self.hydro.time_basis.evaluation_function(phase, constituent) + \
                                                             self.v(-constituent, sig) * self.hydro.time_basis.evaluation_function(phase, -constituent), p1, p2,
                                                             num_horizontal_points, num_vertical_points)
            W = self.get_w_cross_section(x, constituent=constituent, dx=0.01) * self.hydro.time_basis.evaluation_function(phase, constituent) + self.get_w_cross_section(x, constituent=-constituent, dx=0.01) * self.hydro.time_basis.evaluation_function(phase, -constituent)

        
        if flowrange is None:
            maxamp = max(np.amax(Q), -np.amin(Q))
            minamp = -maxamp
        else:
            maxamp = flowrange[1]
            minamp = flowrange[0]

        sig_range = np.linspace(-1, 0, num_vertical_points)
        num_arrows_z = sig_range[::stride].shape[0]
        zquiv_grid = np.array([np.linspace(-np.amax(depth), 0, num_arrows_z) for i in range(num_horizontal_points // stride + 1)]).T
        squiv_grid = s_grid[::stride, ::stride]

        Vquiv = np.zeros_like(zquiv_grid)
        Wquiv = np.zeros_like(zquiv_grid)
        mask = np.zeros_like(zquiv_grid)
        

        for y_index in range(num_horizontal_points // stride + (num_horizontal_points % stride != 0)):
            local_depth = evaluate_CF_point(H, self.hydro.mesh, x, squiv_grid[0, y_index])
            for z_index in range(num_vertical_points // stride):
                if zquiv_grid[z_index, y_index] > -local_depth:
                    mask[z_index, y_index] = 1
                    sig_value = zquiv_grid[z_index, y_index] / local_depth
                    corresponding_sig_index = np.argmin(np.absolute(sig_range - np.ones_like(sig_range) * sig_value))
                    if sig_value >= sig_range[corresponding_sig_index]: # interpolation between sig_range[corresponding_sig_index] and sig_range[corresponding_sig_index + 1]
                        Vquiv[z_index, y_index] = V[corresponding_sig_index, y_index * stride] * (sig_range[corresponding_sig_index + 1] - sig_value) + V[corresponding_sig_index + 1, y_index * stride] * (sig_value - sig_range[corresponding_sig_index])
                        Vquiv[z_index, y_index] /= sig_range[corresponding_sig_index + 1] - sig_range[corresponding_sig_index]
                        Wquiv[z_index, y_index] = W[corresponding_sig_index, y_index * stride] * (sig_range[corresponding_sig_index + 1] - sig_value) + W[corresponding_sig_index + 1, y_index * stride] * (sig_value - sig_range[corresponding_sig_index])
                        Wquiv[z_index, y_index] /= sig_range[corresponding_sig_index + 1] - sig_range[corresponding_sig_index]
                    else:
                        Vquiv[z_index, y_index] = V[corresponding_sig_index - 1, y_index * stride] * (sig_range[corresponding_sig_index] - sig_value) + V[corresponding_sig_index, y_index * stride] * (sig_value - sig_range[corresponding_sig_index - 1])
                        Vquiv[z_index, y_index] /= sig_range[corresponding_sig_index] - sig_range[corresponding_sig_index - 1]
                        Wquiv[z_index, y_index] = W[corresponding_sig_index - 1, y_index * stride] * (sig_range[corresponding_sig_index] - sig_value) + W[corresponding_sig_index, y_index * stride] * (sig_value - sig_range[corresponding_sig_index - 1])
                        Wquiv[z_index, y_index] /= sig_range[corresponding_sig_index] - sig_range[corresponding_sig_index - 1]
            
        fig_crosssection, ax_crosssection = plt.subplots(figsize=figsize)
        color_crosssection = ax_crosssection.pcolormesh(s_grid, z_grid, Q, vmin=minamp, vmax=maxamp, cmap='RdBu')
        cbar_crosssection = plt.colorbar(color_crosssection)
        cbar_crosssection.ax.set_ylabel('Longitudinal velocity [m/s]')

        ax_crosssection.plot(s_range, -depth, linewidth=1, color='k', zorder=3)
        ax_crosssection.fill_between(s_range, -np.amax(depth), -depth, color='silver')
        
        if spacing == 'sigma':
            visual_norms = np.sqrt((V[::stride,::stride] / (width*self.hydro.geometric_information['y_scaling']))**2 + (W[::stride,::stride] / np.amax(depth))**2) # y-dimension in km
            physical_norms = np.sqrt((V[::stride,::stride])**2 + (W[::stride,::stride])**2)
            if alpha == 'variable':
                quiv = ax_crosssection.quiver(s_grid[::stride,::stride], z_grid[::stride,::stride], (V[::stride,::stride] / (width*self.hydro.geometric_information['y_scaling'])) / visual_norms, (W[::stride,::stride] / np.amax(depth)) / visual_norms, color='k', alpha=physical_norms/np.amax(physical_norms), pivot='mid')
            elif alpha == 'equal':
                quiv = ax_crosssection.quiver(s_grid[::stride,::stride], z_grid[::stride,::stride], (V[::stride,::stride] / (width*self.hydro.geometric_information['y_scaling'])) / visual_norms, (W[::stride,::stride] / np.amax(depth)) / visual_norms, color='k', pivot='mid')
        elif spacing == 'equal':
            visual_norms = np.sqrt((Vquiv / (width*self.hydro.geometric_information['y_scaling']))**2 + (Wquiv / np.amax(depth))**2) # y-dimension in km
            physical_norms = np.sqrt((Vquiv)**2 + (Wquiv)**2)
            if alpha == 'variable':
                quiv = ax_crosssection.quiver(squiv_grid, zquiv_grid, (Vquiv / (width*self.hydro.geometric_information['y_scaling'])) / visual_norms, (Wquiv / np.amax(depth)) / visual_norms, color='k', alpha=physical_norms/np.amax(physical_norms), pivot='mid')
            elif alpha == 'equal':
                quiv = ax_crosssection.quiver(squiv_grid, zquiv_grid, (Vquiv / (width*self.hydro.geometric_information['y_scaling'])) / visual_norms, (Wquiv / np.amax(depth)) / visual_norms, color='k', pivot='mid')


        xticks = list(np.linspace(-0.5, 0.5, 5))
        ax_crosssection.set_xticks(xticks)
        ax_crosssection.set_xticklabels(list(np.linspace(-0.5, 0.5, 5) * self.hydro.geometric_information['y_scaling'] / 1000))

        ax_crosssection.set_xlabel('Distance along cross-section [km]')
        ax_crosssection.set_ylabel('-Depth [m]')

        
        if constituent != 0:
            ax_crosssection.set_title(f'Lateral flow at t = {phase}' + r'$\sigma^{-1}$' f' s\nMaximum lateral velocity = {np.round(np.amax(physical_norms),8)}')
        else:
            ax_crosssection.set_title(f'Maximum lateral velocity = {np.round(np.amax(physical_norms),8)}')

        if save is not None:
            fig_crosssection.savefig(save)


    def get_w_in_cross_section(self, t_arr, sig_arr, x=None, y=None, h=0.005, num_horizontal_points=101, print_log=False):
        """
        To save costly computations in computing w, we compute it using central difference approximated derivatives in a cross-section with either x or y constant
        
        """

        if (x is None and y is None) or (x is not None and y is not None):
            raise ValueError("Please provide either x or y (exclusive or)")
        elif x is not None:
            p1 = np.array([x, -0.5])
            p2 = np.array([x, 0.5])
            y_range = np.linspace(-0.5, 0.5, num_horizontal_points)
            x_range = np.ones_like(y_range) * x
        elif y is not None:
            p1 = np.array([0, y])
            p2 = np.array([1, y])        
            x_range = np.linspace(0, 1, num_horizontal_points)
            y_range = np.ones_like(x_range) * y

        H_cf = self.hydro.spatial_parameters['H']
        Hx_cf = self.hydro.spatial_parameters_grad['H'][0]
        Hy_cf = self.hydro.spatial_parameters_grad['H'][1]

        H = evaluate_CF_range(H_cf, self.hydro.mesh, x_range, y_range)
        Hx = evaluate_CF_range(Hx_cf, self.hydro.mesh, x_range, y_range)
        Hy = evaluate_CF_range(Hy_cf, self.hydro.mesh, x_range, y_range)

        L = self.hydro.geometric_information['x_scaling']
        B = self.hydro.geometric_information['y_scaling']
        M = self.hydro.numerical_information['M']
        imax = self.hydro.numerical_information['imax']
        
        W = np.zeros((t_arr.shape[0], sig_arr.shape[0], num_horizontal_points))
        if print_log:
            print('Logging computation of W')
        for i, t in enumerate(np.nditer(t_arr)):
            if print_log:
                print(f"t = {t}")
            Z = evaluate_CF_range(self.zeta_timed(t), self.hydro.mesh, x_range, y_range)
            U = np.zeros((M, num_horizontal_points))
            V = np.zeros((M, num_horizontal_points))
            Ux = np.zeros_like(U)
            Vy = np.zeros_like(V)
            for m in range(M):
                U[m, :] = evaluate_CF_range(sum(self.hydro.alpha_solution[m][j] * self.hydro.time_basis.evaluation_function(t, j) for j in range(-imax, imax + 1)), self.hydro.mesh, x_range, y_range)
                V[m, :] = evaluate_CF_range(sum(self.hydro.beta_solution[m][j] * self.hydro.time_basis.evaluation_function(t, j) for j in range(-imax, imax + 1)), self.hydro.mesh, x_range, y_range)

            if x is not None:
                Zx = (evaluate_CF_range(self.zeta_timed(t), self.hydro.mesh, x_range + 0.5 * h * np.ones_like(x_range), y_range) - evaluate_CF_range(self.zeta_timed(t), self.hydro.mesh, x_range - 0.5 * h * np.ones_like(x_range), y_range)) / h
                Zy = np.zeros_like(Zx)
                Zy[1:-1] = (Z[2:] - Z[:-2]) / (y_range[2] - y_range[0]) # central difference
                Zy[0] = (Z[1] - Z[0]) / (y_range[1] - y_range[0])
                Zy[-1] = (Z[-1] - Z[-2]) / (y_range[1] - y_range[0])

                for m in range(M):
                    Ux[m, :] = (evaluate_CF_range(sum(self.hydro.alpha_solution[m][j] * self.hydro.time_basis.evaluation_function(t, j) for j in range(-imax, imax + 1)), self.hydro.mesh, x_range + 0.5 * h * np.ones_like(x_range), y_range) - 
                                evaluate_CF_range(sum(self.hydro.alpha_solution[m][j] * self.hydro.time_basis.evaluation_function(t, j) for j in range(-imax, imax + 1)), self.hydro.mesh, x_range - 0.5 * h * np.ones_like(x_range), y_range)) / h
                    Vy[m, 1:-1] = (V[m, 2:] - V[m, :-2]) / (y_range[2] - y_range[0]) # central difference
                    Vy[m, 0] = (V[m, 1] - V[m, 0]) / (y_range[1] - y_range[0])
                    Vy[m, -1] = (V[m, -1] - V[m, -2]) / (y_range[1] - y_range[0])

            elif y is not None:
                Zy = (evaluate_CF_range(self.zeta_timed(t), self.hydro.mesh, x_range, y_range + 0.5 * h * np.ones_like(y_range)) - evaluate_CF_range(self.zeta_timed(t), self.hydro.mesh, x_range, y_range - 0.5 * h * np.ones_like(y_range))) / h
                Zx = np.zeros_like(Zy)
                Zx[1:-1] = (Z[2:] - Z[:-2]) / (x_range[2] - x_range[0]) # central difference
                Zx[0] = (Z[1] - Z[0]) / (x_range[1] - x_range[0])
                Zx[-1] = (Z[-1] - Z[-2]) / (x_range[1] - x_range[0])

                for m in range(M):
                    Vy[m, :] = (evaluate_CF_range(sum(self.hydro.beta_solution[m][j] * self.hydro.time_basis.evaluation_function(t, j) for j in range(-imax, imax + 1)), self.hydro.mesh, x_range, y_range + 0.5 * h * np.ones_like(y_range)) - 
                                evaluate_CF_range(sum(self.hydro.beta_solution[m][j] * self.hydro.time_basis.evaluation_function(t, j) for j in range(-imax, imax + 1)), self.hydro.mesh, x_range, y_range - 0.5 * h * np.ones_like(y_range))) / h
                    Ux[m, 1:-1] = (U[m, 2:] - U[m, :-2]) / (x_range[2] - x_range[0]) # central difference
                    Ux[m, 0] = (U[m, 1] - U[m, 0]) / (x_range[1] - x_range[0])
                    Ux[m, -1] = (U[m, -1] - U[m, -2]) / (x_range[1] - x_range[0])

            for k, sig in enumerate(np.nditer(sig_arr)):
                if print_log:
                    print(f"    sig={sig}")

                Ufm = sum(U[m,:] * self.hydro.vertical_basis.evaluation_function(sig, m) for m in range(M))
                Vfm = sum(V[m,:] * self.hydro.vertical_basis.evaluation_function(sig, m) for m in range(M))
                UFm = sum(U[m,:] * self.hydro.vertical_basis.integrated_evaluation_function(sig, m) for m in range(M))
                VFm = sum(V[m,:] * self.hydro.vertical_basis.integrated_evaluation_function(sig, m) for m in range(M))
                UxFm = sum(Ux[m,:] * self.hydro.vertical_basis.integrated_evaluation_function(sig, m) for m in range(M))
                VyFm = sum(Vy[m,:] * self.hydro.vertical_basis.integrated_evaluation_function(sig, m) for m in range(M))
                
                W[i, k, :] = (Zx * (1+sig) + sig * Hx) * Ufm / L + (Zy * (1+sig) + sig * Hy) * Vfm / B - (H + Z) * UxFm / L - (H + Z) * VyFm / B - (Hx + Zx) * UFm / L - (Hy + Zy) * VFm / B

        return W
                




    def plot_TWA_cross_section_circulation(self, x, stride: int, num_horizontal_points: int = 500, num_vertical_points: int = 500, figsize=(7,4), spacing='equal', alpha='equal', save=None, title='Residual currents in the cross-section',
                                           miny=-np.inf, maxy=np.inf, minz=-np.inf, maxz=np.inf, plot_u=True):
        """Variables are evaluated always on the entire z-range, so num_vertical_points refers to that. In the specific z-range you're plotting, there may be fewer vertical points."""
        p1 = np.array([x, max(-0.5, miny)])
        p2 = np.array([x, min(0.5, maxy)])
        width = np.linalg.norm(p1-p2, 2)
        # s_range = np.linspace(max(-0.5, miny), min(0.5, maxy), num_horizontal_points)
        x_range = np.linspace(p1[0], p2[0], num_horizontal_points)
        y_range = np.linspace(p1[1], p2[1], num_horizontal_points)

        H = self.hydro.spatial_parameters['H']

        depth = evaluate_CF_range(H, self.hydro.mesh, x_range, y_range)

        s_grid = np.tile(y_range, (num_vertical_points, 1))
        z_grid = np.array([np.linspace(max(-depth[i], minz), min(0, maxz), num_vertical_points) for i in range(num_horizontal_points)]).T

        if plot_u:
            Q = evaluate_vertical_structure_at_cross_section(self.hydro.mesh, lambda sig: self.u_TWA(sig), p1, p2, num_horizontal_points, num_vertical_points)
        V = evaluate_vertical_structure_at_cross_section(self.hydro.mesh, lambda sig: self.v_TWA(sig), p1, p2, num_horizontal_points, num_vertical_points)
        # W = evaluate_vertical_structure_at_cross_section(self.hydro.mesh, lambda sig: self.w_TWA(sig), p1, p2, num_horizontal_points, num_vertical_points)
    
        sig_range = np.linspace(-1, 0, num_vertical_points)

        # Compute TWA w in cross-section
        t_arr = np.linspace(0, 1, 101)
        W = self.get_w_in_cross_section(t_arr, sig_range, x=x, num_horizontal_points=num_horizontal_points, print_log=False)
        avg_depth = evaluate_CF_range(H + 0.5*np.sqrt(2)*self.hydro.gamma_solution[0], self.hydro.mesh, x_range, y_range)
        depth_over_time = np.zeros((101, sig_range.shape[0], avg_depth.shape[0]))
        for i, t in enumerate(np.nditer(t_arr)):
            depth_over_time[i, 0, :] = evaluate_CF_range(self.zeta_timed(t) + H, self.hydro.mesh, x_range, y_range)

            for j in range(1, sig_range.shape[0]):
                depth_over_time[i, j, :] = depth_over_time[i, 0, :]

        W_TWA = (t_arr[1] - t_arr[0]) * np.sum(W * depth_over_time, axis=0) / avg_depth

        num_arrows_z = sig_range[::stride].shape[0]
        zquiv_grid = np.array([np.linspace(max(-np.amax(depth), minz), 0, num_arrows_z) for i in range(num_horizontal_points // stride + 1)]).T
        squiv_grid = s_grid[::stride, ::stride]

        Vquiv = np.zeros_like(zquiv_grid)
        Wquiv = np.zeros_like(zquiv_grid)
        mask = np.zeros_like(zquiv_grid)

        if plot_u:
            maxamp = np.amax(np.abs(Q))
        
        for y_index in range(num_horizontal_points // stride + (num_horizontal_points % stride != 0)):
            local_depth = evaluate_CF_point(H, self.hydro.mesh, x, squiv_grid[0, y_index])
            for z_index in range(num_vertical_points // stride):
                if zquiv_grid[z_index, y_index] > -local_depth:
                    mask[z_index, y_index] = 1
                    sig_value = zquiv_grid[z_index, y_index] / local_depth
                    corresponding_sig_index = np.argmin(np.absolute(sig_range - np.ones_like(sig_range) * sig_value))
                    if sig_value >= sig_range[corresponding_sig_index]: # interpolation between sig_range[corresponding_sig_index] and sig_range[corresponding_sig_index + 1]
                        Vquiv[z_index, y_index] = V[corresponding_sig_index, y_index * stride] * (sig_range[corresponding_sig_index + 1] - sig_value) + V[corresponding_sig_index + 1, y_index * stride] * (sig_value - sig_range[corresponding_sig_index])
                        Vquiv[z_index, y_index] /= sig_range[corresponding_sig_index + 1] - sig_range[corresponding_sig_index]
                        Wquiv[z_index, y_index] = W_TWA[corresponding_sig_index, y_index * stride] * (sig_range[corresponding_sig_index + 1] - sig_value) + W_TWA[corresponding_sig_index + 1, y_index * stride] * (sig_value - sig_range[corresponding_sig_index])
                        Wquiv[z_index, y_index] /= sig_range[corresponding_sig_index + 1] - sig_range[corresponding_sig_index]
                    else:
                        Vquiv[z_index, y_index] = V[corresponding_sig_index - 1, y_index * stride] * (sig_range[corresponding_sig_index] - sig_value) + V[corresponding_sig_index, y_index * stride] * (sig_value - sig_range[corresponding_sig_index - 1])
                        Vquiv[z_index, y_index] /= sig_range[corresponding_sig_index] - sig_range[corresponding_sig_index - 1]
                        Wquiv[z_index, y_index] = W_TWA[corresponding_sig_index - 1, y_index * stride] * (sig_range[corresponding_sig_index] - sig_value) + W_TWA[corresponding_sig_index, y_index * stride] * (sig_value - sig_range[corresponding_sig_index - 1])
                        Wquiv[z_index, y_index] /= sig_range[corresponding_sig_index] - sig_range[corresponding_sig_index - 1]
        
        
        fig_crosssection, ax_crosssection = plt.subplots(figsize=figsize)
        if plot_u:
            color_crosssection = ax_crosssection.pcolormesh(s_grid, z_grid, Q, vmin=-maxamp, vmax=maxamp, cmap='RdBu')
            cbar_crosssection = plt.colorbar(color_crosssection)
            cbar_crosssection.ax.set_ylabel('Along-channel velocity [m/s]')

        ax_crosssection.plot(y_range, -depth, linewidth=1, color='k', zorder=3)
        ax_crosssection.fill_between(y_range, -np.amax(depth), -depth, color='silver')
        
        if spacing == 'sigma':
            visual_norms = np.sqrt((V[::stride,::stride] / (width*self.hydro.geometric_information['y_scaling']))**2 + (W_TWA[::stride,::stride] / np.amax(depth))**2) # y-dimension in km
            physical_norms = np.sqrt((V[::stride,::stride])**2 + (W_TWA[::stride,::stride])**2)
            if alpha == 'variable':
                quiv = ax_crosssection.quiver(s_grid[::stride,::stride], z_grid[::stride,::stride], (V[::stride,::stride] / (width*self.hydro.geometric_information['y_scaling'])) / visual_norms, (W_TWA[::stride,::stride] / np.amax(depth)) / visual_norms, color='k', alpha=physical_norms/np.amax(physical_norms), pivot='mid', headlength=3, headaxislength=3)
            elif alpha == 'equal':
                quiv = ax_crosssection.quiver(s_grid[::stride,::stride], z_grid[::stride,::stride], (V[::stride,::stride] / (width*self.hydro.geometric_information['y_scaling'])) / visual_norms, (W_TWA[::stride,::stride] / np.amax(depth)) / visual_norms, color='k', pivot='mid', headlength=3, headaxislength=3)
        elif spacing == 'equal':
            visual_norms = np.sqrt((Vquiv / (width*self.hydro.geometric_information['y_scaling']))**2 + (Wquiv / np.amax(depth))**2) # y-dimension in km
            physical_norms = np.sqrt((Vquiv)**2 + (Wquiv)**2)
            if alpha == 'variable':
                quiv = ax_crosssection.quiver(squiv_grid, zquiv_grid, (Vquiv / (width*self.hydro.geometric_information['y_scaling'])) / visual_norms, (Wquiv / np.amax(depth)) / visual_norms, color='k', alpha=physical_norms/np.amax(physical_norms), pivot='mid', headlength=3, headaxislength = 3)
            elif alpha == 'equal':
                quiv = ax_crosssection.quiver(squiv_grid, zquiv_grid, (Vquiv / (width*self.hydro.geometric_information['y_scaling'])) / visual_norms, (Wquiv / np.amax(depth)) / visual_norms, color='k', pivot='mid', headlength=3, headaxislength = 3)


        xticks = list(np.linspace(max(-0.5, miny), min(0.5, maxy), 10))
        ax_crosssection.set_xticks(xticks)
        ticklabels = list(np.round(np.linspace(max(-0.5, miny), min(0.5, maxy), 10), 3) * self.hydro.geometric_information['y_scaling'] / 1000)
        ticklabels_string = [str(np.round(tick, 3))[:4] if tick >= 0 else str(np.round(tick, 3))[:5] for tick in ticklabels]
        ax_crosssection.set_xticklabels(ticklabels_string)

        ax_crosssection.set_xlabel('Distance along cross-section [km]')
        ax_crosssection.set_ylabel(r'-Thickness-weighted depth ($H\varsigma$) [m]')

        ax_crosssection.set_ylim((max(minz, -np.amax(depth)), min(0, maxz)))

        ax_crosssection.set_title(f'{title}\nMaximum lateral velocity = {np.round(np.amax(physical_norms),4)} m/s')

        if save is not None:
            fig_crosssection.savefig(save)


    def plot_TWA_2DV_circulation(self, y, stride: int, num_horizontal_points: int = 500, num_vertical_points: int = 500, figsize=(7,4), spacing='equal', alpha='equal', save=None, title='Residual currents in the cross-section',
                                    minx=-np.inf, maxx=np.inf, minz=-np.inf, maxz=np.inf, plot_v=True):
        """Variables are evaluated always on the entire z-range, so num_vertical_points refers to that. In the specific z-range you're plotting, there may be fewer vertical points."""
        p1 = np.array([max(0, minx), y])
        p2 = np.array([min(maxx, 1), y])
        width = np.linalg.norm(p1-p2, 2)
        # s_range = np.linspace(max(-0.5, miny), min(0.5, maxy), num_horizontal_points)
        x_range = np.linspace(p1[0], p2[0], num_horizontal_points)
        y_range = np.linspace(p1[1], p2[1], num_horizontal_points)

        H = self.hydro.spatial_parameters['H']

        depth = evaluate_CF_range(H, self.hydro.mesh, x_range, y_range)

        s_grid = np.tile(x_range, (num_vertical_points, 1))
        z_grid = np.array([np.linspace(max(-depth[i], minz), min(0, maxz), num_vertical_points) for i in range(num_horizontal_points)]).T

        if plot_v:
            Q = evaluate_vertical_structure_at_cross_section(self.hydro.mesh, lambda sig: self.v_TWA(sig), p1, p2, num_horizontal_points, num_vertical_points)
        U = evaluate_vertical_structure_at_cross_section(self.hydro.mesh, lambda sig: self.u_TWA(sig), p1, p2, num_horizontal_points, num_vertical_points)
        # W = evaluate_vertical_structure_at_cross_section(self.hydro.mesh, lambda sig: self.w_TWA(sig), p1, p2, num_horizontal_points, num_vertical_points)
    
        sig_range = np.linspace(-1, 0, num_vertical_points)

        # Compute TWA w in cross-section
        t_arr = np.linspace(0, 1, 101)
        W = self.get_w_in_cross_section(t_arr, sig_range, y=y, num_horizontal_points=num_horizontal_points, print_log=False)
        avg_depth = evaluate_CF_range(H + 0.5*np.sqrt(2)*self.hydro.gamma_solution[0], self.hydro.mesh, x_range, y_range)
        depth_over_time = np.zeros((101, sig_range.shape[0], avg_depth.shape[0]))
        for i, t in enumerate(np.nditer(t_arr)):
            depth_over_time[i, 0, :] = evaluate_CF_range(self.zeta_timed(t) + H, self.hydro.mesh, x_range, y_range)

            for j in range(1, sig_range.shape[0]):
                depth_over_time[i, j, :] = depth_over_time[i, 0, :]

        W_TWA = (t_arr[1] - t_arr[0]) * np.sum(W * depth_over_time, axis=0) / avg_depth

        num_arrows_z = sig_range[::stride].shape[0]
        zquiv_grid = np.array([np.linspace(max(-np.amax(depth), minz), 0, num_arrows_z) for i in range(num_horizontal_points // stride + 1)]).T
        squiv_grid = s_grid[::stride, ::stride]

        Uquiv = np.zeros_like(zquiv_grid)
        Wquiv = np.zeros_like(zquiv_grid)
        mask = np.zeros_like(zquiv_grid)

        if plot_v:
            maxamp = np.amax(np.abs(Q))
        
        for x_index in range(num_horizontal_points // stride + (num_horizontal_points % stride != 0)):
            local_depth = evaluate_CF_point(H, self.hydro.mesh, squiv_grid[0, x_index], y)
            for z_index in range(num_vertical_points // stride):
                if zquiv_grid[z_index, x_index] > -local_depth:
                    mask[z_index, x_index] = 1
                    sig_value = zquiv_grid[z_index, x_index] / local_depth
                    corresponding_sig_index = np.argmin(np.absolute(sig_range - np.ones_like(sig_range) * sig_value))
                    if sig_value >= sig_range[corresponding_sig_index]: # interpolation between sig_range[corresponding_sig_index] and sig_range[corresponding_sig_index + 1]
                        Uquiv[z_index, x_index] = U[corresponding_sig_index, x_index * stride] * (sig_range[corresponding_sig_index + 1] - sig_value) + U[corresponding_sig_index + 1, x_index * stride] * (sig_value - sig_range[corresponding_sig_index])
                        Uquiv[z_index, x_index] /= sig_range[corresponding_sig_index + 1] - sig_range[corresponding_sig_index]
                        Wquiv[z_index, x_index] = W_TWA[corresponding_sig_index, x_index * stride] * (sig_range[corresponding_sig_index + 1] - sig_value) + W_TWA[corresponding_sig_index + 1, x_index * stride] * (sig_value - sig_range[corresponding_sig_index])
                        Wquiv[z_index, x_index] /= sig_range[corresponding_sig_index + 1] - sig_range[corresponding_sig_index]
                    else:
                        Uquiv[z_index, x_index] = U[corresponding_sig_index - 1, x_index * stride] * (sig_range[corresponding_sig_index] - sig_value) + U[corresponding_sig_index, x_index * stride] * (sig_value - sig_range[corresponding_sig_index - 1])
                        Uquiv[z_index, x_index] /= sig_range[corresponding_sig_index] - sig_range[corresponding_sig_index - 1]
                        Wquiv[z_index, x_index] = W_TWA[corresponding_sig_index - 1, x_index * stride] * (sig_range[corresponding_sig_index] - sig_value) + W_TWA[corresponding_sig_index, x_index * stride] * (sig_value - sig_range[corresponding_sig_index - 1])
                        Wquiv[z_index, x_index] /= sig_range[corresponding_sig_index] - sig_range[corresponding_sig_index - 1]
        
        
        fig_crosssection, ax_crosssection = plt.subplots(figsize=figsize)
        if plot_v:
            color_crosssection = ax_crosssection.pcolormesh(s_grid, z_grid, Q, vmin=-maxamp, vmax=maxamp, cmap='RdBu')
            cbar_crosssection = plt.colorbar(color_crosssection)
            cbar_crosssection.ax.set_ylabel('Cross-channel velocity [m/s]')

        ax_crosssection.plot(x_range, -depth, linewidth=1, color='k', zorder=3)
        ax_crosssection.fill_between(x_range, -np.amax(depth), -depth, color='silver')
        
        if spacing == 'sigma':
            visual_norms = np.sqrt((U[::stride,::stride] / (width*self.hydro.geometric_information['x_scaling']))**2 + (W_TWA[::stride,::stride] / np.amax(depth))**2) # y-dimension in km
            physical_norms = np.sqrt((U[::stride,::stride])**2 + (W_TWA[::stride,::stride])**2)
            if alpha == 'variable':
                quiv = ax_crosssection.quiver(s_grid[::stride,::stride], z_grid[::stride,::stride], (U[::stride,::stride] / (width*self.hydro.geometric_information['x_scaling'])) / visual_norms, (W_TWA[::stride,::stride] / np.amax(depth)) / visual_norms, color='k', alpha=physical_norms/np.amax(physical_norms), pivot='mid', headlength=3, headaxislength=3)
            elif alpha == 'equal':
                quiv = ax_crosssection.quiver(s_grid[::stride,::stride], z_grid[::stride,::stride], (U[::stride,::stride] / (width*self.hydro.geometric_information['x_scaling'])) / visual_norms, (W_TWA[::stride,::stride] / np.amax(depth)) / visual_norms, color='k', pivot='mid', headlength=3, headaxislength=3)
        elif spacing == 'equal':
            visual_norms = np.sqrt((Uquiv / (width*self.hydro.geometric_information['x_scaling']))**2 + (Wquiv / np.amax(depth))**2) # y-dimension in km
            physical_norms = np.sqrt((Uquiv)**2 + (Wquiv)**2)
            if alpha == 'variable':
                quiv = ax_crosssection.quiver(squiv_grid, zquiv_grid, (Uquiv / (width*self.hydro.geometric_information['x_scaling'])) / visual_norms, (Wquiv / np.amax(depth)) / visual_norms, color='k', alpha=physical_norms/np.amax(physical_norms), pivot='mid', headlength=3, headaxislength = 3)
            elif alpha == 'equal':
                quiv = ax_crosssection.quiver(squiv_grid, zquiv_grid, (Uquiv / (width*self.hydro.geometric_information['x_scaling'])) / visual_norms, (Wquiv / np.amax(depth)) / visual_norms, color='k', pivot='mid', headlength=3, headaxislength = 3)


        xticks = list(np.linspace(max(0, minx), min(1, maxx), 10))
        ax_crosssection.set_xticks(xticks)
        ticklabels = list(np.round(np.linspace(max(0, minx), min(1, maxx), 10), 3) * self.hydro.geometric_information['x_scaling'] / 1000)
        ticklabels_string = [str(np.round(tick, 3))[:4] if tick >= 0 else str(np.round(tick, 3))[:5] for tick in ticklabels]
        ax_crosssection.set_xticklabels(ticklabels_string)

        ax_crosssection.set_xlabel('Distance along cross-section [km]')
        ax_crosssection.set_ylabel(r'-Thickness-weighted depth ($H\varsigma$) [m]')

        ax_crosssection.set_ylim((max(minz, -np.amax(depth)), min(0, maxz)))

        ax_crosssection.set_title(f'{title}\nMaximum lateral velocity = {np.round(np.amax(physical_norms),4)} m/s')

        if save is not None:
            fig_crosssection.savefig(save)


    def animate_cross_section_circulation(self, x, stride: int, savename:str, num_frames: int=64, num_horizontal_points: int = 500, num_vertical_points: int = 500, constituent=1, flowrange: tuple=None, figsize=(7,4), spacing='equal', alpha='equal'):
        p1 = np.array([x, -0.5])
        p2 = np.array([x, 0.5])
        width = np.linalg.norm(p1-p2, 2)
        s_range = np.linspace(-width/2, width/2, num_horizontal_points)
        x_range = np.linspace(p1[0], p2[0], num_horizontal_points)
        y_range = np.linspace(p1[1], p2[1], num_horizontal_points)

        H = self.hydro.spatial_parameters['H']

        depth = evaluate_CF_range(H, self.hydro.mesh, x_range, y_range)

        s_grid = np.tile(s_range, (num_vertical_points, 1))
        z_grid = np.array([np.linspace(-depth[i], 0, num_vertical_points) for i in range(num_horizontal_points)]).T

        Q = []
        V = []
        W = []

        # if constituent == 'all':
        #     Q = evaluate_vertical_structure_at_cross_section(self.hydro.mesh, lambda sig: self.u_timed(phase / self.hydro.constant_physical_parameters['sigma'], sig), p1, p2, num_horizontal_points, num_vertical_points)
        #     V = evaluate_vertical_structure_at_cross_section(self.hydro.mesh, lambda sig: self.v_timed(phase/ self.hydro.constant_physical_parameters['sigma'], sig), p1, p2, num_horizontal_points, num_vertical_points)
        #     W = evaluate_vertical_structure_at_cross_section(self.hydro.mesh, lambda sig: self.w_timed(phase/ self.hydro.constant_physical_parameters['sigma'], sig), p1, p2, num_horizontal_points, num_vertical_points)

        for i in range(num_frames):
            if constituent == 0:
                Q.append(evaluate_vertical_structure_at_cross_section(self.hydro.mesh, lambda sig: self.u(constituent, sig) * self.hydro.time_basis.evaluation_function(i/num_frames, constituent), p1, p2,
                                                                num_horizontal_points, num_vertical_points))
                V.append(evaluate_vertical_structure_at_cross_section(self.hydro.mesh, lambda sig: self.v(constituent, sig) * self.hydro.time_basis.evaluation_function(i/num_frames, constituent), p1, p2,
                                                                num_horizontal_points, num_vertical_points))
                W.append(self.get_w_cross_section(x, constituent=constituent, dx=0.01) * self.hydro.time_basis.evaluation_function(i/num_frames, constituent))
            else:
                Q.append(evaluate_vertical_structure_at_cross_section(self.hydro.mesh, lambda sig: self.u(constituent, sig) * self.hydro.time_basis.evaluation_function(i/num_frames, constituent) + \
                                                                self.u(-constituent, sig) * self.hydro.time_basis.evaluation_function(i/num_frames, -constituent), p1, p2,
                                                                num_horizontal_points, num_vertical_points))
                V.append(evaluate_vertical_structure_at_cross_section(self.hydro.mesh, lambda sig: self.v(constituent, sig) * self.hydro.time_basis.evaluation_function(i/num_frames, constituent) + \
                                                                self.v(-constituent, sig) * self.hydro.time_basis.evaluation_function(i/num_frames, -constituent), p1, p2,
                                                                num_horizontal_points, num_vertical_points))
                W.append(self.get_w_cross_section(x, constituent=constituent, dx=0.01) * self.hydro.time_basis.evaluation_function(i/num_frames, constituent) + self.get_w_cross_section(x, constituent=-constituent, dx=0.01) * self.hydro.time_basis.evaluation_function(i/num_frames, -constituent))

        Q = np.array(Q)
        V = np.array(V)
        W = np.array(W)

        if flowrange is None:
            maxamp = max(np.amax(Q.flatten()), -np.amin(Q.flatten()))
            minamp = -maxamp
        else:
            maxamp = flowrange[1]
            minamp = flowrange[0]

        sig_range = np.linspace(-1, 0, num_vertical_points)
        num_arrows_z = sig_range[::stride].shape[0]
        zquiv_grid = np.array([np.linspace(-np.amax(depth), 0, num_arrows_z) for i in range(num_horizontal_points // stride + 1)]).T
        squiv_grid = s_grid[::stride, ::stride]

        Vquiv = np.array([np.zeros_like(zquiv_grid) for _ in range(num_frames)])
        Wquiv = np.array([np.zeros_like(zquiv_grid) for _ in range(num_frames)])

        for i in range(num_frames):
            for y_index in range(num_horizontal_points // stride + 1):
                local_depth = evaluate_CF_point(H, self.hydro.mesh, x, squiv_grid[0, y_index])
                for z_index in range(num_vertical_points // stride):
                    if zquiv_grid[z_index, y_index] > -local_depth:
                        sig_value = zquiv_grid[z_index, y_index] / local_depth
                        corresponding_sig_index = np.argmin(np.absolute(sig_range - np.ones_like(sig_range) * sig_value))
                        if sig_value >= sig_range[corresponding_sig_index]: # interpolation between sig_range[corresponding_sig_index] and sig_range[corresponding_sig_index + 1]
                            Vquiv[i, z_index, y_index] = V[i, corresponding_sig_index, y_index * stride] * (sig_range[corresponding_sig_index + 1] - sig_value) + V[i, corresponding_sig_index + 1, y_index * stride] * (sig_value - sig_range[corresponding_sig_index])
                            Vquiv[i, z_index, y_index] /= sig_range[corresponding_sig_index + 1] - sig_range[corresponding_sig_index]
                            Wquiv[i, z_index, y_index] = W[i, corresponding_sig_index, y_index * stride] * (sig_range[corresponding_sig_index + 1] - sig_value) + W[i, corresponding_sig_index + 1, y_index * stride] * (sig_value - sig_range[corresponding_sig_index])
                            Wquiv[i, z_index, y_index] /= sig_range[corresponding_sig_index + 1] - sig_range[corresponding_sig_index]
                        else:
                            Vquiv[i, z_index, y_index] = V[i, corresponding_sig_index - 1, y_index * stride] * (sig_range[corresponding_sig_index] - sig_value) + V[i, corresponding_sig_index, y_index * stride] * (sig_value - sig_range[corresponding_sig_index - 1])
                            Vquiv[i, z_index, y_index] /= sig_range[corresponding_sig_index] - sig_range[corresponding_sig_index - 1]
                            Wquiv[i, z_index, y_index] = W[i, corresponding_sig_index - 1, y_index * stride] * (sig_range[corresponding_sig_index] - sig_value) + W[i, corresponding_sig_index, y_index * stride] * (sig_value - sig_range[corresponding_sig_index - 1])
                            Wquiv[i, z_index, y_index] /= sig_range[corresponding_sig_index] - sig_range[corresponding_sig_index - 1]
        
        if spacing == 'sigma':
            visual_norms = np.sqrt((V[:, ::stride,::stride] / (width*self.hydro.geometric_information['y_scaling']))**2 + (W[:, ::stride,::stride] / np.amax(depth))**2) # y-dimension in km
            physical_norms = np.sqrt((V[:, ::stride,::stride])**2 + (W[:, ::stride,::stride])**2)
        elif spacing == 'equal':
            visual_norms = np.sqrt((Vquiv / (width*self.hydro.geometric_information['y_scaling']))**2 + (Wquiv / np.amax(depth))**2) # y-dimension in km
            physical_norms = np.sqrt((Vquiv)**2 + (Wquiv)**2)

        for i in range(num_frames):
            fig_crosssection, ax_crosssection = plt.subplots(figsize=figsize)
            color_crosssection = ax_crosssection.pcolormesh(s_grid, z_grid, Q[i, :, :], vmin=minamp, vmax=maxamp, cmap='RdBu')
            cbar_crosssection = plt.colorbar(color_crosssection)
            cbar_crosssection.ax.set_ylabel('Longitudinal velocity [m/s]')

            ax_crosssection.plot(s_range, -depth, linewidth=1, color='k', zorder=3)
            ax_crosssection.fill_between(s_range, -np.amax(depth), -depth, color='silver')

            if alpha == 'equal':
                alpha_arr = np.ones_like(physical_norms[i,:,:])
            elif alpha == 'variable':
                alpha_arr = physical_norms[i, :, :] / np.amax(physical_norms.flatten())

            if spacing == 'sigma':        
                quiv = ax_crosssection.quiver(s_grid[::stride,::stride], z_grid[::stride,::stride], (V[i, ::stride,::stride] / (width*self.hydro.geometric_information['y_scaling'])) / visual_norms[i, :, :], (W[i, ::stride,::stride] / np.amax(depth)) / visual_norms[i, :, :], color='k', alpha=alpha_arr, pivot='mid')
            elif spacing == 'equal':
                quiv = ax_crosssection.quiver(squiv_grid, zquiv_grid, (Vquiv[i, :, :] / (width*self.hydro.geometric_information['y_scaling'])) / visual_norms[i, :, :], (Wquiv[i, :, :] / np.amax(depth)) / visual_norms[i, :, :], color='k', alpha=alpha_arr, pivot='mid')


            xticks = list(np.linspace(-0.5, 0.5, 5))
            ax_crosssection.set_xticks(xticks)
            ax_crosssection.set_xticklabels(list(np.linspace(-0.5, 0.5, 5) * self.hydro.geometric_information['y_scaling'] / 1000))

            ax_crosssection.set_xlabel('Distance along cross-section [km]')
            ax_crosssection.set_ylabel('-Depth [m]')

            phase_string = str(np.round(i/num_frames, 4))
            phase_string += '0' * (6 - len(phase_string))
            ax_crosssection.set_title(f"Lateral flow at " + r"$t=$" + phase_string + r"$\sigma^{-1}$ s" + f'\nMaximum magnitude of lateral flow: {np.round(np.amax(physical_norms.flatten()), 8)}')

            plt.tight_layout()
            if i == 0:
                os.makedirs(savename)
            fig_crosssection.savefig(f'{savename}/frame{i}')


    # def animate_surface_cross_section_circulation(self, x, stride: int, savename:str, num_frames: int=64, num_horizontal_points: int = 500, num_vertical_points: int = 500, flowrange: tuple=None, figsize=(7,4), spacing='equal', alpha='equal'):
    #     p1 = np.array([x, -0.5])
    #     p2 = np.array([x, 0.5])
    #     width = np.linalg.norm(p1-p2, 2)
    #     s_range = np.linspace(-width/2, width/2, num_horizontal_points)
    #     x_range = np.linspace(p1[0], p2[0], num_horizontal_points)
    #     y_range = np.linspace(p1[1], p2[1], num_horizontal_points)

    #     H = self.hydro.spatial_parameters['H']

    #     depth = evaluate_CF_range(H, self.hydro.mesh, x_range, y_range)


    #     s_grid = np.tile(s_range, (num_vertical_points, 1))
        

    #     Q = []
    #     V = []
    #     W = []

    #     # if constituent == 'all':
    #     #     Q = evaluate_vertical_structure_at_cross_section(self.hydro.mesh, lambda sig: self.u_timed(phase / self.hydro.constant_physical_parameters['sigma'], sig), p1, p2, num_horizontal_points, num_vertical_points)
    #     #     V = evaluate_vertical_structure_at_cross_section(self.hydro.mesh, lambda sig: self.v_timed(phase/ self.hydro.constant_physical_parameters['sigma'], sig), p1, p2, num_horizontal_points, num_vertical_points)
    #     #     W = evaluate_vertical_structure_at_cross_section(self.hydro.mesh, lambda sig: self.w_timed(phase/ self.hydro.constant_physical_parameters['sigma'], sig), p1, p2, num_horizontal_points, num_vertical_points)

    #     for i in range(num_frames):

    #         surf = evaluate_CF_range(self.zeta_timed(i/num_frames), self.hydro.mesh, x_range, y_range)
    #         z_grid = np.array([np.linspace(-depth[i], surf[i], num_vertical_points) for i in range(num_horizontal_points)]).T

    #         Q.append(evaluate_vertical_structure_at_cross_section(self.hydro.mesh, lambda sig: self.u_timed(i/num_frames, sig), p1, p2, num_horizontal_points, num_vertical_points))
    #         V.append(evaluate_vertical_structure_at_cross_section(self.hydro.mesh, lambda sig: self.v_timed(i/num_frames, sig), p1, p2, num_horizontal_points, num_vertical_points))


    #         if constituent == 0:
    #             Q.append(evaluate_vertical_structure_at_cross_section(self.hydro.mesh, lambda sig: self.u(constituent, sig) * self.hydro.time_basis.evaluation_function(i/num_frames, constituent), p1, p2,
    #                                                             num_horizontal_points, num_vertical_points))
    #             V.append(evaluate_vertical_structure_at_cross_section(self.hydro.mesh, lambda sig: self.v(constituent, sig) * self.hydro.time_basis.evaluation_function(i/num_frames, constituent), p1, p2,
    #                                                             num_horizontal_points, num_vertical_points))
    #             W.append(self.get_w_cross_section(x, constituent=constituent, dx=0.01) * self.hydro.time_basis.evaluation_function(i/num_frames, constituent))
    #         else:
    #             Q.append(evaluate_vertical_structure_at_cross_section(self.hydro.mesh, lambda sig: self.u(constituent, sig) * self.hydro.time_basis.evaluation_function(i/num_frames, constituent) + \
    #                                                             self.u(-constituent, sig) * self.hydro.time_basis.evaluation_function(i/num_frames, -constituent), p1, p2,
    #                                                             num_horizontal_points, num_vertical_points))
    #             V.append(evaluate_vertical_structure_at_cross_section(self.hydro.mesh, lambda sig: self.v(constituent, sig) * self.hydro.time_basis.evaluation_function(i/num_frames, constituent) + \
    #                                                             self.v(-constituent, sig) * self.hydro.time_basis.evaluation_function(i/num_frames, -constituent), p1, p2,
    #                                                             num_horizontal_points, num_vertical_points))
    #             W.append(self.get_w_cross_section(x, constituent=constituent, dx=0.01) * self.hydro.time_basis.evaluation_function(i/num_frames, constituent) + self.get_w_cross_section(x, constituent=-constituent, dx=0.01) * self.hydro.time_basis.evaluation_function(i/num_frames, -constituent))

    #     Q = np.array(Q)
    #     V = np.array(V)
    #     W = np.array(W)

    #     if flowrange is None:
    #         maxamp = max(np.amax(Q.flatten()), -np.amin(Q.flatten()))
    #         minamp = -maxamp
    #     else:
    #         maxamp = flowrange[1]
    #         minamp = flowrange[0]

    #     sig_range = np.linspace(-1, 0, num_vertical_points)
        
    #     squiv_grid = s_grid[::stride, ::stride]

    #     Vquiv = np.array([np.zeros_like(zquiv_grid) for _ in range(num_frames)])
    #     Wquiv = np.array([np.zeros_like(zquiv_grid) for _ in range(num_frames)])

    #     for i in range(num_frames):

    #         surf = evaluate_CF_range(self.zeta_timed(i/num_frames), self.hydro.mesh, x_range, y_range)

    #         num_arrows_z = sig_range[::stride].shape[0]
    #         zquiv_grid = np.array([np.linspace(-np.amax(depth), 0, num_arrows_z) for i in range(num_horizontal_points // stride + 1)]).T


    #         for y_index in range(num_horizontal_points // stride + 1):
    #             local_depth = evaluate_CF_point(H, self.hydro.mesh, x, squiv_grid[0, y_index])
    #             for z_index in range(num_vertical_points // stride):
    #                 if zquiv_grid[z_index, y_index] > -local_depth:
    #                     sig_value = zquiv_grid[z_index, y_index] / local_depth
    #                     corresponding_sig_index = np.argmin(np.absolute(sig_range - np.ones_like(sig_range) * sig_value))
    #                     if sig_value >= sig_range[corresponding_sig_index]: # interpolation between sig_range[corresponding_sig_index] and sig_range[corresponding_sig_index + 1]
    #                         Vquiv[i, z_index, y_index] = V[i, corresponding_sig_index, y_index * stride] * (sig_range[corresponding_sig_index + 1] - sig_value) + V[i, corresponding_sig_index + 1, y_index * stride] * (sig_value - sig_range[corresponding_sig_index])
    #                         Vquiv[i, z_index, y_index] /= sig_range[corresponding_sig_index + 1] - sig_range[corresponding_sig_index]
    #                         Wquiv[i, z_index, y_index] = W[i, corresponding_sig_index, y_index * stride] * (sig_range[corresponding_sig_index + 1] - sig_value) + W[i, corresponding_sig_index + 1, y_index * stride] * (sig_value - sig_range[corresponding_sig_index])
    #                         Wquiv[i, z_index, y_index] /= sig_range[corresponding_sig_index + 1] - sig_range[corresponding_sig_index]
    #                     else:
    #                         Vquiv[i, z_index, y_index] = V[i, corresponding_sig_index - 1, y_index * stride] * (sig_range[corresponding_sig_index] - sig_value) + V[i, corresponding_sig_index, y_index * stride] * (sig_value - sig_range[corresponding_sig_index - 1])
    #                         Vquiv[i, z_index, y_index] /= sig_range[corresponding_sig_index] - sig_range[corresponding_sig_index - 1]
    #                         Wquiv[i, z_index, y_index] = W[i, corresponding_sig_index - 1, y_index * stride] * (sig_range[corresponding_sig_index] - sig_value) + W[i, corresponding_sig_index, y_index * stride] * (sig_value - sig_range[corresponding_sig_index - 1])
    #                         Wquiv[i, z_index, y_index] /= sig_range[corresponding_sig_index] - sig_range[corresponding_sig_index - 1]
        
    #     if spacing == 'sigma':
    #         visual_norms = np.sqrt((V[:, ::stride,::stride] / (width*self.hydro.geometric_information['y_scaling']))**2 + (W[:, ::stride,::stride] / np.amax(depth))**2) # y-dimension in km
    #         physical_norms = np.sqrt((V[:, ::stride,::stride])**2 + (W[:, ::stride,::stride])**2)
    #     elif spacing == 'equal':
    #         visual_norms = np.sqrt((Vquiv / (width*self.hydro.geometric_information['y_scaling']))**2 + (Wquiv / np.amax(depth))**2) # y-dimension in km
    #         physical_norms = np.sqrt((Vquiv)**2 + (Wquiv)**2)

    #     for i in range(num_frames):
    #         fig_crosssection, ax_crosssection = plt.subplots(figsize=figsize)
    #         color_crosssection = ax_crosssection.pcolormesh(s_grid, z_grid, Q[i, :, :], vmin=minamp, vmax=maxamp, cmap='RdBu')
    #         cbar_crosssection = plt.colorbar(color_crosssection)
    #         cbar_crosssection.ax.set_ylabel('Longitudinal velocity [m/s]')

    #         ax_crosssection.plot(s_range, -depth, linewidth=1, color='k', zorder=3)
    #         ax_crosssection.fill_between(s_range, -np.amax(depth), -depth, color='silver')

    #         if alpha == 'equal':
    #             alpha_arr = np.ones_like(physical_norms[i,:,:])
    #         elif alpha == 'variable':
    #             alpha_arr = physical_norms[i, :, :] / np.amax(physical_norms.flatten())

    #         if spacing == 'sigma':        
    #             quiv = ax_crosssection.quiver(s_grid[::stride,::stride], z_grid[::stride,::stride], (V[i, ::stride,::stride] / (width*self.hydro.geometric_information['y_scaling'])) / visual_norms[i, :, :], (W[i, ::stride,::stride] / np.amax(depth)) / visual_norms[i, :, :], color='k', alpha=alpha_arr, pivot='mid')
    #         elif spacing == 'equal':
    #             quiv = ax_crosssection.quiver(squiv_grid, zquiv_grid, (Vquiv[i, :, :] / (width*self.hydro.geometric_information['y_scaling'])) / visual_norms[i, :, :], (Wquiv[i, :, :] / np.amax(depth)) / visual_norms[i, :, :], color='k', alpha=alpha_arr, pivot='mid')


    #         xticks = list(np.linspace(-0.5, 0.5, 5))
    #         ax_crosssection.set_xticks(xticks)
    #         ax_crosssection.set_xticklabels(list(np.linspace(-0.5, 0.5, 5) * self.hydro.geometric_information['y_scaling'] / 1000))

    #         ax_crosssection.set_xlabel('Distance along cross-section [km]')
    #         ax_crosssection.set_ylabel('-Depth [m]')

    #         phase_string = str(np.round(i/num_frames, 4))
    #         phase_string += '0' * (6 - len(phase_string))
    #         ax_crosssection.set_title(f"Lateral flow at " + r"$t=$" + phase_string + r"$\sigma^{-1}$ s" + f'\nMaximum magnitude of lateral flow: {np.round(np.amax(physical_norms.flatten()), 8)}')

    #         plt.tight_layout()
    #         if i == 0:
    #             os.makedirs(savename)
    #         fig_crosssection.savefig(f'{savename}/frame{i}')


    def plot_cross_section_residual_forcing_mechanisms(self, p1: np.ndarray, p2: np.ndarray, num_horizontal_points, num_vertical_points, figsize=(12,6), cmap='RdBu', savename=None, component='u', **kwargs):
        """Plots all of the different forcing mechanisms for along-channel residual currents, along with the total forcing and the resulting residual flow."""

        width = np.linalg.norm(p1-p2, 2)
        s_range = np.linspace(width/2, -width/2, num_horizontal_points)
        x_range = np.linspace(p1[0], p2[0], num_horizontal_points)
        y_range = np.linspace(p1[1], p2[1], num_horizontal_points)
        sigma_range = np.linspace(-1, 0, num_vertical_points)

        H = self.hydro.spatial_physical_parameters['H']
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

        H = self.hydro.spatial_physical_parameters['H']

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


    def exchange_rate(self, num_vertical_points=100, quantity='u0'):
        """Computes the integral of max(u_subtidal, 0) over the entire domain. The dimension of this quantity is m^4/s.
        
        Arguments:

        - num_vertical_points (int): number of sigma layers used to compute the integral
        - quantity (str): indicator what quantity to use ('u0': positive residual along-channel velocity,
                          'v0' positive residual cross-channel velocity, 'u1', semidiurnal along-channel
                          velocity amplitude, 'v1', semidiurnal cross-channel velocity amplitude
        
        """

        x_scaling = self.hydro.geometric_information['x_scaling']
        y_scaling = self.hydro.geometric_information['y_scaling']
        H = self.hydro.spatial_parameters['H']
        R = self.hydro.spatial_parameters['R']

        sigma_range = np.linspace(-1, 0, num_vertical_points)
        dsigma = sigma_range[1] - sigma_range[0]

        if quantity == 'u0':
            maxU0 = lambda sigma: ngsolve.IfPos(self.u(0, sigma), self.u(0, sigma), 0)
        elif quantity == 'v0':
            maxU0 = lambda sigma: ngsolve.IfPos(self.v(0, sigma), self.v(0, sigma), 0)
        elif quantity == 'u1':
            maxU0 = lambda sigma: self.u_abs(1, sigma)
        elif quantity == 'v1':
            maxU0 = lambda sigma: self.v_abs(1, sigma)

        depth_integrated_maxU0 = dsigma * sum([maxU0(sigma_range[i]) for i in range(num_vertical_points)]) * (H+R)
        return x_scaling * y_scaling * ngsolve.Integrate(depth_integrated_maxU0, self.hydro.display_mesh)


class HydroPlot(object):

    def __init__(self, hydro: Hydrodynamics, pp: PostProcessing, num_figures: tuple = (1, 1), figsize=(7,4)):
        self.fig, self.ax = plt.subplots(num_figures[0], num_figures[1], figsize=(figsize[0] * num_figures[1], figsize[1] * num_figures[0]))
        self.plot_counter = 0
        self.hydro = hydro
        self.num_figures = num_figures
        self.pp = pp


    def set_suptitle(self, title):
        self.fig.suptitle(title)


    def save(self, filename):
        self.fig.savefig(filename)


    def get_next_index(self):
        new_index = self.plot_counter
        ij = (new_index // self.num_figures[0], new_index % self.num_figures[1])
        return ij
    

    def show(self):
        self.fig.tight_layout()
        plt.show()

    def add_topview_plot(self, title, exclude_ramping_zone=True, colormap_quantity=None, cmap='RdBu', center_range=True, clabel='Color [unit]', contours=True, refinement_level=3, vectorfield_quantity=None, num_arrows:tuple=(0,0), arrow_color='k', length_indication='alpha', **kwargs):
        x_scaling = self.hydro.geometric_information['x_scaling']
        y_scaling = self.hydro.geometric_information['y_scaling']
        
        if self.num_figures == (1, 1):
            current_ax = self.ax
        elif self.num_figures[0] == 1 or self.num_figures[1] == 1:
            current_ax = self.ax[self.plot_counter]
        else:
            current_ax = self.ax[self.get_next_index()]

        if colormap_quantity is not None:
            if self.hydro.numerical_information['mesh_generation_method'] != 'structured_quads':

                if exclude_ramping_zone:
                    triangulation = get_triangulation(self.hydro.display_mesh.ngmesh)
                else:
                    triangulation = get_triangulation(self.hydro.mesh.ngmesh)

                refiner = tri.UniformTriRefiner(triangulation)
                refined_triangulation = refiner.refine_triangulation(subdiv=refinement_level)
                
                eval_gfu = evaluate_CF_range(colormap_quantity, self.hydro.mesh, refined_triangulation.x, refined_triangulation.y)

                if center_range:
                    maxamp = max(np.amax(eval_gfu), -np.amin(eval_gfu))

                if center_range:
                    colormesh = current_ax.tripcolor(refined_triangulation, eval_gfu, vmin=-maxamp, vmax=maxamp, cmap='RdBu', **kwargs)
                else:
                    colormesh = current_ax.tripcolor(refined_triangulation, eval_gfu, **kwargs)

                if contours:
                    try:
                        levels = np.linspace(np.min(eval_gfu), np.max(eval_gfu), 10*(3))
                        contour = current_ax.tricontour(refined_triangulation, eval_gfu, levels, colors=['k'] + ["0.4"] * 2, linewidths=[.5] * (3))
                        current_ax.clabel(contour, levels[0::3], inline=1, fontsize=10, fmt='%1.4f')
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
                    Q[:, i] = evaluate_CF_range(colormap_quantity, self.hydro.mesh, x, y[i] * np.ones_like(x))

                if center_range:
                    maxamp = np.amax(np.absolute(Q.flatten()))

                if center_range:
                    colormesh = current_ax.pcolormesh(X, Y, Q, vmin=-maxamp, vmax=maxamp, cmap=cmap, **kwargs)
                else:
                    colormesh = current_ax.pcolormesh(X, Y, Q, cmap=cmap, **kwargs)
                
                if contours:
                    num_levels = 10
                    subamplitude_lines = 2
                    try:
                        levels = np.linspace(np.min(Q.flatten()), np.max(Q.flatten()), num_levels*(subamplitude_lines+1))
                        contour = current_ax.contour(X, Y, Q, levels, colors=['k'] + ["0.4"] * subamplitude_lines, linewidths=[.5] * (1+subamplitude_lines))
                        current_ax.clabel(contour, levels[0::subamplitude_lines+1], inline=1, fontsize=10, fmt='%1.4f')
                    except ValueError:
                        print("Constant solution; plotting contour lines impossible")

            
            cbar = self.fig.colorbar(colormesh, ax = current_ax)
            cbar.ax.set_ylabel(clabel)
        
        if vectorfield_quantity is not None:
            if exclude_ramping_zone:
                xquiv = np.linspace(0, self.hydro.geometric_information['riverine_boundary_x'] / x_scaling, num_arrows[0]) # This is a temporary solution; we assume now that if a structured grid is used, then the (scaled) domain is a unit square
            else:
                xquiv = np.linspace(-(self.hydro.geometric_information['L_BL_sea']-self.hydro.geometric_information['L_R_sea']-self.hydro.geometric_information['L_RA_sea']) / x_scaling,
                                (self.hydro.geometric_information['riverine_boundary_x']+self.hydro.geometric_information['L_RA_river']+self.hydro.geometric_information['L_R_river'] + self.hydro.geometric_information['L_BL_river'])/x_scaling,
                                num_arrows[0])
            yquiv = np.linspace(-0.5, 0.5, num_arrows[1])
            X, Y = np.meshgrid(xquiv, yquiv, indexing='ij')
            Xquiv = np.zeros_like(X)
            Yquiv = np.zeros_like(Y)

            for i in range(Xquiv.shape[1]):
                Xquiv[:, i] = evaluate_CF_range(vectorfield_quantity[0], self.hydro.mesh, xquiv, yquiv[i] * np.ones_like(xquiv))
                Yquiv[:, i] = evaluate_CF_range(vectorfield_quantity[1], self.hydro.mesh, xquiv, yquiv[i] * np.ones_like(xquiv))

            visual_norms = np.sqrt((Xquiv/x_scaling)**2 + (Yquiv/y_scaling)**2)
            norms = np.sqrt(Xquiv**2 + Yquiv**2)

            if length_indication == 'alpha':
                arrows = current_ax.quiver(X, Y, (Xquiv/x_scaling)/visual_norms, (Yquiv/y_scaling)/visual_norms, color=arrow_color, pivot='mid', alpha=norms / np.amax(norms))
            elif length_indication == 'length':
                arrows = current_ax.quiver(X, Y, (Xquiv/x_scaling), (Yquiv/y_scaling), color=arrow_color, pivot='mid')
            elif length_indication == 'none':
                arrows = current_ax.quiver(X, Y, (Xquiv/x_scaling)/visual_norms, (Yquiv/y_scaling)/visual_norms, color=arrow_color, pivot='mid')


        if vectorfield_quantity is None:
            current_ax.set_title(title)
        else:
            current_ax.set_title(title + f'\nMaximum magnitude of arrows: {np.round(np.amax(norms), 8)}')


        if exclude_ramping_zone:
            x_ticks = list(np.linspace(0, 1, 11)) # Also a temporary solution; more domain types and compatibility with these functions will be added in the future
        else:
            x_ticks = list(np.linspace((-self.hydro.geometric_information['L_BL_sea']-self.hydro.geometric_information['L_R_sea']-self.hydro.geometric_information['L_RA_sea'])/x_scaling,
                                (self.hydro.geometric_information['riverine_boundary_x']+self.hydro.geometric_information['L_RA_river']+self.hydro.geometric_information['L_R_river'] + self.hydro.geometric_information['L_BL_river'])/x_scaling,
                                11))
        y_ticks = list(np.linspace(-0.5, 0.5, 5))
        current_ax.set_xticks(x_ticks)
        current_ax.set_yticks(y_ticks)
        current_ax.set_xticklabels(np.round(np.array(x_ticks) * x_scaling / 1e3, 1))
        current_ax.set_yticklabels(np.round(np.array(y_ticks) * y_scaling / 1e3, 1))

        current_ax.set_xlabel('x [km]')
        current_ax.set_ylabel('y [km]')
        self.plot_counter += 1


    def add_cross_section_plot(self, title, x, num_horizontal_points = 500, num_vertical_points = 500, colormap_quantity_function=None, clabel='Color [unit]', cmap='RdBu', center_range=True, contours=False, plot_circulation=False, stride=5, length_indication='alpha', arrow_color='black', spacing='equal', **kwargs):
        if self.num_figures == (1, 1):
            current_ax = self.ax
        elif self.num_figures[0] == 1 or self.num_figures[1] == 1:
            current_ax = self.ax[self.plot_counter]
        else:
            current_ax = self.ax[self.get_next_index()]

        x_scaling = self.hydro.geometric_information['x_scaling']
        y_scaling = self.hydro.geometric_information['y_scaling']
        
        p1 = np.array([x, -0.5])
        p2 = np.array([x, 0.5])
        x_range = np.linspace(p1[0], p2[0], num_horizontal_points)
        y_range = np.linspace(p1[1], p2[1], num_horizontal_points)

        if colormap_quantity_function is not None:

            H = self.hydro.spatial_parameters['H']

            depth = evaluate_CF_range(H, self.hydro.mesh, x_range, y_range)

            s_grid = np.tile(y_range, (num_vertical_points, 1))
            z_grid = np.array([np.linspace(-depth[i], 0, num_vertical_points) for i in range(num_horizontal_points)]).T

            if isinstance(colormap_quantity_function, np.ndarray):
                Q = colormap_quantity_function
            else:
                Q = evaluate_vertical_structure_at_cross_section(self.hydro.mesh, colormap_quantity_function, p1, p2, num_horizontal_points, num_vertical_points)

            if center_range:
                maxamp = max(np.amax(Q), -np.amin(Q))

            if center_range:
                color_crosssection = current_ax.pcolormesh(s_grid, z_grid, Q, vmin=-maxamp, vmax=maxamp, cmap=cmap, **kwargs)
            else:
                color_crosssection = current_ax.pcolormesh(s_grid, z_grid, Q, cmap=cmap, **kwargs)
            cbar_crosssection = plt.colorbar(color_crosssection, ax=current_ax)
            cbar_crosssection.ax.set_ylabel(clabel)

            if contours:
                num_levels = 10
                subamplitude_lines = 2

                levels = np.linspace(np.min(Q), np.max(Q), num_levels*(subamplitude_lines+1))
                contour = current_ax.contour(s_grid, z_grid, Q, levels, colors=['k'] + ["0.4"] * subamplitude_lines, linewidths=[.5] * (1+subamplitude_lines))
                current_ax.clabel(contour, levels[0::subamplitude_lines+1], inline=1, fontsize=10, fmt='%1.4f')
                
        if plot_circulation:
            V = evaluate_vertical_structure_at_cross_section(self.hydro.mesh, self.pp.v_TWA, p1, p2, num_horizontal_points, num_vertical_points)
            sig_range = np.linspace(-1, 0, num_vertical_points)

            # Compute TWA w in cross-section
            t_arr = np.linspace(0, 1, 101)
            W = self.pp.get_w_in_cross_section(t_arr, sig_range, x=x, num_horizontal_points=num_horizontal_points, print_log=False)
            avg_depth = evaluate_CF_range(H + 0.5*np.sqrt(2)*self.hydro.gamma_solution[0], self.hydro.mesh, x_range, y_range)
            depth_over_time = np.zeros((101, sig_range.shape[0], avg_depth.shape[0]))
            for i, t in enumerate(np.nditer(t_arr)):
                depth_over_time[i, 0, :] = evaluate_CF_range(self.pp.zeta_timed(t) + H, self.hydro.mesh, x_range, y_range)

                for j in range(1, sig_range.shape[0]):
                    depth_over_time[i, j, :] = depth_over_time[i, 0, :]

            W_TWA = (t_arr[1] - t_arr[0]) * np.sum(W * depth_over_time, axis=0) / avg_depth

            num_arrows_z = sig_range[::stride].shape[0]
            zquiv_grid = np.array([np.linspace(-np.amax(depth), 0, num_arrows_z) for i in range(num_horizontal_points // stride + 1)]).T
            squiv_grid = s_grid[::stride, ::stride]

            Vquiv = np.zeros_like(zquiv_grid)
            Wquiv = np.zeros_like(zquiv_grid)
            mask = np.zeros_like(zquiv_grid)

            for y_index in range(num_horizontal_points // stride + (num_horizontal_points % stride != 0)):
                local_depth = evaluate_CF_point(H, self.hydro.mesh, x, squiv_grid[0, y_index])
                for z_index in range(num_vertical_points // stride):
                    if zquiv_grid[z_index, y_index] > -local_depth:
                        mask[z_index, y_index] = 1
                        sig_value = zquiv_grid[z_index, y_index] / local_depth
                        corresponding_sig_index = np.argmin(np.absolute(sig_range - np.ones_like(sig_range) * sig_value))
                        if sig_value >= sig_range[corresponding_sig_index]: # interpolation between sig_range[corresponding_sig_index] and sig_range[corresponding_sig_index + 1]
                            Vquiv[z_index, y_index] = V[corresponding_sig_index, y_index * stride] * (sig_range[corresponding_sig_index + 1] - sig_value) + V[corresponding_sig_index + 1, y_index * stride] * (sig_value - sig_range[corresponding_sig_index])
                            Vquiv[z_index, y_index] /= sig_range[corresponding_sig_index + 1] - sig_range[corresponding_sig_index]
                            Wquiv[z_index, y_index] = W_TWA[corresponding_sig_index, y_index * stride] * (sig_range[corresponding_sig_index + 1] - sig_value) + W_TWA[corresponding_sig_index + 1, y_index * stride] * (sig_value - sig_range[corresponding_sig_index])
                            Wquiv[z_index, y_index] /= sig_range[corresponding_sig_index + 1] - sig_range[corresponding_sig_index]
                        else:
                            Vquiv[z_index, y_index] = V[corresponding_sig_index - 1, y_index * stride] * (sig_range[corresponding_sig_index] - sig_value) + V[corresponding_sig_index, y_index * stride] * (sig_value - sig_range[corresponding_sig_index - 1])
                            Vquiv[z_index, y_index] /= sig_range[corresponding_sig_index] - sig_range[corresponding_sig_index - 1]
                            Wquiv[z_index, y_index] = W_TWA[corresponding_sig_index - 1, y_index * stride] * (sig_range[corresponding_sig_index] - sig_value) + W_TWA[corresponding_sig_index, y_index * stride] * (sig_value - sig_range[corresponding_sig_index - 1])
                            Wquiv[z_index, y_index] /= sig_range[corresponding_sig_index] - sig_range[corresponding_sig_index - 1]

            if spacing == 'sigma':
                visual_norms = np.sqrt((V[::stride,::stride] / (y_scaling))**2 + (W_TWA[::stride,::stride] / np.amax(depth))**2) # y-dimension in km
                physical_norms = np.sqrt((V[::stride,::stride])**2 + (W_TWA[::stride,::stride])**2)
                if length_indication == 'alpha':
                    quiv = current_ax.quiver(s_grid[::stride,::stride], z_grid[::stride,::stride], (V[::stride,::stride] / (y_scaling)) / visual_norms, (W_TWA[::stride,::stride] / np.amax(depth)) / visual_norms, color=arrow_color, alpha=physical_norms/np.amax(physical_norms), pivot='mid', headlength=3, headaxislength=3)
                elif length_indication == 'none':
                    quiv = current_ax.quiver(s_grid[::stride,::stride], z_grid[::stride,::stride], (V[::stride,::stride] / (y_scaling)) / visual_norms, (W_TWA[::stride,::stride] / np.amax(depth)) / visual_norms, color=arrow_color, pivot='mid', headlength=3, headaxislength=3)
                elif length_indication == 'length':
                    quiv = current_ax.quiver(s_grid[::stride,::stride], z_grid[::stride,::stride], V[::stride,::stride] / (y_scaling), W_TWA[::stride,::stride] / np.amax(depth), color=arrow_color, pivot='mid', headlength=3, headaxislength=3)
            elif spacing == 'equal':
                visual_norms = np.sqrt((Vquiv / (y_scaling))**2 + (Wquiv / np.amax(depth))**2) # y-dimension in km
                physical_norms = np.sqrt((Vquiv)**2 + (Wquiv)**2)
                if length_indication == 'alpha':
                    quiv = current_ax.quiver(squiv_grid, zquiv_grid, (Vquiv / (y_scaling)) / visual_norms, (Wquiv / np.amax(depth)) / visual_norms, color=arrow_color, alpha=physical_norms/np.amax(physical_norms), pivot='mid', headlength=3, headaxislength = 3)
                elif length_indication == 'none':
                    quiv = current_ax.quiver(squiv_grid, zquiv_grid, (Vquiv / (y_scaling)) / visual_norms, (Wquiv / np.amax(depth)) / visual_norms, color=arrow_color, pivot='mid', headlength=3, headaxislength = 3)
                elif length_indication == 'length':
                    quiv = current_ax.quiver(squiv_grid, zquiv_grid, Vquiv / (y_scaling), Wquiv / np.amax(depth), color=arrow_color, pivot='mid', headlength=3, headaxislength=3)


        current_ax.plot(y_range, -depth, linewidth=1, color='k', zorder=3)
        current_ax.fill_between(y_range, -np.amax(depth), -depth, color='silver')

        if plot_circulation:
            current_ax.set_title(f'{title}\nMaximum lateral velocity = {np.round(np.amax(physical_norms),4)} m/s')
        else:
            current_ax.set_title(title)
        xticks = list(np.linspace(-0.5,0.5, 10))
        current_ax.set_xticks(xticks)
        ticklabels = list(np.round(np.linspace(-0.5, 0.5, 10), 3) * y_scaling / 1000)
        ticklabels_string = [str(np.round(tick, 3))[:4] if tick >= 0 else str(np.round(tick, 3))[:5] for tick in ticklabels]
        current_ax.set_xticklabels(ticklabels_string)

        current_ax.set_xlabel('y [km]')
        current_ax.set_ylabel('-Depth [m]')
        self.plot_counter += 1


    def add_2DV_plot(self, title, y, num_horizontal_points = 500, num_vertical_points = 500, colormap_quantity_function=None, clabel='Color [unit]', cmap='RdBu', center_range=True, contours=False, plot_circulation=False, stride=5, length_indication='alpha', arrow_color='black', spacing='equal', **kwargs):
        if self.num_figures == (1, 1):
            current_ax = self.ax
        elif self.num_figures[0] == 1 or self.num_figures[1] == 1:
            current_ax = self.ax[self.plot_counter]
        else:
            current_ax = self.ax[self.get_next_index()]

        x_scaling = self.hydro.geometric_information['x_scaling']
        y_scaling = self.hydro.geometric_information['y_scaling']
        
        p1 = np.array([0, y])
        p2 = np.array([1, y])
        x_range = np.linspace(p1[0], p2[0], num_horizontal_points)
        y_range = np.linspace(p1[1], p2[1], num_horizontal_points)

        H = self.hydro.spatial_parameters['H']

        depth = evaluate_CF_range(H, self.hydro.mesh, x_range, y_range)

        s_grid = np.tile(x_range, (num_vertical_points, 1))
        z_grid = np.array([np.linspace(-depth[i], 0, num_vertical_points) for i in range(num_horizontal_points)]).T

        if colormap_quantity_function is not None:
            if isinstance(colormap_quantity_function, np.ndarray):
                Q = colormap_quantity_function
            else:
                Q = evaluate_vertical_structure_at_cross_section(self.hydro.mesh, colormap_quantity_function, p1, p2, num_horizontal_points, num_vertical_points)

            if center_range:
                maxamp = max(np.amax(Q), -np.amin(Q))

            if center_range:
                color_crosssection = current_ax.pcolormesh(s_grid, z_grid, Q, vmin=-maxamp, vmax=maxamp, cmap=cmap, **kwargs)
            else:
                color_crosssection = current_ax.pcolormesh(s_grid, z_grid, Q, cmap=cmap, **kwargs)
            cbar_crosssection = plt.colorbar(color_crosssection, ax=current_ax)
            cbar_crosssection.ax.set_ylabel(clabel)

            if contours:
                num_levels = 10
                subamplitude_lines = 2

                levels = np.linspace(np.min(Q), np.max(Q), num_levels*(subamplitude_lines+1))
                contour = current_ax.contour(s_grid, z_grid, Q, levels, colors=['k'] + ["0.4"] * subamplitude_lines, linewidths=[.5] * (1+subamplitude_lines))
                current_ax.clabel(contour, levels[0::subamplitude_lines+1], inline=1, fontsize=10, fmt='%1.4f')
                
        if plot_circulation:
            U = evaluate_vertical_structure_at_cross_section(self.hydro.mesh, self.pp.u_TWA, p1, p2, num_horizontal_points, num_vertical_points)
            sig_range = np.linspace(-1, 0, num_vertical_points)

            # Compute TWA w in cross-section
            t_arr = np.linspace(0, 1, 101)
            W = self.pp.get_w_in_cross_section(t_arr, sig_range, y=y, num_horizontal_points=num_horizontal_points, print_log=False)
            avg_depth = evaluate_CF_range(H + 0.5*np.sqrt(2)*self.hydro.gamma_solution[0], self.hydro.mesh, x_range, y_range)
            depth_over_time = np.zeros((101, sig_range.shape[0], avg_depth.shape[0]))
            for i, t in enumerate(np.nditer(t_arr)):
                depth_over_time[i, 0, :] = evaluate_CF_range(self.pp.zeta_timed(t) + H, self.hydro.mesh, x_range, y_range)

                for j in range(1, sig_range.shape[0]):
                    depth_over_time[i, j, :] = depth_over_time[i, 0, :]

            W_TWA = (t_arr[1] - t_arr[0]) * np.sum(W * depth_over_time, axis=0) / avg_depth

            num_arrows_z = sig_range[::stride].shape[0]
            zquiv_grid = np.array([np.linspace(-np.amax(depth), 0, num_arrows_z) for i in range(num_horizontal_points // stride + 1)]).T
            squiv_grid = s_grid[::stride, ::stride]

            Uquiv = np.zeros_like(zquiv_grid)
            Wquiv = np.zeros_like(zquiv_grid)
            mask = np.zeros_like(zquiv_grid)

            for x_index in range(num_horizontal_points // stride + (num_horizontal_points % stride != 0)):
                local_depth = evaluate_CF_point(H, self.hydro.mesh, squiv_grid[0, x_index], y)
                for z_index in range(num_vertical_points // stride):
                    if zquiv_grid[z_index, x_index] > -local_depth:
                        mask[z_index, x_index] = 1
                        sig_value = zquiv_grid[z_index, x_index] / local_depth
                        corresponding_sig_index = np.argmin(np.absolute(sig_range - np.ones_like(sig_range) * sig_value))
                        if sig_value >= sig_range[corresponding_sig_index]: # interpolation between sig_range[corresponding_sig_index] and sig_range[corresponding_sig_index + 1]
                            Uquiv[z_index, x_index] = U[corresponding_sig_index, x_index * stride] * (sig_range[corresponding_sig_index + 1] - sig_value) + U[corresponding_sig_index + 1, x_index * stride] * (sig_value - sig_range[corresponding_sig_index])
                            Uquiv[z_index, x_index] /= sig_range[corresponding_sig_index + 1] - sig_range[corresponding_sig_index]
                            Wquiv[z_index, x_index] = W_TWA[corresponding_sig_index, x_index * stride] * (sig_range[corresponding_sig_index + 1] - sig_value) + W_TWA[corresponding_sig_index + 1, x_index * stride] * (sig_value - sig_range[corresponding_sig_index])
                            Wquiv[z_index, x_index] /= sig_range[corresponding_sig_index + 1] - sig_range[corresponding_sig_index]
                        else:
                            Uquiv[z_index, x_index] = U[corresponding_sig_index - 1, x_index * stride] * (sig_range[corresponding_sig_index] - sig_value) + U[corresponding_sig_index, x_index * stride] * (sig_value - sig_range[corresponding_sig_index - 1])
                            Uquiv[z_index, x_index] /= sig_range[corresponding_sig_index] - sig_range[corresponding_sig_index - 1]
                            Wquiv[z_index, x_index] = W_TWA[corresponding_sig_index - 1, x_index * stride] * (sig_range[corresponding_sig_index] - sig_value) + W_TWA[corresponding_sig_index, x_index * stride] * (sig_value - sig_range[corresponding_sig_index - 1])
                            Wquiv[z_index, x_index] /= sig_range[corresponding_sig_index] - sig_range[corresponding_sig_index - 1]

            if spacing == 'sigma':
                visual_norms = np.sqrt((U[::stride,::stride] / (x_scaling))**2 + (W_TWA[::stride,::stride] / np.amax(depth))**2) # y-dimension in km
                physical_norms = np.sqrt((U[::stride,::stride])**2 + (W_TWA[::stride,::stride])**2)
                if length_indication == 'alpha':
                    quiv = current_ax.quiver(s_grid[::stride,::stride], z_grid[::stride,::stride], (U[::stride,::stride] / (x_scaling)) / visual_norms, (W_TWA[::stride,::stride] / np.amax(depth)) / visual_norms, color=arrow_color, alpha=physical_norms/np.amax(physical_norms), pivot='mid', headlength=3, headaxislength=3)
                elif length_indication == 'none':
                    quiv = current_ax.quiver(s_grid[::stride,::stride], z_grid[::stride,::stride], (U[::stride,::stride] / (x_scaling)) / visual_norms, (W_TWA[::stride,::stride] / np.amax(depth)) / visual_norms, color=arrow_color, pivot='mid', headlength=3, headaxislength=3)
                elif length_indication == 'length':
                    quiv = current_ax.quiver(s_grid[::stride,::stride], z_grid[::stride,::stride], U[::stride,::stride] / (x_scaling), W_TWA[::stride,::stride] / np.amax(depth), color=arrow_color, pivot='mid', headlength=3, headaxislength=3)
            elif spacing == 'equal':
                visual_norms = np.sqrt((Uquiv / (x_scaling))**2 + (Wquiv / np.amax(depth))**2) # y-dimension in km
                physical_norms = np.sqrt((Uquiv)**2 + (Wquiv)**2)
                if length_indication == 'alpha':
                    quiv = current_ax.quiver(squiv_grid, zquiv_grid, (Uquiv / (x_scaling)) / visual_norms, (Wquiv / np.amax(depth)) / visual_norms, color=arrow_color, alpha=physical_norms/np.amax(physical_norms), pivot='mid', headlength=3, headaxislength = 3)
                elif length_indication == 'none':
                    quiv = current_ax.quiver(squiv_grid, zquiv_grid, (Uquiv / (x_scaling)) / visual_norms, (Wquiv / np.amax(depth)) / visual_norms, color=arrow_color, pivot='mid', headlength=3, headaxislength = 3)
                elif length_indication == 'length':
                    quiv = current_ax.quiver(squiv_grid, zquiv_grid, Uquiv / (x_scaling), Wquiv / np.amax(depth), color=arrow_color, pivot='mid', headlength=3, headaxislength=3)


        current_ax.plot(x_range, -depth, linewidth=1, color='k', zorder=3)
        current_ax.fill_between(x_range, -np.amax(depth), -depth, color='silver')

        if plot_circulation:
            current_ax.set_title(f'{title}\nMaximum lateral velocity = {np.round(np.amax(physical_norms),4)} m/s')
        else:
            current_ax.set_title(title)
        xticks = list(np.linspace(0, 1, 10))
        current_ax.set_xticks(xticks)
        ticklabels = list(np.round(np.linspace(0, 1, 10), 3) * x_scaling / 1000)
        ticklabels_string = [str(np.round(tick, 3))[:4] if tick >= 0 else str(np.round(tick, 3))[:5] for tick in ticklabels]
        current_ax.set_xticklabels(ticklabels_string)

        current_ax.set_xlabel('x [km]')
        current_ax.set_ylabel('-Depth [m]')
        self.plot_counter += 1





# def decompose_hydro(hydro: Hydrodynamics, include_in_LHS_list: list, forcings: dict, folder_name: str = 'Decomposition', **kwargs):
#     """
#     terms in include_in_LHS_list are included in every single simulation and are viewed as part of the differential operator.
    
#     forcings is a dictionary with the following structure:

#     {'name of one of your chosen forcings': list of terms associated to that forcing, ...}

#     Example:
    
#     {'advection': ['momentum_advection', 'momentum_advection_surface_interactions'],
#     'tide': ['incoming_tide'],
#     'coriolis': ['coriolis', 'coriolis_surface_interactions']}

#     **kwargs for solve function
    
#     """

#     for forcing_name, as_forcing_list in forcings.items():
#         sub_hydro = ForcedLinearHydrodynamics(hydro, include_in_LHS_list, as_forcing_list)
#         solve(sub_hydro, **kwargs)
#         sub_hydro.save(f'{folder_name}/{forcing_name}')


# def decompose_hydro(postpro: PostProcessing, folder_name:str = 'Decomposition', processes: list = ['coriolis', 'momentum_advection_total', 'surface_interactions', 'density', 'tide', 'discharge']):
#     """Supported processes:
    
#     - Coriolis ('coriolis');
#     - Momentum advection ('momentum_advection_fullsplit', 'momentum_advection_componentsplit', 'momentum_advection_total')
#     - Density ('density');
#     - Tidal forcing ('tide');
#     - River discharge ('discharge')
    
    
#     """


#     M = postpro.hydro.numerical_information['M']
#     imax = postpro.hydro.numerical_information['imax']

#     H = postpro.hydro.spatial_parameters['H']
#     Hx = postpro.hydro.spatial_parameters_grad['H'][0]
#     Hy = postpro.hydro.spatial_parameters_grad['H'][1]
#     R = postpro.hydro.spatial_parameters['R']

#     if postpro.hydro.model_options['sea_boundary_treatment'] == 'exact':
#         sea_interpolant = ((postpro.hydro.geometric_information['riverine_boundary_x'] / postpro.hydro.geometric_information['x_scaling']) + (postpro.hydro.geometric_information['L_BL_river']/postpro.hydro.geometric_information['x_scaling']) + (postpro.hydro.geometric_information['L_R_river']/postpro.hydro.geometric_information['x_scaling']) + \
#                            (postpro.hydro.geometric_information['L_RA_river']/postpro.hydro.geometric_information['x_scaling']) - ngsolve.x) / \
#                            ((postpro.hydro.geometric_information['riverine_boundary_x']+postpro.hydro.geometric_information['L_BL_river']+postpro.hydro.geometric_information['L_R_river']+postpro.hydro.geometric_information['L_RA_river'] +
#                             postpro.hydro.geometric_information['L_BL_sea'] + postpro.hydro.geometric_information['L_R_sea'] + postpro.hydro.geometric_information['L_RA_sea']) / postpro.hydro.geometric_information['x_scaling'])
#         sea_interpolant_x = -1 / ((postpro.hydro.geometric_information['riverine_boundary_x']+postpro.hydro.geometric_information['L_BL_river']+postpro.hydro.geometric_information['L_R_river']+postpro.hydro.geometric_information['L_RA_river'] +
#                             postpro.hydro.geometric_information['L_BL_sea'] + postpro.hydro.geometric_information['L_R_sea'] + postpro.hydro.geometric_information['L_RA_sea']) / postpro.hydro.geometric_information['x_scaling'])
#     if postpro.hydro.model_options['river_boundary_treatment'] == 'exact':
#         river_interpolant = (-(postpro.hydro.geometric_information['L_BL_sea']/postpro.hydro.geometric_information['x_scaling']) - (postpro.hydro.geometric_information['L_R_sea']/postpro.hydro.geometric_information['x_scaling']) - \
#                              (postpro.hydro.geometric_information['L_RA_sea']/postpro.hydro.geometric_information['x_scaling']) + ngsolve.x) / \
#                              ((postpro.hydro.geometric_information['riverine_boundary_x']+postpro.hydro.geometric_information['L_BL_river']+postpro.hydro.geometric_information['L_R_river']+postpro.hydro.geometric_information['L_RA_river'] +
#                              postpro.hydro.geometric_information['L_BL_sea'] + postpro.hydro.geometric_information['L_R_sea'] + postpro.hydro.geometric_information['L_RA_sea']) / postpro.hydro.geometric_information['x_scaling'])
#         river_interpolant_x = 1 / ((postpro.hydro.geometric_information['riverine_boundary_x']+postpro.hydro.geometric_information['L_BL_river']+postpro.hydro.geometric_information['L_R_river']+postpro.hydro.geometric_information['L_RA_river'] +
#                              postpro.hydro.geometric_information['L_BL_sea'] + postpro.hydro.geometric_information['L_R_sea'] + postpro.hydro.geometric_information['L_RA_sea']) / postpro.hydro.geometric_information['x_scaling'])


#     os.makedirs(folder_name)

#     for process in processes:

#         if process == 'coriolis':
#             print("\nCoriolis\n")
#             forcing_u = [dict() for _ in range(M)]
#             forcing_v = [dict() for _ in range(M)]
#             forcing_z = dict()
#             forcing_dens = [0, 0]
#             forcing_tideBC = False
#             forcing_riverBC = False

#             for l in range(-imax, imax + 1):
#                 forcing_z[l] = 0

#             for p in range(M):
#                 for l in range(-imax, imax + 1):
#                     forcing_u[p][l] = -postpro.hydro.constant_physical_parameters['f'] * postpro.hydro.beta_solution[p][l] * (H+R)
#                     forcing_v[p][l] = postpro.hydro.constant_physical_parameters['f'] * postpro.hydro.alpha_solution[p][l] * (H+R)

#                     if postpro.hydro.model_options['river_boundary_treatment'] == 'exact':
#                         forcing_v[p][l] += postpro.hydro.constant_physical_parameters['f'] * postpro.hydro.Q_solution[l] * postpro.hydro.riverine_forcing.normal_alpha[p] * river_interpolant * (H+R)

#         elif process == 'tide':
#             print("\nTide\n")
#             forcing_u = [dict() for _ in range(M)]
#             forcing_v = [dict() for _ in range(M)]
#             forcing_z = dict()
#             forcing_dens = [0, 0]
#             forcing_tideBC = True
#             forcing_riverBC = False

#             for l in range(-imax, imax + 1):
#                 forcing_z[l] = 0

#             for p in range(M):
#                 for l in range(-imax, imax + 1):
#                     forcing_u[p][l] = 0
#                     forcing_v[p][l] = 0

#         elif process == 'discharge':
#             print("\nDischarge\n")
#             forcing_u = [dict() for _ in range(M)]
#             forcing_v = [dict() for _ in range(M)]
#             forcing_z = dict()
#             forcing_dens = [0, 0]
#             forcing_tideBC = False
#             forcing_riverBC = True

#             for l in range(-imax, imax + 1):
#                 forcing_z[l] = 0

#             for p in range(M):
#                 for l in range(-imax, imax + 1):
#                     forcing_u[p][l] = 0
#                     forcing_v[p][l] = 0

#         elif process == 'density':
#             print("\nDensity\n")
#             forcing_u = [dict() for _ in range(M)]
#             forcing_v = [dict() for _ in range(M)]
#             forcing_z = dict()
#             forcing_dens = [(1/postpro.hydro.geometric_information['x_scaling']) * 0.5 * np.sqrt(2) * postpro.hydro.vertical_basis.tensor_dict['G5'](p) * (H+R) * (H+R) * postpro.hydro.spatial_parameters_grad['rho'][0] / postpro.hydro.spatial_parameters['rho'], 
#                             (1/postpro.hydro.geometric_information['y_scaling']) * 0.5 * np.sqrt(2) * postpro.hydro.vertical_basis.tensor_dict['G5'](p) * (H+R) * (H+R) * postpro.hydro.spatial_parameters_grad['rho'][1] / postpro.hydro.spatial_parameters['rho']]
#             forcing_tideBC = False
#             forcing_riverBC = False

#             for l in range(-imax, imax + 1):
#                 forcing_z[l] = 0

#             for p in range(M):
#                 for l in range(-imax, imax + 1):
#                     forcing_u[p][l] = 0
#                     forcing_v[p][l] = 0

#         # Solve the linear model for the 'simple' processes
#         forced_hydro = ForcedLinearHydrodynamics(postpro.hydro, forcing_u, forcing_v, forcing_z, forcing_dens, forcing_tideBC, forcing_riverBC)
#         forced_hydro.setup_weak_form()
#         forced_hydro.solve()
#         forced_hydro.save(f"{folder_name}/{process}")

#         if process in ['momentum_advection_fullsplit', 'momentum_advection_componentsplit']:

#             # Construct the non-linear ramp
#             if postpro.hydro.geometric_information['L_R_sea'] > 1e-16:
#                 ramp_sea = ngsolve.IfPos(
#                     -postpro.hydro.geometric_information['L_R_sea']/postpro.hydro.geometric_information['x_scaling'] - ngsolve.x,
#                     ngsolve.IfPos(
#                         -postpro.hydro.geometric_information['L_R_sea']/postpro.hydro.geometric_information['x_scaling'] - postpro.hydro.geometric_information['L_RA_sea']/postpro.hydro.geometric_information['x_scaling'] - ngsolve.x,
#                         0,
#                         0.5 * (1 + ngsolve_tanh((4 * (ngsolve.x + postpro.hydro.geometric_information['L_RA_sea']/postpro.hydro.geometric_information['x_scaling'] + 0.5 * postpro.hydro.geometric_information['L_R_sea']/postpro.hydro.geometric_information['x_scaling']) / (postpro.hydro.geometric_information['L_R_sea']/postpro.hydro.geometric_information['x_scaling'])) / 
#                                                 (1 - (2*(ngsolve.x + postpro.hydro.geometric_information['L_RA_sea']/postpro.hydro.geometric_information['x_scaling'] + 0.5*postpro.hydro.geometric_information['L_R_sea']/postpro.hydro.geometric_information['x_scaling']) / (postpro.hydro.geometric_information['L_R_sea']/postpro.hydro.geometric_information['x_scaling']))**2)))
#                     ),
#                     1
#                 )
#             else:
#                 ramp_sea = ngsolve(1)

#             if postpro.hydro.geometric_information['L_R_river'] > 1e-16:
#                 ramp_river = ngsolve.IfPos(
#                             -(postpro.hydro.geometric_information['riverine_boundary_x'] + postpro.hydro.geometric_information['L_RA_river']) / postpro.hydro.geometric_information['x_scaling'] + ngsolve.x,
#                             ngsolve.IfPos(
#                                 -(postpro.hydro.geometric_information['riverine_boundary_x'] + postpro.hydro.geometric_information['L_R_river'] + postpro.hydro.geometric_information['L_RA_river']) / postpro.hydro.geometric_information['x_scaling'] + ngsolve.x,
#                                 0,
#                                 0.5 * (1 + ngsolve_tanh((-4 * (ngsolve.x - postpro.hydro.geometric_information['L_RA_river'] / postpro.hydro.geometric_information['x_scaling'] - 0.5 * postpro.hydro.geometric_information["L_R_river"] / postpro.hydro.geometric_information['x_scaling'] - postpro.hydro.geometric_information['riverine_boundary_x'] / postpro.hydro.geometric_information['x_scaling']) / 
#                                                         (postpro.hydro.geometric_information['L_R_river'] / postpro.hydro.geometric_information['x_scaling'])) / (1 - (2*(ngsolve.x- postpro.hydro.geometric_information['L_RA_river']/postpro.hydro.geometric_information['x_scaling'] - 0.5 * postpro.hydro.geometric_information["L_R_river"]/postpro.hydro.geometric_information['x_scaling'] - postpro.hydro.geometric_information['riverine_boundary_x']/postpro.hydro.geometric_information['x_scaling'])/(postpro.hydro.geometric_information["L_R_river"]/postpro.hydro.geometric_information['x_scaling']))**2)))
#                             ),
#                             1
#                         )
#             else:
#                 ramp_river = ngsolve(1)

#             ramp = ramp_sea * ramp_river  

#             G2 = postpro.hydro.vertical_basis.tensor_dict['G1']
#             H3 = postpro.hydro.time_basis.tensor_dict['H3']
#             G6 = postpro.hydro.vertical_basis.tensor_dict['G6']

#             # project vertical velocity
#             proj_W = [dict() for _ in range(M)]
#             sig_range = np.linspace(-1, 0, 200)
#             dsig = sig_range[1] - sig_range[0]
#             for l in range(-imax, imax+1):
#                 for p in range(M):
#                     proj_W[p][l] = dsig * sum([postpro.w(l, sig) * postpro.hydro.vertical_basis.evaluation_function(sig, p) for sig in list(sig_range)]) / postpro.hydro.vertical_basis.inner_product(p, p)

#             os.makedirs(f"{folder_name}/momentum_advection")

#             for i in range(0, imax+1):
#                 for j in range(0, imax+1):
#                     print(f"\nMomentum advection M{2*i} x M{2*j}")
#                     # Effect of interaction M_(2i) x M_(2j)
#                     forcing_uux = [dict() for _ in range(M)]
#                     forcing_vuy = [dict() for _ in range(M)]
#                     forcing_wuz = [dict() for _ in range(M)]
#                     forcing_uvx = [dict() for _ in range(M)]
#                     forcing_vvy = [dict() for _ in range(M)]
#                     forcing_wvz = [dict() for _ in range(M)]

#                     if process == 'momentum_advection_componentsplit':
#                         forcing_u = [dict() for _ in range(M)]
#                         forcing_v = [dict() for _ in range(M)]
#                         forcing_z = {l: 0 for l in range(-imax, imax + 1)}

#                     for p in range(M):
#                         for l in range(-imax, imax+1):
#                             if postpro.hydro.model_options['river_boundary_treatment'] == 'simple':
#                                 forcing_uux[p][l] = H3(i, j, l) * ramp * sum([sum([H / postpro.hydro.geometric_information['x_scaling'] * G2(m, n, p) * postpro.hydro.alpha_solution[m][i] * ngsolve.grad(postpro.hydro.alpha_solution[n][j])[0] - \
#                                                                             Hx * G6(m, n, p) * postpro.hydro.alpha_solution[m][i] * postpro.hydro.alpha_solution[n][j] for n in range(M)]) for m in range(M)])
#                                 forcing_vuy[p][l] = H3(i, j, l) * ramp * sum([sum([H / postpro.hydro.geometric_information['y_scaling'] * G2(m, n, p) * postpro.hydro.beta_solution[m][i] * ngsolve.grad(postpro.hydro.alpha_solution[n][j])[1] - \
#                                                                             Hy * G6(m, n, p) * postpro.hydro.beta_solution[m][i] * postpro.hydro.alpha_solution[n][j] for n in range(M)]) for m in range(M)])
#                                 forcing_uvx[p][l] = H3(i, j, l) * ramp * sum([sum([H / postpro.hydro.geometric_information['x_scaling'] * G2(m, n, p) * postpro.hydro.alpha_solution[m][i] * ngsolve.grad(postpro.hydro.beta_solution[n][j])[0] - \
#                                                                             Hx * G6(m, n, p) * postpro.hydro.alpha_solution[m][i] * postpro.hydro.beta_solution[n][j] for n in range(M)]) for m in range(M)])
#                                 forcing_vvy[p][l] = H3(i, j, l) * ramp * sum([sum([H / postpro.hydro.geometric_information['y_scaling'] * G2(m, n, p) * postpro.hydro.beta_solution[m][i] * ngsolve.grad(postpro.hydro.beta_solution[n][j])[1] - \
#                                                                             Hy * G6(m, n, p) * postpro.hydro.beta_solution[m][i] * postpro.hydro.beta_solution[n][j] for n in range(M)]) for m in range(M)])
                                
#                                 forcing_wuz[p][l] = H3(i, j, l) * ramp * sum([sum([G6(m, n, p) * proj_W[p][l] * postpro.hydro.alpha_solution[p][l] for n in range(M)]) for m in range(M)])
#                                 forcing_wvz[p][l] = H3(i, j, l) * ramp * sum([sum([G6(m, n, p) * proj_W[p][l] * postpro.hydro.beta_solution[p][l] for n in range(M)]) for m in range(M)])
#                             elif postpro.hydro.model_options['river_boundary_treatment'] == 'exact':
#                                 forcing_uux[p][l] = H3(i, j, l) * ramp * sum([sum([H / postpro.hydro.geometric_information['x_scaling'] * G2(m, n, p) * (postpro.hydro.alpha_solution[m][i] + postpro.hydro.Q_solution[i] * postpro.hydro.riverine_forcing.normal_alpha[m] * river_interpolant) * (ngsolve.grad(postpro.hydro.alpha_solution[n][j])[0] + postpro.hydro.Q_solution[j] * postpro.hydro.riverine_forcing.normal_alpha[n] * river_interpolant_x) - \
#                                                                             Hx * G6(m, n, p) * (postpro.hydro.alpha_solution[m][i] + postpro.hydro.Q_solution[i] * postpro.hydro.riverine_forcing.normal_alpha[m] * river_interpolant) * (postpro.hydro.alpha_solution[n][j] + postpro.hydro.Q_solution[j] * postpro.hydro.riverine_forcing.normal_alpha[n] * river_interpolant) for n in range(M)]) for m in range(M)])
#                                 forcing_vuy[p][l] = H3(i, j, l) * ramp * sum([sum([H / postpro.hydro.geometric_information['y_scaling'] * G2(m, n, p) * postpro.hydro.beta_solution[m][i] * (ngsolve.grad(postpro.hydro.alpha_solution[n][j])[1]+postpro.hydro.Q_solution[j] * postpro.hydro.riverine_forcing.normal_alpha_y[n] * river_interpolant) - \
#                                                                             Hy * G6(m, n, p) * postpro.hydro.beta_solution[m][i] * (postpro.hydro.alpha_solution[n][j] + postpro.hydro.Q_solution[j] * postpro.hydro.riverine_forcing.normal_alpha[n] * river_interpolant) for n in range(M)]) for m in range(M)])
#                                 forcing_uvx[p][l] = H3(i, j, l) * ramp * sum([sum([H / postpro.hydro.geometric_information['x_scaling'] * G2(m, n, p) * (postpro.hydro.alpha_solution[m][i] + postpro.hydro.Q_solution[i] * postpro.hydro.riverine_forcing.normal_alpha[m] * river_interpolant) * ngsolve.grad(postpro.hydro.beta_solution[n][j])[0] - \
#                                                                             Hx * G6(m, n, p) * (postpro.hydro.alpha_solution[m][i] + postpro.hydro.Q_solution[i] * postpro.hydro.riverine_forcing.normal_alpha[m] * river_interpolant) * postpro.hydro.beta_solution[n][j] for n in range(M)]) for m in range(M)])
#                                 forcing_vvy[p][l] = H3(i, j, l) * ramp * sum([sum([H / postpro.hydro.geometric_information['y_scaling'] * G2(m, n, p) * postpro.hydro.beta_solution[m][i] * ngsolve.grad(postpro.hydro.beta_solution[n][j])[1] - \
#                                                                             Hy * G6(m, n, p) * postpro.hydro.beta_solution[m][i] * postpro.hydro.beta_solution[n][j] for n in range(M)]) for m in range(M)])
                                
#                                 forcing_wuz[p][l] = H3(i, j, l) * ramp * sum([sum([G6(m, n, p) * proj_W[p][l] * (postpro.hydro.alpha_solution[m][i] + postpro.hydro.Q_solution[i] * postpro.hydro.riverine_forcing.normal_alpha[m] * river_interpolant) for n in range(M)]) for m in range(M)])
#                                 forcing_wvz[p][l] = H3(i, j, l) * ramp * sum([sum([G6(m, n, p) * proj_W[p][l] * postpro.hydro.beta_solution[p][l] for n in range(M)]) for m in range(M)])

#                             if process == 'momentum_advection_componentsplit':
#                                 forcing_u[p][l] = forcing_uux[p][l] + forcing_vuy[p][l] + forcing_wuz[p][l]
#                                 forcing_v[p][l] = forcing_uvx[p][l] + forcing_vvy[p][l] + forcing_wvz[p][l]

#                     if process == 'momentum_advection_componentsplit':
#                         forced_hydro = ForcedLinearHydrodynamics(postpro.hydro, forcing_u, forcing_v, forcing_z, [0, 0], False, False)
#                         forced_hydro.setup_weak_form()
#                         forced_hydro.solve()
#                         forced_hydro.save(f'{folder_name}/momentum_advection/M{2 * i}M{2 * j}')
#                     elif process == 'momentum_advection_fullsplit':
#                         forced_hydro_lonu = ForcedLinearHydrodynamics(postpro.hydro, forcing_uux, [{l: 0 for l in range(-imax, imax + 1)} for _ in range(M)], {l: 0 for l in range(-imax, imax + 1)}, [0, 0], False, False)
#                         forced_hydro_lonu.setup_weak_form()
#                         forced_hydro_lonu.solve()
#                         forced_hydro_lonu.save(f"{folder_name}/momentum_advection/M{2*i}M{2*j}uux")

#                         forced_hydro_latu = ForcedLinearHydrodynamics(postpro.hydro, forcing_vuy, [{l: 0 for l in range(-imax, imax + 1)} for _ in range(M)], {l: 0 for l in range(-imax, imax + 1)}, [0, 0], False, False)
#                         forced_hydro_latu.setup_weak_form()
#                         forced_hydro_latu.solve()
#                         forced_hydro_latu.save(f"{folder_name}/momentum_advection/M{2*i}M{2*j}vuy")

#                         forced_hydro_veru = ForcedLinearHydrodynamics(postpro.hydro, forcing_wuz, [{l: 0 for l in range(-imax, imax + 1)} for _ in range(M)], {l: 0 for l in range(-imax, imax + 1)}, [0, 0], False, False)
#                         forced_hydro_veru.setup_weak_form()
#                         forced_hydro_veru.solve()
#                         forced_hydro_veru.save(f"{folder_name}/momentum_advection/M{2*i}M{2*j}wuz")

#                         forced_hydro_lonv = ForcedLinearHydrodynamics(postpro.hydro, [{l: 0 for l in range(-imax, imax + 1)} for _ in range(M)], forcing_uvx, {l: 0 for l in range(-imax, imax + 1)}, [0, 0], False, False)
#                         forced_hydro_lonv.setup_weak_form()
#                         forced_hydro_lonv.solve()
#                         forced_hydro_lonv.save(f"{folder_name}/momentum_advection/M{2*i}M{2*j}uvx")

#                         forced_hydro_latv = ForcedLinearHydrodynamics(postpro.hydro, [{l: 0 for l in range(-imax, imax + 1)} for _ in range(M)], forcing_vvy, {l: 0 for l in range(-imax, imax + 1)}, [0, 0], False, False)
#                         forced_hydro_latv.setup_weak_form()
#                         forced_hydro_latv.solve()
#                         forced_hydro_latv.save(f"{folder_name}/momentum_advection/M{2*i}M{2*j}vvy")

#                         forced_hydro_verv = ForcedLinearHydrodynamics(postpro.hydro, [{l: 0 for l in range(-imax, imax + 1)} for _ in range(M)], forcing_wvz, {l: 0 for l in range(-imax, imax + 1)}, [0, 0], False, False)
#                         forced_hydro_verv.setup_weak_form()
#                         forced_hydro_verv.solve()
#                         forced_hydro_verv.save(f"{folder_name}/momentum_advection/M{2*i}M{2*j}wvz")
                    
#         elif process == 'momentum_advection_total':
#             print("\nMomentum advection\n")

#             # Construct the non-linear ramp
#             if postpro.hydro.geometric_information['L_R_sea'] > 1e-16:
#                 ramp_sea = ngsolve.IfPos(
#                     -postpro.hydro.geometric_information['L_R_sea']/postpro.hydro.geometric_information['x_scaling'] - ngsolve.x,
#                     ngsolve.IfPos(
#                         -postpro.hydro.geometric_information['L_R_sea']/postpro.hydro.geometric_information['x_scaling'] - postpro.hydro.geometric_information['L_RA_sea']/postpro.hydro.geometric_information['x_scaling'] - ngsolve.x,
#                         0,
#                         0.5 * (1 + ngsolve_tanh((4 * (ngsolve.x + postpro.hydro.geometric_information['L_RA_sea']/postpro.hydro.geometric_information['x_scaling'] + 0.5 * postpro.hydro.geometric_information['L_R_sea']/postpro.hydro.geometric_information['x_scaling']) / (postpro.hydro.geometric_information['L_R_sea']/postpro.hydro.geometric_information['x_scaling'])) / 
#                                                 (1 - (2*(ngsolve.x + postpro.hydro.geometric_information['L_RA_sea']/postpro.hydro.geometric_information['x_scaling'] + 0.5*postpro.hydro.geometric_information['L_R_sea']/postpro.hydro.geometric_information['x_scaling']) / (postpro.hydro.geometric_information['L_R_sea']/postpro.hydro.geometric_information['x_scaling']))**2)))
#                     ),
#                     1
#                 )
#             else:
#                 ramp_sea = ngsolve(1)

#             if postpro.hydro.geometric_information['L_R_river'] > 1e-16:
#                 ramp_river = ngsolve.IfPos(
#                             -(postpro.hydro.geometric_information['riverine_boundary_x'] + postpro.hydro.geometric_information['L_RA_river']) / postpro.hydro.geometric_information['x_scaling'] + ngsolve.x,
#                             ngsolve.IfPos(
#                                 -(postpro.hydro.geometric_information['riverine_boundary_x'] + postpro.hydro.geometric_information['L_R_river'] + postpro.hydro.geometric_information['L_RA_river']) / postpro.hydro.geometric_information['x_scaling'] + ngsolve.x,
#                                 0,
#                                 0.5 * (1 + ngsolve_tanh((-4 * (ngsolve.x - postpro.hydro.geometric_information['L_RA_river'] / postpro.hydro.geometric_information['x_scaling'] - 0.5 * postpro.hydro.geometric_information["L_R_river"] / postpro.hydro.geometric_information['x_scaling'] - postpro.hydro.geometric_information['riverine_boundary_x'] / postpro.hydro.geometric_information['x_scaling']) / 
#                                                         (postpro.hydro.geometric_information['L_R_river'] / postpro.hydro.geometric_information['x_scaling'])) / (1 - (2*(ngsolve.x- postpro.hydro.geometric_information['L_RA_river']/postpro.hydro.geometric_information['x_scaling'] - 0.5 * postpro.hydro.geometric_information["L_R_river"]/postpro.hydro.geometric_information['x_scaling'] - postpro.hydro.geometric_information['riverine_boundary_x']/postpro.hydro.geometric_information['x_scaling'])/(postpro.hydro.geometric_information["L_R_river"]/postpro.hydro.geometric_information['x_scaling']))**2)))
#                             ),
#                             1
#                         )
#             else:
#                 ramp_river = ngsolve(1)

#             ramp = ramp_sea * ramp_river  

#             G2 = postpro.hydro.vertical_basis.tensor_dict['G1']
#             H3 = postpro.hydro.time_basis.tensor_dict['H3']
#             G6 = postpro.hydro.vertical_basis.tensor_dict['G6']

#             # project vertical velocity
#             proj_W = [dict() for _ in range(M)]
#             sig_range = np.linspace(-1, 0, 200)
#             dsig = sig_range[1] - sig_range[0]
#             for l in range(-imax, imax+1):
#                 for p in range(M):
#                     proj_W[p][l] = dsig * sum([postpro.w(l, sig) * postpro.hydro.vertical_basis.evaluation_function(sig, p) for sig in list(sig_range)]) / postpro.hydro.vertical_basis.inner_product(p, p)

#             forcing_u = [dict() for _ in range(M)]
#             forcing_v = [dict() for _ in range(M)]
#             forcing_z = {l: 0 for l in range(-imax, imax)}

#             for p in range(M):
#                 for l in range(M):
#                     if postpro.hydro.model_options['river_boundary_treatment'] == 'simple':
#                         forcing_uux = ramp * sum([sum([sum([sum([H * H3(i, j, l) * G2(m, n, p) / postpro.hydro.geometric_information['x_scaling'] * postpro.hydro.alpha_solution[m][i] * ngsolve.grad(postpro.hydro.alpha_solution[n][j])[0] - \
#                                                                 H3(i, j, l) * G6(m, n, p) * Hx * postpro.hydro.alpha_solution[m][i] * postpro.hydro.alpha_solution[n][j] for j in range(-imax, imax + 1)]) for i in range(-imax, imax + 1)]) for n in range(M)]) for m in range(M)])
#                         forcing_vuy = ramp * sum([sum([sum([sum([H * H3(i, j, l) * G2(m, n, p) / postpro.hydro.geometric_information['y_scaling'] * postpro.hydro.beta_solution[m][i] * ngsolve.grad(postpro.hydro.alpha_solution[n][j])[1] - \
#                                                                 H3(i, j, l) * G6(m, n, p) * Hy * postpro.hydro.beta_solution[m][i] * postpro.hydro.alpha_solution[n][j] for j in range(-imax, imax + 1)]) for i in range(-imax, imax + 1)]) for n in range(M)]) for m in range(M)])
#                         forcing_wuz = ramp * sum([sum([sum([sum([H3(i, j, l) * G6(m, n, p) * proj_W[m][i] * postpro.hydro.alpha_solution[n][j] for j in range(-imax, imax + 1)]) for i in range(-imax, imax + 1)]) for n in range(M)]) for m in range(M)])
#                         forcing_uvx = ramp * sum([sum([sum([sum([H * H3(i, j, l) * G2(m, n, p) / postpro.hydro.geometric_information['x_scaling'] * postpro.hydro.alpha_solution[m][i] * ngsolve.grad(postpro.hydro.beta_solution[n][j])[0] - \
#                                                                 H3(i, j, l) * G6(m, n, p) * Hx * postpro.hydro.alpha_solution[m][i] * postpro.hydro.beta_solution[n][j] for j in range(-imax, imax + 1)]) for i in range(-imax, imax + 1)]) for n in range(M)]) for m in range(M)])
#                         forcing_vvy = ramp * sum([sum([sum([sum([H * H3(i, j, l) * G2(m, n, p) / postpro.hydro.geometric_information['y_scaling'] * postpro.hydro.beta_solution[m][i] * ngsolve.grad(postpro.hydro.beta_solution[n][j])[1] - \
#                                                                 H3(i, j, l) * G6(m, n, p) * Hy * postpro.hydro.beta_solution[m][i] * postpro.hydro.beta_solution[n][j] for j in range(-imax, imax + 1)]) for i in range(-imax, imax + 1)]) for n in range(M)]) for m in range(M)])
#                         forcing_wvz = ramp * sum([sum([sum([sum([H3(i, j, l) * G6(m, n, p) * proj_W[m][i] * postpro.hydro.beta_solution[n][j] for j in range(-imax, imax + 1)]) for i in range(-imax, imax + 1)]) for n in range(M)]) for m in range(M)])

#                     elif postpro.hydro.model_options['river_boundary_treatment'] == 'exact':
#                         forcing_uux = ramp * sum([sum([sum([sum([H * H3(i, j, l) * G2(m, n, p) / postpro.hydro.geometric_information['x_scaling'] * (postpro.hydro.alpha_solution[m][i] + postpro.hydro.Q_solution[i] * postpro.hydro.riverine_forcing.normal_alpha[m] * river_interpolant) * (ngsolve.grad(postpro.hydro.alpha_solution[n][j])[0] + postpro.hydro.Q_solution[j] * postpro.hydro.riverine_forcing.normal_alpha[n] * river_interpolant_x) - \
#                                                                 H3(i, j, l) * G6(m, n, p) * Hx * (postpro.hydro.alpha_solution[m][i] + postpro.hydro.Q_solution[i] * postpro.hydro.riverine_forcing.normal_alpha[m] * river_interpolant) * (postpro.hydro.alpha_solution[n][j] + postpro.hydro.Q_solution[j] * postpro.hydro.riverine_forcing.normal_alpha[n] * river_interpolant) for j in range(-imax, imax + 1)]) for i in range(-imax, imax + 1)]) for n in range(M)]) for m in range(M)])
#                         forcing_vuy = ramp * sum([sum([sum([sum([H * H3(i, j, l) * G2(m, n, p) / postpro.hydro.geometric_information['y_scaling'] * postpro.hydro.beta_solution[m][i] * (ngsolve.grad(postpro.hydro.alpha_solution[n][j])[1] + postpro.hydro.Q_solution[j] * postpro.hydro.riverine_forcing.normal_alpha_y[n] * river_interpolant) - \
#                                                                 H3(i, j, l) * G6(m, n, p) * Hy * postpro.hydro.beta_solution[m][i] * (postpro.hydro.alpha_solution[n][j] + postpro.hydro.Q_solution[j] * postpro.hydro.riverine_forcing.normal_alpha[n] * river_interpolant) for j in range(-imax, imax + 1)]) for i in range(-imax, imax + 1)]) for n in range(M)]) for m in range(M)])
#                         forcing_wuz = ramp * sum([sum([sum([sum([H3(i, j, l) * G6(m, n, p) * proj_W[m][i] * (postpro.hydro.alpha_solution[n][j] + postpro.hydro.Q_solution[j] * postpro.hydro.riverine_forcing.normal_alpha[n] * river_interpolant) for j in range(-imax, imax + 1)]) for i in range(-imax, imax + 1)]) for n in range(M)]) for m in range(M)])
#                         forcing_uvx = ramp * sum([sum([sum([sum([H * H3(i, j, l) * G2(m, n, p) / postpro.hydro.geometric_information['x_scaling'] * (postpro.hydro.alpha_solution[m][i] + postpro.hydro.Q_solution[i] * postpro.hydro.riverine_forcing.normal_alpha[m] * river_interpolant) * ngsolve.grad(postpro.hydro.beta_solution[n][j])[0] - \
#                                                                 H3(i, j, l) * G6(m, n, p) * Hx * (postpro.hydro.alpha_solution[m][i] + postpro.hydro.Q_solution[i] * postpro.hydro.riverine_forcing.normal_alpha[m] * river_interpolant) * postpro.hydro.beta_solution[n][j] for j in range(-imax, imax + 1)]) for i in range(-imax, imax + 1)]) for n in range(M)]) for m in range(M)])
#                         forcing_vvy = ramp * sum([sum([sum([sum([H * H3(i, j, l) * G2(m, n, p) / postpro.hydro.geometric_information['y_scaling'] * postpro.hydro.beta_solution[m][i] * ngsolve.grad(postpro.hydro.beta_solution[n][j])[1] - \
#                                                                 H3(i, j, l) * G6(m, n, p) * Hy * postpro.hydro.beta_solution[m][i] * postpro.hydro.beta_solution[n][j] for j in range(-imax, imax + 1)]) for i in range(-imax, imax + 1)]) for n in range(M)]) for m in range(M)])
#                         forcing_wvz = ramp * sum([sum([sum([sum([H3(i, j, l) * G6(m, n, p) * proj_W[m][i] * postpro.hydro.beta_solution[n][j] for j in range(-imax, imax + 1)]) for i in range(-imax, imax + 1)]) for n in range(M)]) for m in range(M)])


#                     forcing_u[p][l] = forcing_uux + forcing_vuy + forcing_wuz
#                     forcing_v[p][l] = forcing_uvx + forcing_vvy + forcing_wvz

#             forced_hydro = ForcedLinearHydrodynamics(postpro.hydro, forcing_u, forcing_v, forcing_z, [0, 0], False, False)
#             forced_hydro.setup_weak_form()
#             forced_hydro.solve()
#             forced_hydro.save(f'{folder_name}/momentum_advection')
        