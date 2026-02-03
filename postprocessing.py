import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.tri as tri
import ngsolve
import os

from NiFlow.hydrodynamics import Hydrodynamics
from NiFlow.utils import *
from NiFlow.solve import solve


mpl.rcParams['axes.formatter.useoffset'] = False
mpl.rcParams['font.family'] = 'sans-serif'


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

    def __init__(self, hydro: Hydrodynamics, parent_hydro: Hydrodynamics = None, transport_analysis_mode = 'own_surface'):

        """
        Creates functions that represent the velocity fields in different ways (timed, basis coefficients, depth-averaged, ...).

        Arguments:

        - hydro: hydrodynamics-object of the simulation to postprocess
        - parent_hydro: hydrodynamics-object of the full model, that you provide if this PostProcessing object is associated to a contribution in the decomposition.
        - transport_analysis_mode: either 'own_surface', 'parent_surface', 'no_surface'. Indicates whether transport quantities are computed with the own surface solution, the parent's surface solution, or no surface at all (linearised tide contributions)
        
        """

        self.hydro = hydro
        self.parent_hydro = parent_hydro
        self.transport_analysis_mode = transport_analysis_mode

        self.M = hydro.numerical_information['M']
        self.imax = hydro.numerical_information['imax']

        self.x_scaling = hydro.geometric_information['x_scaling']
        self.y_scaling = hydro.geometric_information['y_scaling']

        self.river_interpolant = ((hydro.geometric_information['L_BL_sea']/self.x_scaling) + (hydro.geometric_information['L_R_sea']/self.x_scaling) + \
                            (hydro.geometric_information['L_RA_sea']/self.x_scaling) + ngsolve.x) / \
                            ((hydro.geometric_information['riverine_boundary_x']+hydro.geometric_information['L_BL_river']+hydro.geometric_information['L_R_river']+hydro.geometric_information['L_RA_river'] +
                            hydro.geometric_information['L_BL_sea'] + hydro.geometric_information['L_R_sea'] + hydro.geometric_information['L_RA_sea']) / self.x_scaling)
        self.river_interpolant_x = 1 / ((hydro.geometric_information['riverine_boundary_x']+hydro.geometric_information['L_BL_river']+hydro.geometric_information['L_R_river']+hydro.geometric_information['L_RA_river'] +
                            hydro.geometric_information['L_BL_sea'] + hydro.geometric_information['L_R_sea'] + hydro.geometric_information['L_RA_sea']) / self.x_scaling)


        if hydro.model_options['river_boundary_treatment'] == 'exact':
            Q = self.hydro.Q_solution
            self.eval_Q = evaluate_CF_point(Q[0], self.hydro.mesh, 0.5, 0)
        else:
            Q = {q: ngsolve.CF(0) for q in range(-self.imax, self.imax + 1)}
            self.eval_Q = 0
        

        self.u = lambda q, sigma : sum([(hydro.alpha_solution[m][q]+ Q[q] * self.river_interpolant) * hydro.vertical_basis.evaluation_function(sigma, m) for m in range(self.M)])
        self.v = lambda q, sigma : sum([hydro.beta_solution[m][q] * hydro.vertical_basis.evaluation_function(sigma, m) for m in range(self.M)])
        self.gamma = lambda q : hydro.gamma_solution[q]

        self.u_timed = lambda t, sigma: sum([sum([(hydro.alpha_solution[m][q]+Q[q]*self.river_interpolant) * hydro.vertical_basis.evaluation_function(sigma, m) * hydro.time_basis.evaluation_function(t, q) for m in range(self.M)]) for q in range(-self.imax, self.imax+1)])
        self.v_timed = lambda t, sigma: sum([sum([hydro.beta_solution[m][q] * hydro.vertical_basis.evaluation_function(sigma, m) * hydro.time_basis.evaluation_function(t, q) for m in range(self.M)]) for q in range(-self.imax, self.imax + 1)])
        self.zeta_timed = lambda t: sum([self.hydro.gamma_solution[l] * self.hydro.time_basis.evaluation_function(t, l) for l in range(-self.imax, self.imax + 1)])
        self.zetat_timed = lambda t: sum([self.hydro.gamma_solution[l] * self.hydro.time_basis.derivative_evaluation_function(t, l) for l in range(-self.imax, self.imax + 1)])

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

        # Get derivative gridfunctions

        self._construct_gradients()
        self._construct_transport()
        self._construct_TWA_velocities()



    def _construct_gradients(self):

        if self.hydro.model_options['river_boundary_treatment'] == 'exact':
            Q = self.hydro.Q_solution
        else:
            Q = {q: ngsolve.CF(0) for q in range(-self.imax, self.imax + 1)}

        self.ux = lambda q, sigma : sum([(ngsolve.grad(self.hydro.alpha_solution[m][q])[0] + Q[q]*self.river_interpolant_x)* self.hydro.vertical_basis.evaluation_function(sigma, m) for m in range(self.M)]) / self.x_scaling
        self.vx = lambda q, sigma : sum([ngsolve.grad(self.hydro.beta_solution[m][q])[0] * self.hydro.vertical_basis.evaluation_function(sigma, m) for m in range(self.M)]) / self.x_scaling
        self.gammax = lambda q: ngsolve.grad(self.hydro.gamma_solution[q])[0] / self.x_scaling

        self.uy = lambda q, sigma : sum([ngsolve.grad(self.hydro.alpha_solution[m][q])[1] * self.hydro.vertical_basis.evaluation_function(sigma, m) for m in range(self.M)]) / self.y_scaling 
        self.vy = lambda q, sigma : sum([ngsolve.grad(self.hydro.beta_solution[m][q])[1] * self.hydro.vertical_basis.evaluation_function(sigma, m) for m in range(self.M)]) / self.y_scaling
        self.gammay = lambda q: ngsolve.grad(self.hydro.gamma_solution[q])[1] / self.y_scaling

        self.usig = lambda q, sigma : sum([(self.hydro.alpha_solution[m][q] + Q[q]*self.river_interpolant)* self.hydro.vertical_basis.derivative_evaluation_function(sigma, m) for m in range(self.M)])
        self.vsig = lambda q, sigma : sum([self.hydro.beta_solution[m][q] * self.hydro.vertical_basis.derivative_evaluation_function(sigma, m) for m in range(self.M)]) #same

        self.usigsig = lambda q, sigma: sum([(self.hydro.alpha_solution[m][q] + Q[q]*self.river_interpolant) * self.hydro.vertical_basis.second_derivative_evaluation_function(sigma, m) for m in range(self.M)])
        self.vsigsig = lambda q, sigma: sum([self.hydro.beta_solution[m][q] * self.hydro.vertical_basis.second_derivative_evaluation_function(sigma, m) for m in range(self.M)]) #This is the derivative w.r.t. sigma, and not z. To transform this to the derivative w.r.t. z, divide by H^2

        self.ux_timed = lambda t, sigma: sum([sum([(ngsolve.grad(self.hydro.alpha_solution[m][q])[0] + Q[q]*self.river_interpolant_x) * self.hydro.vertical_basis.evaluation_function(sigma, m) * self.hydro.time_basis.evaluation_function(t, q) for m in range(self.M)]) for q in range(-self.imax, self.imax+1)]) / self.x_scaling
        self.vx_timed = lambda t, sigma: sum([sum([ngsolve.grad(self.hydro.beta_solution[m][q])[0] * self.hydro.vertical_basis.evaluation_function(sigma, m) * self.hydro.time_basis.evaluation_function(t, q) for m in range(self.M)]) for q in range(-self.imax, self.imax + 1)]) / self.x_scaling

        self.uy_timed = lambda t, sigma: sum([sum([ngsolve.grad(self.hydro.alpha_solution[m][q])[1] * self.hydro.vertical_basis.evaluation_function(sigma, m) * self.hydro.time_basis.evaluation_function(t, q) for m in range(self.M)]) for q in range(-self.imax, self.imax+1)]) / self.y_scaling        
        self.vy_timed = lambda t, sigma: sum([sum([ngsolve.grad(self.hydro.beta_solution[m][q])[1] *self. hydro.vertical_basis.evaluation_function(sigma, m) * self.hydro.time_basis.evaluation_function(t, q) for m in range(self.M)]) for q in range(-self.imax, self.imax + 1)]) / self.y_scaling

        self.usig_timed = lambda t, sigma: sum([sum([(self.hydro.alpha_solution[m][q] + Q[q] * self.river_interpolant) * self.hydro.vertical_basis.derivative_evaluation_function(sigma, m) * self.hydro.time_basis.evaluation_function(t, q) for m in range(self.M)]) for q in range(-self.imax, self.imax+1)])
        self.vsig_timed = lambda t, sigma: sum([sum([self.hydro.beta_solution[m][q] * self.hydro.vertical_basis.derivative_evaluation_function(sigma, m) * self.hydro.time_basis.evaluation_function(t, q) for m in range(self.M)]) for q in range(-self.imax, self.imax + 1)])


    def _construct_transport(self):
        """Constructs depth-integrated velocity as a function of (x,y,t) as well as the time-averaged variant of that. Assumes that surface interactions have been
        taken into account."""

        G4 = self.hydro.vertical_basis.tensor_dict['G4']

        H = self.hydro.spatial_parameters['H']
        alpha = self.hydro.alpha_solution
        beta = self.hydro.beta_solution

        # choose surface based on your transport analysis mode
        if self.transport_analysis_mode == 'own_surface':
            gamma = self.hydro.gamma_solution # use own zeta if not a decomposition
        elif self.transport_analysis_mode == 'parent_surface':
            gamma = self.parent_hydro.gamma_solution
        elif self.transport_analysis_mode == 'no_surface':
            gamma = {q: ngsolve.CF(0) for q in range(-self.imax, self.imax + 1)}

        # add correction term if internal boundary condition was used
        if self.hydro.model_options['river_boundary_treatment'] == 'exact':
            Q = self.hydro.Q_solution
        else:
            Q = {q: ngsolve.CF(0) for q in range(-self.imax, self.imax + 1)}

        # construct the Stokes drift term
        self.stokes_transport = []
        self.stokes_transport.append(
             sum([sum([0.5 * gamma[q] * (alpha[m][q] + Q[q]*self.river_interpolant) * G4(m) for q in range(-self.imax, self.imax+1)]) for m in range(self.M)])
        )
        self.stokes_transport.append(
             sum([sum([0.5 * gamma[q] * beta[m][q] * G4(m) for q in range(-self.imax, self.imax+1)]) for m in range(self.M)])
        )

        # total time averaged transport
        self.TA_transport = []
        self.TA_transport.append(
            0.5 * np.sqrt(2) * H * sum([(alpha[m][0]+Q[0]*self.river_interpolant) * G4(m) for m in range(self.M)]) + self.stokes_transport[0]
        )
        self.TA_transport.append(
            0.5 * np.sqrt(2) * H * sum([beta[m][0] * G4(m) for m in range(self.M)]) + self.stokes_transport[1]
        )

        # instantaneous transport
        self.timed_transport = []
        self.timed_transport.append(
            lambda t: sum([sum([H * (alpha[m][i]+Q[i]*self.river_interpolant) * G4(m) * self.hydro.time_basis.evaluation_function(t, i) for i in range(-self.imax, self.imax + 1)]) for m in range(self.M)]) + \
            sum([sum([sum([gamma[i] * (alpha[m][j]+Q[j]*self.river_interpolant) * G4(m) * self.hydro.time_basis.evaluation_function(t, i) * self.hydro.time_basis.evaluation_function(t, j) for i in range(-self.imax, self.imax + 1)]) for j in range(-self.imax, self.imax + 1)]) for m in range(self.M)])
        )
        self.timed_transport.append(
            lambda t: sum([sum([H * beta[m][i] * G4(m) * self.hydro.time_basis.evaluation_function(t, i) for i in range(-self.imax, self.imax + 1)]) for m in range(self.M)]) + \
            sum([sum([sum([gamma[i] * beta[m][j] * G4(m) * self.hydro.time_basis.evaluation_function(t, i) * self.hydro.time_basis.evaluation_function(t, j) for i in range(-self.imax, self.imax + 1)]) for j in range(-self.imax, self.imax + 1)]) for m in range(self.M)])
        )

        
    def _construct_w_timed(self):
        H = self.hydro.spatial_parameters['H']
        Hx = self.hydro.spatial_parameters_grad['H'][0]
        Hy = self.hydro.spatial_parameters_grad['H'][1]

        h = self.hydro.time_basis.evaluation_function
        f = self.hydro.vertical_basis.evaluation_function
        F = self.hydro.vertical_basis.integrated_evaluation_function

        sig_h_f_term = lambda t, sig: sum(sum(
            (Hx * self.hydro.alpha_solution[m][q] / self.x_scaling + Hy * self.hydro.beta_solution[m][q] / self.x_scaling) * sig * f(sig, m) * h(t, q)
            for q in range(-self.imax, self.imax + 1)) for m in range(self.M))
        
        onepsig_hh_f_term = lambda t, sig: sum(sum(sum(
            (self.hydro.gamma_grad[i][0] * self.hydro.alpha_solution[m][j] / self.x_scaling + self.hydro.gamma_grad[i][1] * self.hydro.beta_solution[m][j] / self.x_scaling) *
            (1 + sig) * h(t, i) * h(t, j) * f(sig, m)
        for i in range(-self.imax, self.imax + 1)) for j in range(-self.imax, self.imax + 1)) for m in range(self.M))

        h_F_term = lambda t, sig: sum(sum(
            (H * self.hydro.alpha_grad[m][i][0] / self.x_scaling + Hx * self.hydro.alpha_solution[m][i] / self.x_scaling + 
            H * self.hydro.beta_grad[m][i][1] / self.x_scaling + Hy * self.hydro.beta_solution[m][i] / self.x_scaling) * h(t, i) * F(sig, m)
        for i in range(-self.imax, self.imax + 1)) for m in range(self.M))

        hh_F_term = lambda t, sig: sum(sum(sum(
            (self.hydro.gamma_solution[i] * self.hydro.alpha_grad[m][j][0] / self.x_scaling + self.hydro.gamma_grad[i][0] * self.hydro.alpha_solution[m][j] / self.x_scaling +
             self.hydro.gamma_solution[i] * self.hydro.beta_grad[m][j][1] / self.x_scaling + self.hydro.gamma_grad[i][1] * self.hydro.beta_solution[m][j] / self.x_scaling) *
             h(t, i) * h(t, j) * F(sig, m)
        for i in range(-self.imax, self.imax + 1)) for j in range(-self.imax, self.imax + 1)) for m in range(self.M))

        self.w_timed = lambda t, sig: sig_h_f_term(t, sig) + onepsig_hh_f_term(t, sig) - h_F_term(t, sig) - hh_F_term(t, sig)
    

    def _construct_w_TA(self):

        H = self.hydro.spatial_parameters['H']
        Hx = self.hydro.spatial_parameters_grad['H'][0] / self.x_scaling
        Hy = self.hydro.spatial_parameters_grad['H'][1] / self.y_scaling        

        self.w_TA_sig_fm_coefficients = [0.5 * np.sqrt(2) * (self.hydro.alpha_solution[m][0] * Hx + self.hydro.beta_solution[m][0] * Hy) for m in range(self.M)]
        self.w_TA_onepsig_fm_coefficients = [sum(0.5 * self.hydro.gamma_grad[q][0] / self.x_scaling * self.hydro.alpha_solution[m][q] + 
                                                 0.5 * self.hydro.gamma_grad[q][1] / self.y_scaling * self.hydro.beta_solution[m][q] for q in range(-self.imax, self.imax + 1)) for m in range(self.M)]
        self.w_TA_FM_coefficients = [
            self.hydro.alpha_grad[m][0][0] * H / self.x_scaling + self.hydro.beta_grad[m][0][1] * H / self.y_scaling +
            self.hydro.alpha_solution[m][0] * Hx + self.hydro.beta_solution[m][0] * Hy + 
            0.5 * sum(
                self.hydro.gamma_solution[q] * self.hydro.alpha_grad[m][q][0] / self.x_scaling + self.hydro.gamma_solution[q] * self.hydro.beta_grad[m][q][1] / self.y_scaling +
                self.hydro.gamma_grad[q][0] * self.hydro.alpha_solution[m][q] / self.x_scaling + self.hydro.gamma_grad[q][1] * self.hydro.beta_solution[m][q] / self.y_scaling
                for q in range(-self.imax, self.imax + 1)
            )
            for m in range(self.M)
        ]

        vb = self.hydro.vertical_basis
        self.w_TA = lambda sig: sum(
            self.w_TA_sig_fm_coefficients[m] * sig * vb.evaluation_function(sig, m) + 
            self.w_TA_onepsig_fm_coefficients[m] * (1 + sig) * vb.evaluation_function(sig, m) + 
            self.w_TA_FM_coefficients[m] * vb.integrated_evaluation_function(sig, m)
            for m in range(self.M))


    def _construct_TWA_velocities(self):
        """Constructs thickness-weighted-averaged velocity functions, see Klingbeil et al. (2019)."""

        if self.transport_analysis_mode == 'own_surface':
            gamma = self.hydro.gamma_solution
        elif self.transport_analysis_mode == 'parent_surface':
            gamma = self.parent_hydro.gamma_solution
        elif self.transport_analysis_mode == 'no_surface':
            gamma = {q: ngsolve.CF(0) for q in range(-self.imax, self.imax + 1)}

        if self.hydro.model_options['river_boundary_treatment'] == 'exact':
            Q = self.hydro.Q_solution
        else:
            Q = {q: ngsolve.CF(0) for q in range(-self.imax, self.imax + 1)}

        H = self.hydro.spatial_parameters['H']

        u_depth_correlation = lambda sig: H * np.sqrt(2) / 2 * \
                    sum((self.hydro.alpha_solution[m][0] + self.river_interpolant * Q[0]) * self.hydro.vertical_basis.evaluation_function(sig, m) for m in range(self.M)) + \
                    0.5 * sum(sum(gamma[i] * (self.hydro.alpha_solution[m][i] + Q[i]*self.river_interpolant) * self.hydro.vertical_basis.evaluation_function(sig, m) for i in range(-self.imax, self.imax + 1)) for m in range(self.M))

        v_depth_correlation = lambda sig: H * np.sqrt(2) / 2 * sum(self.hydro.beta_solution[m][0] * self.hydro.vertical_basis.evaluation_function(sig, m) for m in range(self.M)) + \
                                0.5 * sum(sum(gamma[i] * self.hydro.beta_solution[m][i] * self.hydro.vertical_basis.evaluation_function(sig, m) for i in range(-self.imax, self.imax + 1)) for m in range(self.M))
        
        oscillating_components = list(range(-self.imax, 0)) + list(range(1, self.imax + 1))
        u_depth_correlation_stokes = lambda sig: 0.5 * sum(sum(gamma[i] * (self.hydro.alpha_solution[m][i] + Q[i]*self.river_interpolant) * self.hydro.vertical_basis.evaluation_function(sig, m) for i in oscillating_components) for m in range(self.M))
        v_depth_correlation_stokes = lambda sig: 0.5 * sum(sum(gamma[i] * self.hydro.beta_solution[m][i] * self.hydro.vertical_basis.evaluation_function(sig, m) for i in oscillating_components) for m in range(self.M))

        D_averaged = self.hydro.spatial_parameters['H'] + gamma[0] * np.sqrt(2)/2

        self.u_TWA = lambda sig: u_depth_correlation(sig) / D_averaged
        self.v_TWA = lambda sig: v_depth_correlation(sig) / D_averaged

        self.u_TWA_stokes = lambda sig: u_depth_correlation_stokes(sig) / D_averaged
        self.v_TWA_stokes = lambda sig: v_depth_correlation_stokes(sig) / D_averaged


    def get_tilde_w_TWA_in_cross_section(self, sig_arr, h=0.01, x=None, y=None, num_horizontal_points = 101, stokes_w=False): # stokes_w indicates that only the surface is used. This is an essential part of the tide decomposition

        if (x is None and y is None) or (x is not None and y is not None):
            raise ValueError("Please provide either x or y (exclusive or)")
        elif x is not None:
            y_range = np.linspace(-0.5, 0.5, num_horizontal_points)
            x_range = np.ones_like(y_range) * x
        elif y is not None:     
            x_range = np.linspace(0, 1, num_horizontal_points)
            y_range = np.ones_like(x_range) * y

        if self.transport_analysis_mode == 'own_surface':
            gamma = self.hydro.gamma_solution
        elif self.transport_analysis_mode == 'parent_surface':
            gamma = self.parent_hydro.gamma_solution
        elif self.transport_analysis_mode == 'no_surface':
            gamma = {q: ngsolve.CF(0) for q in range(-self.imax, self.imax + 1)}

        if self.hydro.model_options['river_boundary_treatment'] == 'exact':
            Q = self.hydro.Q_solution
        else:
            Q = {q: ngsolve.CF(0) for q in range(-self.imax, self.imax + 1)}

        H_cf = self.hydro.spatial_parameters['H']
        
        til_w_TWA = np.zeros((sig_arr.shape[0], num_horizontal_points))
    
        U = np.zeros((self.M, num_horizontal_points))
        V = np.zeros((self.M, num_horizontal_points))
        Ux = np.zeros_like(U)
        Vy = np.zeros_like(V)

        for m in range(self.M):
            u_transp = 0.5 * sum(gamma[i]*(self.hydro.alpha_solution[m][i] + Q[i]*self.river_interpolant) for i in range(-self.imax, self.imax + 1)) # first add only oscillating surface components
            v_transp = 0.5 * sum(gamma[i]*self.hydro.beta_solution[m][i] for i in range(-self.imax, self.imax + 1))
            
            if not stokes_w: # then add linear component if we are computing w not only with the surface
                u_transp += H_cf * 0.5 * np.sqrt(2) * (self.hydro.alpha_solution[m][0]+Q[0]*self.river_interpolant)
                v_transp += H_cf * 0.5 * np.sqrt(2) * self.hydro.beta_solution[m][0]

            U[m, :] = evaluate_CF_range(u_transp, self.hydro.mesh, x_range, y_range)
            V[m, :] = evaluate_CF_range(v_transp, self.hydro.mesh, x_range, y_range)

            if x is not None:
                Ux[m, :] = (evaluate_CF_range(u_transp, self.hydro.mesh, x_range + 0.5 * h * np.ones_like(x_range), y_range) - 
                            evaluate_CF_range(u_transp, self.hydro.mesh, x_range - 0.5 * h * np.ones_like(x_range), y_range)) / h
                Vy[m, 1:-1] = (V[m, 2:] - V[m, :-2]) / (y_range[2] - y_range[0]) # central difference
                Vy[m, 0] = (V[m, 1] - V[m, 0]) / (y_range[1] - y_range[0])
                Vy[m, -1] = (V[m, -1] - V[m, -2]) / (y_range[1] - y_range[0])
            elif y is not None:
                Vy[m, :] = (evaluate_CF_range(v_transp, self.hydro.mesh, x_range, y_range + 0.5 * h * np.ones_like(y_range)) - 
                            evaluate_CF_range(v_transp, self.hydro.mesh, x_range, y_range - 0.5 * h * np.ones_like(y_range))) / h
                Ux[m, 1:-1] = (U[m, 2:] - U[m, :-2]) / (x_range[2] - x_range[0]) # central difference
                Ux[m, 0] = (U[m, 1] - U[m, 0]) / (x_range[1] - x_range[0])
                Ux[m, -1] = (U[m, -1] - U[m, -2]) / (x_range[1] - x_range[0])
                

        for k, sig in enumerate(np.nditer(sig_arr)):
            til_w_TWA[k, :] = sum((-Ux[m,:]/self.x_scaling-Vy[m,:]/self.y_scaling) * self.hydro.vertical_basis.integrated_evaluation_function(float(sig), m) for m in range(self.M))
            

        return til_w_TWA
        
        
    def transport_through_cross_section(self, x, num_quadrature_points=501): # for checking whether solutions follow physical principles
        if not isinstance(x, np.ndarray):
            y_range = np.linspace(-0.5, 0.5, num_quadrature_points)# cross-section from y=-0.5, to y=+0.5
            vals = evaluate_CF_range(self.TA_transport[0], self.hydro.mesh, x*np.ones_like(y_range), y_range)
            return vals[:-1].sum() * (y_range[1] - y_range[0]) * self.hydro.geometric_information['y_scaling'] # leftpoint rule
        else:
            transport = np.zeros_like(x)
            for i, xval in enumerate(np.nditer(x)):
                transport[i] = self.transport_through_cross_section(float(xval), num_quadrature_points)
            return transport
        

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


    # Basic functionality

    def set_suptitle(self, title):
        self.fig.suptitle(title)


    def save(self, filename):
        self.fig.tight_layout()
        self.fig.savefig(filename)


    def get_next_index(self):
        new_index = self.plot_counter
        ij = (new_index // self.num_figures[1], new_index % self.num_figures[1])
        return ij
    

    def show(self):
        self.fig.tight_layout()
        plt.show()


    # Fundamental plotting functions

    def add_topview_plot(self, title, exclude_ramping_zone=True, colormap_quantity=None, cmap='RdBu', center_range=True, clabel='Color [unit]', contours=True, refinement_level=3, vectorfield_quantity=None, num_arrows:tuple=(30,30), arrow_color='k', length_indication='alpha', **kwargs):
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


    def add_cross_section_plot(self, title, x, num_horizontal_points = 500, num_vertical_points = 500, colormap_quantity_function=None, clabel='Color [unit]', cmap='RdBu', center_range=True, contours=False, vectorfield_quantity_function=None, stride=5, length_indication='alpha', arrow_color='black', spacing='equal', **kwargs):
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

        H = self.hydro.spatial_parameters['H']

        depth = evaluate_CF_range(H, self.hydro.mesh, x_range, y_range)

        depth_y = np.zeros_like(depth)
        depth_y[1:-1] = (depth[2:] - depth[:-2]) / (y_range[2] - y_range[0])
        depth_y[0] = (depth[1] - depth[0]) / (y_range[1] - y_range[0])
        depth_y[-1] = (depth[-1] - depth[-2]) / (y_range[1] - y_range[0])
        depth_x = (evaluate_CF_range(H, self.hydro.mesh, x_range+0.01*np.ones_like(x_range), y_range) - 
                evaluate_CF_range(H, self.hydro.mesh, x_range-0.01*np.ones_like(x_range), y_range)) / 0.02
        depth_x /= x_scaling
        depth_y /= y_scaling

        s_grid = np.tile(y_range, (num_vertical_points, 1))
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
                
        if vectorfield_quantity_function is not None:

            V = vectorfield_quantity_function[0] if isinstance(vectorfield_quantity_function[0], np.ndarray) else evaluate_vertical_structure_at_cross_section(self.hydro.mesh, vectorfield_quantity_function[0], p1, p2, num_horizontal_points, num_vertical_points)
            W = vectorfield_quantity_function[1] if isinstance(vectorfield_quantity_function[1], np.ndarray) else evaluate_vertical_structure_at_cross_section(self.hydro.mesh, vectorfield_quantity_function[1], p1, p2, num_horizontal_points, num_vertical_points)

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

            if spacing == 'sigma':
                visual_norms = np.sqrt((V[::stride,::stride] / (y_scaling))**2 + (W[::stride,::stride] / np.amax(depth))**2) # y-dimension in km
                physical_norms = np.sqrt((V[::stride,::stride])**2 + (W[::stride,::stride])**2)
                if length_indication == 'alpha':
                    quiv = current_ax.quiver(s_grid[::stride,::stride], z_grid[::stride,::stride], (V[::stride,::stride] / (y_scaling)) / visual_norms, (W[::stride,::stride] / np.amax(depth)) / visual_norms, color=arrow_color, alpha=physical_norms/np.amax(physical_norms), pivot='mid', headlength=3, headaxislength=3)
                elif length_indication == 'none':
                    quiv = current_ax.quiver(s_grid[::stride,::stride], z_grid[::stride,::stride], (V[::stride,::stride] / (y_scaling)) / visual_norms, (W[::stride,::stride] / np.amax(depth)) / visual_norms, color=arrow_color, pivot='mid', headlength=3, headaxislength=3)
                elif length_indication == 'length':
                    quiv = current_ax.quiver(s_grid[::stride,::stride], z_grid[::stride,::stride], V[::stride,::stride] / (y_scaling), W[::stride,::stride] / np.amax(depth), color=arrow_color, pivot='mid', headlength=3, headaxislength=3)
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

        if vectorfield_quantity_function is not None:
            current_ax.set_title(f'{title}\nMaximum arrow magnitude = {np.format_float_scientific(np.amax(physical_norms),unique=False, precision=4)}')
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


    def add_2DV_plot(self, title, y, num_horizontal_points = 500, num_vertical_points = 500, colormap_quantity_function=None, clabel='Color [unit]', cmap='RdBu', center_range=True, contours=False, vectorfield_quantity_function=None, stride=5, length_indication='alpha', arrow_color='black', spacing='equal', **kwargs):
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
        depth_x = np.zeros_like(depth)
        depth_x[1:-1] = (depth[2:] - depth[:-2]) / (x_range[2] - x_range[0])
        depth_x[0] = (depth[1] - depth[0]) / (x_range[1] - x_range[0])
        depth_x[-1] = (depth[-1] - depth[-2]) / (x_range[1] - x_range[0])
        depth_y = (evaluate_CF_range(H, self.hydro.mesh, x_range, y_range + 0.01*np.ones_like(y_range)) - 
                   evaluate_CF_range(H, self.hydro.mesh, x_range, y_range - 0.01*np.ones_like(y_range))) / 0.02
        depth_x /= x_scaling
        depth_y /= y_scaling

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
                

        if vectorfield_quantity_function is not None:

            U = vectorfield_quantity_function[0] if isinstance(vectorfield_quantity_function[0], np.ndarray) else evaluate_vertical_structure_at_cross_section(self.hydro.mesh, vectorfield_quantity_function[0], p1, p2, num_horizontal_points, num_vertical_points)
            W = vectorfield_quantity_function[1] if isinstance(vectorfield_quantity_function[1], np.ndarray) else evaluate_vertical_structure_at_cross_section(self.hydro.mesh, vectorfield_quantity_function[1], p1, p2, num_horizontal_points, num_vertical_points)
            sig_range = np.linspace(-1, 0, num_vertical_points)
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
                            Wquiv[z_index, x_index] = W[corresponding_sig_index, x_index * stride] * (sig_range[corresponding_sig_index + 1] - sig_value) + W[corresponding_sig_index + 1, x_index * stride] * (sig_value - sig_range[corresponding_sig_index])
                            Wquiv[z_index, x_index] /= sig_range[corresponding_sig_index + 1] - sig_range[corresponding_sig_index]
                        else:
                            Uquiv[z_index, x_index] = U[corresponding_sig_index - 1, x_index * stride] * (sig_range[corresponding_sig_index] - sig_value) + U[corresponding_sig_index, x_index * stride] * (sig_value - sig_range[corresponding_sig_index - 1])
                            Uquiv[z_index, x_index] /= sig_range[corresponding_sig_index] - sig_range[corresponding_sig_index - 1]
                            Wquiv[z_index, x_index] = W[corresponding_sig_index - 1, x_index * stride] * (sig_range[corresponding_sig_index] - sig_value) + W[corresponding_sig_index, x_index * stride] * (sig_value - sig_range[corresponding_sig_index - 1])
                            Wquiv[z_index, x_index] /= sig_range[corresponding_sig_index] - sig_range[corresponding_sig_index - 1]

            if spacing == 'sigma':
                visual_norms = np.sqrt((U[::stride,::stride] / (x_scaling))**2 + (W[::stride,::stride] / np.amax(depth))**2) # y-dimension in km
                physical_norms = np.sqrt((U[::stride,::stride])**2 + (W[::stride,::stride])**2)
                if length_indication == 'alpha':
                    quiv = current_ax.quiver(s_grid[::stride,::stride], z_grid[::stride,::stride], (U[::stride,::stride] / (x_scaling)) / visual_norms, (W[::stride,::stride] / np.amax(depth)) / visual_norms, color=arrow_color, alpha=physical_norms/np.amax(physical_norms), pivot='mid', headlength=3, headaxislength=3)
                elif length_indication == 'none':
                    quiv = current_ax.quiver(s_grid[::stride,::stride], z_grid[::stride,::stride], (U[::stride,::stride] / (x_scaling)) / visual_norms, (W[::stride,::stride] / np.amax(depth)) / visual_norms, color=arrow_color, pivot='mid', headlength=3, headaxislength=3)
                elif length_indication == 'length':
                    quiv = current_ax.quiver(s_grid[::stride,::stride], z_grid[::stride,::stride], U[::stride,::stride] / (x_scaling), W[::stride,::stride] / np.amax(depth), color=arrow_color, pivot='mid', headlength=3, headaxislength=3)
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

        if vectorfield_quantity_function is not None:
            current_ax.set_title(f'{title}\nMaximum arrow magnitude = {np.format_float_scientific(np.amax(physical_norms),unique=False, precision=4)}')
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
        

    def add_1D_plot(self, title, x_pnts, y_pnts_list, xlabel='x-axis [unit]', ylabel='y-axis [unit]', colors=None, labels=None, linestyles=None, legend = False, **kwargs):

        if self.num_figures == (1, 1):
            current_ax = self.ax
        elif self.num_figures[0] == 1 or self.num_figures[1] == 1:
            current_ax = self.ax[self.plot_counter]
        else:
            current_ax = self.ax[self.get_next_index()]

        if legend and labels is None:
            legend = False
            print("Warning: legend was turned on without plot labels. Skipping the legend.")

        if colors is None:
            colors = [None] * len(y_pnts_list)
        if labels is None:
            labels = [None] * len(y_pnts_list)
        if linestyles is None:
            linestyles = [None] * len(y_pnts_list)
            
        current_ax.set_title(title)

        for i, y_pnts in enumerate(y_pnts_list):
            current_ax.plot(x_pnts, y_pnts, color=colors[i], label=labels[i], linestyle=linestyles[i], **kwargs)
        
        if legend:
            current_ax.legend()

        current_ax.set_xlabel(xlabel)
        current_ax.set_ylabel(ylabel)
        self.plot_counter += 1


    def add_mesh_wireframe(self, title: str=None, save: str=None, **kwargs):

        """
        Plots the computational mesh of the model simulation as a wireframe.

        Arguments:

        - title (str): title of the plot
        - save (str): name of the file this plot will be saved to; if None, then the plot is not saved.
        - **kwargs: keyword arguments for matplotlib triplot in case of triangular mesh, and matplotlib hlines/vlines in case of rectangular elements.
        
        """

        if self.num_figures == (1, 1):
            current_ax = self.ax
        elif self.num_figures[0] == 1 or self.num_figures[1] == 1:
            current_ax = self.ax[self.plot_counter]
        else:
            current_ax = self.ax[self.get_next_index()]

        if self.hydro.numerical_information['mesh_generation_method'] != 'structured_quads':
            coords = mesh_to_coordinate_array(self.hydro.mesh.ngmesh)
            triangles = mesh2d_to_triangles(self.hydro.mesh.ngmesh)
            triangulation = tri.Triangulation(coords[:, 0], coords[:, 1], triangles)

            current_ax.triplot(triangulation, **kwargs)
        else:
            x = np.linspace(0, 1, self.hydro.numerical_information['grid_size'][0] + 1)
            y = np.linspace(-0.5, 0.5, self.hydro.numerical_information['grid_size'][1] + 1)

            current_ax.hlines(y, xmin=0, xmax=1, **kwargs)
            current_ax.vlines(x, ymin=-0.5, ymax=0.5, **kwargs)

        if title:
            current_ax.set_title(title)

        self.plot_counter += 1
        

    ## Special often used cases

    def add_cross_section_TWA_circulation(self, x, title=None, plot_u=True, gridsize=151, **kwargs):
        
        if title is None:
            title = f'Transport circulation in lateral cross section at x = {x}'
        
        x_scaling = self.hydro.geometric_information['x_scaling']
        y_scaling = self.hydro.geometric_information['y_scaling']
        
        p1 = np.array([x, -0.5])
        p2 = np.array([x, 0.5])
        x_range = np.linspace(p1[0], p2[0], gridsize)
        y_range = np.linspace(p1[1], p2[1], gridsize)

        H = self.hydro.spatial_parameters['H']

        depth = evaluate_CF_range(H, self.hydro.mesh, x_range, y_range)

        depth_y = np.zeros_like(depth)
        depth_y[1:-1] = (depth[2:] - depth[:-2]) / (y_range[2] - y_range[0])
        depth_y[0] = (depth[1] - depth[0]) / (y_range[1] - y_range[0])
        depth_y[-1] = (depth[-1] - depth[-2]) / (y_range[1] - y_range[0])
        depth_x = (evaluate_CF_range(H, self.hydro.mesh, x_range+0.01*np.ones_like(x_range), y_range) - 
                evaluate_CF_range(H, self.hydro.mesh, x_range-0.01*np.ones_like(x_range), y_range)) / 0.02
        depth_x /= x_scaling
        depth_y /= y_scaling

        v_func = self.pp.v_TWA
        u_func = self.pp.u_TWA

        V = evaluate_vertical_structure_at_cross_section(self.hydro.mesh, v_func, p1, p2, gridsize, gridsize)
        U = evaluate_vertical_structure_at_cross_section(self.hydro.mesh, u_func, p1, p2, gridsize, gridsize)

        colormap_QF = U if plot_u else None
        
        sig_range = np.linspace(-1, 0, gridsize)
        W_TWA = self.pp.get_tilde_w_TWA_in_cross_section(sig_range, num_horizontal_points=gridsize, x=x, stokes_w=False)
        for k, sig in enumerate(np.nditer(sig_range)):
            W_TWA[k, :] += sig * (depth_x * U[k, :] + depth_y * V[k, :]) # correction for different coordinate system


        self.add_cross_section_plot(title, x, gridsize, gridsize, colormap_quantity_function=colormap_QF, clabel='Axial transport velocity [m/s]', vectorfield_quantity_function=(V, W_TWA), stride=(gridsize-1)//30, **kwargs)


    def add_cross_section_stokes_circulation(self, x, return_flow_pp: PostProcessing, title=None, plot_u=True, gridsize=151, **kwargs):

        if title is None:
            title = f'Transport circulation in lateral cross section at x = {x}'
        
        x_scaling = self.hydro.geometric_information['x_scaling']
        y_scaling = self.hydro.geometric_information['y_scaling']
        
        p1 = np.array([x, -0.5])
        p2 = np.array([x, 0.5])
        x_range = np.linspace(p1[0], p2[0], gridsize)
        y_range = np.linspace(p1[1], p2[1], gridsize)

        H = self.hydro.spatial_parameters['H']

        depth = evaluate_CF_range(H, self.hydro.mesh, x_range, y_range)

        depth_y = np.zeros_like(depth)
        depth_y[1:-1] = (depth[2:] - depth[:-2]) / (y_range[2] - y_range[0])
        depth_y[0] = (depth[1] - depth[0]) / (y_range[1] - y_range[0])
        depth_y[-1] = (depth[-1] - depth[-2]) / (y_range[1] - y_range[0])
        depth_x = (evaluate_CF_range(H, self.hydro.mesh, x_range+0.01*np.ones_like(x_range), y_range) - 
                evaluate_CF_range(H, self.hydro.mesh, x_range-0.01*np.ones_like(x_range), y_range)) / 0.02
        depth_x /= x_scaling
        depth_y /= y_scaling

        v_func = lambda sig: self.pp.v_TWA_stokes(sig) + return_flow_pp.v_TWA(sig)
        u_func = lambda sig: self.pp.u_TWA_stokes(sig) + return_flow_pp.u_TWA(sig)

        V = evaluate_vertical_structure_at_cross_section(self.hydro.mesh, v_func, p1, p2, gridsize, gridsize)
        U = evaluate_vertical_structure_at_cross_section(self.hydro.mesh, u_func, p1, p2, gridsize, gridsize)

        colormap_QF = U if plot_u else None
        
        sig_range = np.linspace(-1, 0, gridsize)
        W_TWA_stokes = self.pp.get_tilde_w_TWA_in_cross_section(sig_range, num_horizontal_points=gridsize, x=x, stokes_w=True)
        W_TWA_rf = return_flow_pp.get_tilde_w_TWA_in_cross_section(sig_range, num_horizontal_points=gridsize, x=x, stokes_w=False)
        W_TWA = W_TWA_stokes + W_TWA_rf
        for k, sig in enumerate(np.nditer(sig_range)):
            W_TWA[k, :] += sig * (depth_x * U[k, :] + depth_y * V[k, :]) # correction for different coordinate system


        self.add_cross_section_plot(title, x, gridsize, gridsize, colormap_quantity_function=colormap_QF, clabel='Axial transport velocity [m/s]', vectorfield_quantity_function=(V, W_TWA), stride=(gridsize-1)//30, **kwargs)


    def add_2DV_TWA_circulation(self, y, title=None, plot_u=True, gridsize=151, **kwargs):
        
        if title is None:
            title = f'Transport circulation in axial cross section at y = {y}'
        
        x_scaling = self.hydro.geometric_information['x_scaling']
        y_scaling = self.hydro.geometric_information['y_scaling']
        
        p1 = np.array([0, y])
        p2 = np.array([1, y])
        x_range = np.linspace(p1[0], p2[0], gridsize)
        y_range = np.linspace(p1[1], p2[1], gridsize)

        H = self.hydro.spatial_parameters['H']

        depth = evaluate_CF_range(H, self.hydro.mesh, x_range, y_range)
        depth_x = np.zeros_like(depth)
        depth_x[1:-1] = (depth[2:] - depth[:-2]) / (x_range[2] - x_range[0])
        depth_x[0] = (depth[1] - depth[0]) / (x_range[1] - x_range[0])
        depth_x[-1] = (depth[-1] - depth[-2]) / (x_range[1] - x_range[0])
        depth_y = (evaluate_CF_range(H, self.hydro.mesh, x_range, y_range + 0.01*np.ones_like(y_range)) - 
                   evaluate_CF_range(H, self.hydro.mesh, x_range, y_range - 0.01*np.ones_like(y_range))) / 0.02
        depth_x /= x_scaling
        depth_y /= y_scaling

        v_func = self.pp.v_TWA
        u_func = self.pp.u_TWA

        V = evaluate_vertical_structure_at_cross_section(self.hydro.mesh, v_func, p1, p2, gridsize, gridsize)
        U = evaluate_vertical_structure_at_cross_section(self.hydro.mesh, u_func, p1, p2, gridsize, gridsize)

        colormap_QF = U if plot_u else None

        sig_range = np.linspace(-1, 0, gridsize)
        W_TWA = self.pp.get_tilde_w_TWA_in_cross_section(sig_range, num_horizontal_points=gridsize, y=y, stokes_w=False)
        for k, sig in enumerate(np.nditer(sig_range)):
            W_TWA[k, :] += sig * (depth_x * U[k, :] + depth_y * V[k, :]) # correction for different coordinate system


        self.add_2DV_plot(title, y, gridsize, gridsize, colormap_quantity_function=colormap_QF, clabel='Axial transport velocity [m/s]', vectorfield_quantity_function=(U, W_TWA), stride=(gridsize-1)//30, **kwargs)

    
    def add_2DV_stokes_circulation(self, y, return_flow_pp: PostProcessing, title=None, plot_u=True, gridsize=151, **kwargs):
        if title is None:
            title = f'Transport circulation in axial cross section at y = {y}'
        
        x_scaling = self.hydro.geometric_information['x_scaling']
        y_scaling = self.hydro.geometric_information['y_scaling']
        
        p1 = np.array([0, y])
        p2 = np.array([1, y])
        x_range = np.linspace(p1[0], p2[0], gridsize)
        y_range = np.linspace(p1[1], p2[1], gridsize)

        H = self.hydro.spatial_parameters['H']

        depth = evaluate_CF_range(H, self.hydro.mesh, x_range, y_range)
        depth_x = np.zeros_like(depth)
        depth_x[1:-1] = (depth[2:] - depth[:-2]) / (x_range[2] - x_range[0])
        depth_x[0] = (depth[1] - depth[0]) / (x_range[1] - x_range[0])
        depth_x[-1] = (depth[-1] - depth[-2]) / (x_range[1] - x_range[0])
        depth_y = (evaluate_CF_range(H, self.hydro.mesh, x_range, y_range + 0.01*np.ones_like(y_range)) - 
                   evaluate_CF_range(H, self.hydro.mesh, x_range, y_range - 0.01*np.ones_like(y_range))) / 0.02
        depth_x /= x_scaling
        depth_y /= y_scaling

        v_func = lambda sig: self.pp.v_TWA_stokes(sig) + return_flow_pp.v_TWA(sig)
        u_func = lambda sig: self.pp.u_TWA_stokes(sig) + return_flow_pp.u_TWA(sig)

        V = evaluate_vertical_structure_at_cross_section(self.hydro.mesh, v_func, p1, p2, gridsize, gridsize)
        U = evaluate_vertical_structure_at_cross_section(self.hydro.mesh, u_func, p1, p2, gridsize, gridsize)

        colormap_QF = U if plot_u else None
        
        sig_range = np.linspace(-1, 0, gridsize)
        W_TWA_stokes = self.pp.get_tilde_w_TWA_in_cross_section(sig_range, num_horizontal_points=gridsize, y=y, stokes_w=True)
        W_TWA_rf = return_flow_pp.get_tilde_w_TWA_in_cross_section(sig_range, num_horizontal_points=gridsize, y=y, stokes_w=False)
        W_TWA = W_TWA_stokes + W_TWA_rf
        for k, sig in enumerate(np.nditer(sig_range)):
            W_TWA[k, :] += sig * (depth_x * U[k, :] + depth_y * V[k, :]) # correction for different coordinate system


        self.add_2DV_plot(title, y, gridsize, gridsize, colormap_quantity_function=colormap_QF, clabel='Axial transport velocity [m/s]', vectorfield_quantity_function=(U, W_TWA), stride=(gridsize-1)//30, **kwargs)

