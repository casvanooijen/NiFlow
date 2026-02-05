import ngsolve
import numpy as np

from NiFlow.truncationbasis.truncationbasis import TruncationBasis
from NiFlow.utils import *
from NiFlow.geometry.create_geometry import BOUNDARY_DICT, SEA, RIVER, WALL, WALLDOWN, WALLUP


TERMS = ['DIC_time_derivative', 'transport_divergence', 'Stokes_transport_divergence', 'MOM_time_derivative', 'MOM_time_derivative_surface_interactions',
         'momentum_advection', 'momentum_advection_surface_interactions', 'Coriolis', 'Coriolis_surface_interactions', 'barotropic_pressure_gradient',
         'barotropic_pressure_gradient_surface_interactions', 'baroclinic_pressure_gradient', 'baroclinic_pressure_gradient_surface_interactions',
         'vertical_eddy_viscosity', 'horizontal_eddy_viscosity', 'horizontal_eddy_viscosity_surface_interactions', 'tide_bc', 'river_bc']

def ngsolve_tanh(argument):
    return ngsolve.sinh(argument) / ngsolve.cosh(argument)


def construct_non_linear_weak_form(weak_form, model_options, geometric_information, numerical_information, constant_parameters, spatial_parameters, spatial_parameters_grad,
                                   time_basis, vertical_basis, alpha_trial_functions, beta_trial_functions, gamma_trial_functions,
                                   umom_test_functions, vmom_test_functions, DIC_test_functions, A_trial_functions=None, Q_trial_functions=None,
                                   sea_bc_test_functions=None, river_bc_test_functions=None, normal_alpha=None, normal_alpha_y=None,
                                   operator='full', as_forcing_list=[], forcing_instruction=None, forcing_alpha=None, forcing_beta=None, forcing_gamma=None, 
                                   forcing_Q=None, linear_form=None):
    
    constructor = WeakFormConstructor(weak_form, model_options, geometric_information, numerical_information, constant_parameters, spatial_parameters, spatial_parameters_grad,
                                      time_basis, vertical_basis, alpha_trial_functions, beta_trial_functions, gamma_trial_functions,
                                      umom_test_functions, vmom_test_functions, DIC_test_functions, A_trial_functions=A_trial_functions, Q_trial_functions=Q_trial_functions,
                                      sea_bc_test_functions=sea_bc_test_functions, river_bc_test_functions=river_bc_test_functions, 
                                      normal_alpha=normal_alpha, normal_alpha_y=normal_alpha_y,
                                      operator=operator, as_forcing_list=as_forcing_list, forcing_instruction=forcing_instruction, forcing_alpha=forcing_alpha,
                                      forcing_beta=forcing_beta, forcing_gamma=forcing_gamma, forcing_Q=forcing_Q, linear_form=linear_form)
    
    constructor.add_momentum_equation(equation='u')
    constructor.add_momentum_equation(equation='v')
    constructor.add_depth_integrated_continuity_equation()

    if model_options['sea_boundary_treatment'] == 'exact':
        delta_width = 0.05
        constructor.add_internal_sea_boundary_condition(delta_width)
    
    if model_options['river_boundary_treatment'] == 'exact':
        delta_width = 0.05
        constructor.add_internal_river_boundary_condition(delta_width)


def construct_linearised_weak_form(weak_form, model_options, geometric_information, numerical_information, constant_parameters, spatial_parameters, spatial_parameters_grad,
                                   time_basis, vertical_basis, alpha_trial_functions, alpha0, beta_trial_functions, beta0, gamma_trial_functions, gamma0,
                                   umom_test_functions, vmom_test_functions, DIC_test_functions, A_trial_functions=None, Q_trial_functions=None, Q0=None,
                                   sea_bc_test_functions=None, river_bc_test_functions=None, normal_alpha=None, normal_alpha_y=None, operator='full', oseen_linearisation=False, forcing_instruction=None):
    
    constructor = WeakFormConstructor(weak_form, model_options, geometric_information, numerical_information, constant_parameters, spatial_parameters, spatial_parameters_grad, 
                                      time_basis, vertical_basis, alpha_trial_functions, beta_trial_functions, gamma_trial_functions,
                                      umom_test_functions, vmom_test_functions, DIC_test_functions, alpha0=alpha0, beta0=beta0, gamma0=gamma0, A_trial_functions=A_trial_functions, Q_trial_functions=Q_trial_functions,
                                      sea_bc_test_functions=sea_bc_test_functions, river_bc_test_functions=river_bc_test_functions, 
                                      normal_alpha=normal_alpha, normal_alpha_y=normal_alpha_y, operator=operator,
                                      as_forcing_list=[], forcing_instruction=forcing_instruction, forcing_alpha=None, forcing_beta=None, forcing_gamma=None, forcing_Q=None, oseen_linearisation=oseen_linearisation)
    
    constructor.add_momentum_equation_linearised(equation='u', Q0=Q0)
    constructor.add_momentum_equation_linearised(equation='v', Q0=Q0)
    constructor.add_depth_integrated_continuity_equation_linearised(Q0=Q0)

    if model_options['sea_boundary_treatment'] == 'exact':
        delta_width = 0.05
        constructor.add_internal_sea_boundary_condition_linearised(dirac_delta_width=delta_width)
    
    if model_options['river_boundary_treatment'] == 'exact':
        delta_width = 0.05
        constructor.add_internal_river_boundary_condition_linearised(dirac_delta_width=delta_width, Q0=Q0)



class WeakFormConstructor(object):

    # Idea for application of non-linear weak form: make new functions for the basic weak form terms (double-product etc.), and rewrite the equation term functions with a modifier that chooses the basic function that corresponds to the operation you want to do

    def __init__(self, weak_form, model_options, geometric_information, numerical_information,
                 constant_parameters, spatial_parameters, spatial_parameters_grad, time_basis, vertical_basis,
                 alpha_trial_functions, beta_trial_functions, gamma_trial_functions,
                 umom_test_functions, vmom_test_functions, DIC_test_functions,
                 alpha0=None, beta0=None, gamma0=None,
                 A_trial_functions=None, Q_trial_functions=None,
                 sea_bc_test_functions=None, river_bc_test_functions=None,
                 normal_alpha=None, normal_alpha_y=None,
                 operator='full', as_forcing_list=[], forcing_instruction=None,
                 forcing_alpha=None, forcing_beta=None, forcing_gamma=None,forcing_Q = None,
                 oseen_linearisation=False, linear_form=None):
        
        '''as_forcing_list and include_in_LHS_list must have an empty intersection'''
        
        self.weak_form = weak_form
        self.linear_form = linear_form # used for forcing decomposition
        
        self.internal_sea_bc = (model_options['sea_boundary_treatment'] == 'exact')
        self.internal_river_bc = (model_options['river_boundary_treatment'] == 'exact')

        self.constant_parameters = constant_parameters
        self.spatial_parameters = spatial_parameters
        self.spatial_parameters_grad = spatial_parameters_grad

        self.time_basis: TruncationBasis = time_basis
        self.vertical_basis: TruncationBasis = vertical_basis

        self.geometric_information = geometric_information
        self.x_scaling = geometric_information['x_scaling']
        self.y_scaling = geometric_information['y_scaling']
        self.model_options = model_options

        self.advection_matrix = model_options['advection_influence_matrix']
        self.surface_matrix = model_options['surface_interaction_influence_matrix']
        self.surface_in_sigma = (np.any(self.surface_matrix) and self.constant_parameters['surface_epsilon'] > 1e-12)
        self.oseen_linearisation = oseen_linearisation

        self.M = numerical_information['M']
        self.imax = numerical_information['imax']

        self.alpha = alpha_trial_functions
        self.beta = beta_trial_functions
        self.gamma = gamma_trial_functions

        if alpha0 is not None and beta0 is not None and gamma0 is not None:
            self.alpha0 = alpha0
            self.beta0 = beta0
            self.gamma0 = gamma0

            # precompute gradients of these gridfunctions
        
            self.alpha0_grad = [{} for _ in range(self.M)]
            self.beta0_grad = [{} for _ in range(self.M)]
            self.gamma0_grad = {}

            for i in range(-self.imax, self.imax + 1):
                self.gamma0_grad[i] = ngsolve.grad(self.gamma0[i])
                for m in range(self.M):
                    self.alpha0_grad[m][i] = ngsolve.grad(self.alpha0[m][i])
                    self.beta0_grad[m][i] = ngsolve.grad(self.beta0[m][i])

            for i in range(-self.imax, self.imax + 1):
                self.gamma0_grad[i][0].Compile()
                self.gamma0_grad[i][1].Compile()
                for m in range(self.M):
                    self.alpha0_grad[m][i][0].Compile()
                    self.alpha0_grad[m][i][1].Compile()
                    self.beta0_grad[m][i][0].Compile()
                    self.beta0_grad[m][i][1].Compile()
        
        if operator == 'full':
            include_in_LHS_list = ['DIC_time_derivative', 'transport_divergence', 'stokes_transport',
                                   'momentum_time_derivative', 'time_derivative_surface_interactions',
                                   'momentum_advection', 'advection_surface_interactions',
                                   'coriolis', 'coriolis_surface_interactions',
                                   'barotropic_pressure_gradient', 'tide_bc', 'barotropic_pressure_gradient_surface_interactions',
                                   'baroclinic_pressure_gradient', 'baroclinic_pressure_gradient_surface_interactions',
                                   'vertical_eddy_viscosity']
            
        elif operator == 'linear_for_decomposition':
            include_in_LHS_list = ['DIC_time_derivative', 'transport_divergence', 'momentum_time_derivative', 'barotropic_pressure_gradient',
                                   'vertical_eddy_viscosity']
            
        if forcing_instruction is None:
            self.forcing_instruction = {term: {'surface': 0, 'velocity': 0} for term in TERMS} # this indicates that for every available term, both surface and velocity (if both are present) are used as unknowns of the system
            self.forcing_instruction['barotropic_pressure_gradient_surface_interactions'] = {'surface_1': 0, 'surface_2': 0} # the only term that differs from the standard format
        else:
            self.forcing_instruction = forcing_instruction    
    

        self.include_in_LHS_list = include_in_LHS_list
        self.as_forcing_list = as_forcing_list

        # Compile forcings
        self.forcing_alpha = forcing_alpha
        self.forcing_beta = forcing_beta
        self.forcing_gamma = forcing_gamma
        self.forcing_Q = forcing_Q

        self.umom_test_functions = umom_test_functions
        self.vmom_test_functions = vmom_test_functions
        self.DIC_test_functions = DIC_test_functions

        if self.internal_sea_bc:
            self.A = A_trial_functions
            self.sea_bc_test_functions = sea_bc_test_functions
        
        if self.internal_river_bc:
            self.Q = Q_trial_functions
            self.river_bc_test_functions = river_bc_test_functions

            self.normal_alpha = normal_alpha
            self.normal_alpha_y = normal_alpha_y

        self._construct_advection_ramp()
        self._construct_interpolants()


    # PRIVATE METHODS #################

    # Functions that construct coefficient functions
    def _construct_advection_ramp(self):
        """
        Makes Coefficient Function of the ramp function used to gradually increase non-linearity at the boundaries.
        
        """

        x_scaling = self.geometric_information['x_scaling']
        riverine_boundary_x = self.geometric_information['riverine_boundary_x']

        L_R_sea = self.geometric_information['L_R_sea']
        L_RA_sea = self.geometric_information['L_RA_sea']
        L_R_river = self.geometric_information['L_R_river']
        L_RA_river = self.geometric_information['L_RA_river']


        # Ramping at the seaside
        if L_R_sea > 1e-16: # Check if ramping at seaside should be applied
            ramp_sea = ngsolve.IfPos(
                -L_R_sea/x_scaling - ngsolve.x,
                ngsolve.IfPos(
                    -L_R_sea/x_scaling - L_RA_sea/x_scaling - ngsolve.x,
                    0,
                    0.5 * (1 + ngsolve_tanh((4 * (ngsolve.x + L_RA_sea/x_scaling + 0.5 * L_R_sea/x_scaling) / (L_R_sea/x_scaling)) / 
                                            (1 - (2*(ngsolve.x + L_RA_sea/x_scaling + 0.5*L_R_sea/x_scaling) / (L_R_sea/x_scaling))**2)))
                ),
                1
            )
        else: # Otherwise simple constant 1
            ramp_sea = ngsolve.CF(1)

        # Ramping at the riverside
        if L_R_river > 1e-16: # Check if ramping at riverside should be applied
            ramp_river = ngsolve.IfPos(
                        -(riverine_boundary_x + L_RA_river) / x_scaling + ngsolve.x,
                        ngsolve.IfPos(
                            -(riverine_boundary_x + L_R_river + L_RA_river) / x_scaling + ngsolve.x,
                            0,
                            0.5 * (1 + ngsolve_tanh((-4 * (ngsolve.x - L_RA_river / x_scaling - 0.5 * L_R_river / x_scaling - riverine_boundary_x / x_scaling) / 
                                                    (L_R_river / x_scaling)) / (1 - (2*(ngsolve.x- L_RA_river/x_scaling - 0.5 * L_R_river/x_scaling - riverine_boundary_x/x_scaling)/(L_R_river/x_scaling))**2)))
                        ),
                        1
                    )
        else: # Otherwise simple constant 1
            ramp_river = ngsolve.CF(1)

        self.ramp = ramp_sea * ramp_river 


    def _construct_interpolants(self):
        """Constructs interpolant functions to incorporate the internal boundary conditions"""

        # Shorthands for variables
        L_BL_sea = self.geometric_information['L_BL_sea']
        L_R_sea = self.geometric_information['L_R_sea']
        L_RA_sea = self.geometric_information['L_RA_sea']
        L_BL_river = self.geometric_information['L_BL_river']
        L_R_river = self.geometric_information['L_R_river']
        L_RA_river = self.geometric_information['L_RA_river']

        x_scaling = self.geometric_information['x_scaling']
        riverine_boundary_x = self.geometric_information['riverine_boundary_x']

        # Construct interpolant functions
        self.sea_interpolant = ((riverine_boundary_x / x_scaling) + (L_BL_river/x_scaling) + (L_R_river/x_scaling) + (L_RA_river/x_scaling) - ngsolve.x) / \
                                ((riverine_boundary_x + L_BL_river + L_R_river + L_RA_river + L_BL_sea + L_R_sea + L_RA_sea) / x_scaling)
        self.sea_interpolant_x = -1 / ((riverine_boundary_x + L_BL_river + L_R_river + L_RA_river + L_BL_sea + L_R_sea + L_RA_sea) / x_scaling) / x_scaling # x-gradients always need to be scaled by the scale factor in the weak forms, so we divide by x_scaling here

        self.river_interpolant = ((L_BL_sea/x_scaling) + (L_R_sea/x_scaling) + (L_RA_sea/x_scaling) + ngsolve.x) / \
                            ((riverine_boundary_x + L_BL_river + L_R_river + L_RA_river + L_BL_sea + L_R_sea + L_RA_sea) / x_scaling)
        self.river_interpolant_x = 1 / ((riverine_boundary_x + L_BL_river + L_R_river + L_RA_river + L_BL_sea + L_R_sea + L_RA_sea) / x_scaling) / x_scaling

        # Construct the functions that make sure the computational boundary conditions are actually satisfied
        if self.internal_sea_bc:
            self.sea_bc_trial_functions = {l: self.A[l] * self.sea_interpolant for l in range(-self.imax, self.imax + 1)}
            self.sea_bc_trial_functions_x = {l: self.A[l] * self.sea_interpolant_x for l in range(-self.imax, self.imax + 1)}

        if self.internal_river_bc:
            self.river_bc_trial_functions = [{0: self.Q[0] * self.river_interpolant} for m in range(self.M)]
            self.river_bc_trial_functions_x = [{0: self.Q[0] * self.river_interpolant_x} for m in range(self.M)]
            self.river_bc_trial_functions_y = [{0: 0} for m in range(self.M)] # divide by y-scale factor since y-derivatives need to be scaled by y_scaling

            if self.forcing_alpha is not None and self.forcing_Q is not None:
                self.forcing_river_bc_trial_functions = [[{0: self.forcing_Q[k][0] * self.river_interpolant} for m in range(self.M)] for k in range(len(self.forcing_alpha))]
                self.forcing_river_bc_trial_functions_x = [[{0: self.forcing_Q[k][0] * self.river_interpolant_x} for m in range(self.M)] for k in range(len(self.forcing_alpha))]
                self.forcing_river_bc_trial_functions_y = [[{0: 0} for m in range(self.M)] for k in range(len(self.forcing_alpha))] # divide by y-scale factor since y-derivatives need to be scaled by y_scaling


    # PUBLIC METHODS ###############################

    # Basic terms

    def add_time_derivative(self, factor, trial_function_cos, trial_function_sin, test_function_cos, test_function_sin, side='lhs'): 
        if side == 'lhs':       
            self.weak_form += factor * trial_function_cos * test_function_sin * ngsolve.dx
            self.weak_form += -factor * trial_function_sin * test_function_cos * ngsolve.dx
        elif side == 'rhs':
            self.linear_form += -factor * trial_function_cos * test_function_sin * ngsolve.dx # different sign because the term is moved from the lhs to the rhs
            self.linear_form += factor * trial_function_sin * test_function_cos * ngsolve.dx


    def add_horizontal_derivative(self, factor, direction, trial_function, test_function, side='lhs'):
        scale_factor = 1/self.x_scaling if direction == 0 else 1/self.y_scaling
        if side == 'lhs':
            self.weak_form += scale_factor * factor * ngsolve.grad(trial_function)[direction] * test_function * ngsolve.dx
        elif side == 'rhs':
            self.linear_form += -scale_factor * factor * ngsolve.grad(trial_function)[direction] * test_function * ngsolve.dx


    def add_double_product(self, factor, trial_function_1, trial_function_2, test_function, side='lhs'):
        if side == 'lhs':
            self.weak_form += factor * trial_function_1 * trial_function_2 * test_function * ngsolve.dx
        elif side == 'rhs':
            self.linear_form += -factor * trial_function_1 * trial_function_2 * test_function * ngsolve.dx


    def add_double_product_linearised(self, factor, trial_function_1, trial_function_1_previous, trial_function_2, trial_function_2_previous, test_function):
        self.weak_form += factor * (trial_function_1 * trial_function_2_previous + trial_function_1_previous * trial_function_2) * test_function * ngsolve.dx


    def add_double_product_horizontal_derivative(self, factor, direction, trial_function_1, trial_function_2, test_function, side='lhs'):
        scale_factor = 1/self.x_scaling if direction == 0 else 1/self.y_scaling
        if side == 'lhs':
            self.weak_form += scale_factor * factor * trial_function_1 * ngsolve.grad(trial_function_2)[direction] * test_function * ngsolve.dx
        elif side == 'rhs':
            self.linear_form += -scale_factor * factor * trial_function_1 * ngsolve.grad(trial_function_2)[direction] * test_function * ngsolve.dx


    def add_double_product_horizontal_derivative_linearised(self, factor, direction, trial_function_1, trial_function_1_previous, trial_function_2, trial_function_2_previous, trial_function_2_previous_grad, test_function):
        scale_factor = 1/self.x_scaling if direction == 0 else 1/self.y_scaling
        self.weak_form += scale_factor * factor * (trial_function_1_previous * ngsolve.grad(trial_function_2)[direction]) * test_function * ngsolve.dx
        if not self.oseen_linearisation:
            self.weak_form += scale_factor * factor * (trial_function_1 * trial_function_2_previous_grad[direction]) * test_function * ngsolve.dx


    def add_double_product_horizontal_second_derivative(self, factor, direction, trial_function_1, trial_function_2, test_function, side='lhs'):
        scale_factor = 1/self.x_scaling if direction == 0 else 1/self.y_scaling
        if side == 'lhs':
            self.weak_form += scale_factor**2 * factor * trial_function_1 * ngsolve.grad(trial_function_2)[direction] * ngsolve.grad(test_function)[direction] * ngsolve.dx
        else:
            self.linear_form += -scale_factor**2 * factor * trial_function_1 * ngsolve.grad(trial_function_2)[direction] * ngsolve.grad(test_function)[direction] * ngsolve.dx


    def add_double_product_horizontal_second_derivative_linearised(self, factor, direction, trial_function_1, trial_function_1_previous, trial_function_2, trial_function_2_previous, trial_function_2_previous_grad, test_function):
        scale_factor = 1/self.x_scaling if direction == 0 else 1/self.y_scaling
        self.weak_form += scale_factor**2 * factor * trial_function_1_previous * ngsolve.grad(trial_function_2)[direction] * ngsolve.grad(test_function)[direction] * ngsolve.dx
        if not self.oseen_linearisation:
            self.weak_form += scale_factor**2 * factor * trial_function_1 * trial_function_2_previous_grad[direction] * ngsolve.grad(test_function)[direction] * ngsolve.dx


    def add_triple_product(self, factor, trial_function_1, trial_function_2, trial_function_3, test_function, side='lhs'):
        if side == 'lhs':
            self.weak_form += factor * (trial_function_1 * trial_function_2 * trial_function_3) * test_function * ngsolve.dx
        elif side == 'rhs':
            self.linear_form += -factor * (trial_function_1 * trial_function_2 * trial_function_3) * test_function * ngsolve.dx
    

    def add_triple_product_linearised(self, factor, trial_function_1, trial_function_1_previous, trial_function_2, trial_function_2_previous, trial_function_3, trial_function_3_previous, test_function):
        self.weak_form += factor * (trial_function_1 * trial_function_2_previous * trial_function_3_previous + 
                                    trial_function_1_previous * trial_function_2 * trial_function_3_previous + 
                                    trial_function_1_previous * trial_function_2_previous * trial_function_3) * test_function * ngsolve.dx


    def add_triple_product_horizontal_derivative(self, factor, direction, trial_function_1, trial_function_2, trial_function_3, test_function, side='lhs'):
        scale_factor = 1/self.x_scaling if direction == 0 else 1/self.y_scaling
        if side == 'lhs':
            self.weak_form += scale_factor * factor * (trial_function_1 * trial_function_2 * ngsolve.grad(trial_function_3)[direction]) * test_function * ngsolve.dx
        elif side == 'rhs':
            self.linear_form += -scale_factor * factor * (trial_function_1 * trial_function_2 * ngsolve.grad(trial_function_3)[direction]) * test_function * ngsolve.dx


    def add_triple_product_horizontal_derivative_linearised(self, factor, direction, trial_function_1, trial_function_1_previous, trial_function_2, trial_function_2_previous, trial_function_3, trial_function_3_previous, trial_function_3_previous_grad, test_function):
        scale_factor = 1/self.x_scaling if direction == 0 else 1/self.y_scaling
        self.weak_form += scale_factor * factor * (trial_function_1_previous * trial_function_2_previous * ngsolve.grad(trial_function_3)[direction]) * test_function * ngsolve.dx
        if not self.oseen_linearisation:
            self.weak_form += scale_factor * factor * (trial_function_1 * trial_function_2_previous * trial_function_3_previous_grad[direction] + 
                                                       trial_function_1_previous * trial_function_2 * trial_function_3_previous_grad[direction]) * test_function * ngsolve.dx
        

    def add_forcing(self, forcing_cf, test_function, side='lhs'):
        """Can also be used for terms like Coriolis and vertical eddy viscosity: terms of the form (u, v) or (forcing, v). This function just adds a function times the test function and integrates."""
        if side == 'lhs':
            self.weak_form += forcing_cf * test_function * ngsolve.dx
        elif side == 'rhs':
            self.linear_form += -forcing_cf * test_function * ngsolve.dx


    def add_horizontal_second_derivative(self, factor, direction, trial_function, test_function, side='lhs'):
        scale_factor = 1/self.x_scaling if direction == 0 else 1/self.y_scaling
        if side == 'lhs':
            self.weak_form += scale_factor**2 * factor * ngsolve.grad(trial_function)[direction] * ngsolve.grad(test_function)[direction] * ngsolve.dx # not inner product of gradients, because geometric scaling requires x- and y-second derivative to be treated differently
        elif side == 'rhs':
            self.linear_form += -scale_factor**2 * factor * ngsolve.grad(trial_function)[direction] * ngsolve.grad(test_function)[direction] * ngsolve.dx


    def add_boundary_forcing(self, forcing_cf, test_function, boundary_name, side='lhs'):
        if side == 'lhs':
            self.weak_form += forcing_cf * test_function * ngsolve.ds(boundary_name)
        elif side == 'rhs':
            self.linear_form += -forcing_cf * test_function * ngsolve.ds(boundary_name)

    # More specific single terms: e.g. Coriolis, baroclinic, barotropic, transport divergence, etc.

    # terms for depth-integrated continuity equation ###############################################

    def add_transport_divergence(self, l: int, forcing={'surface': 0, 'velocity': 0}):
        """
        """

        alpha = self.forcing_alpha[forcing['velocity'] - 1] if forcing['velocity'] > 0 else self.alpha
        beta = self.forcing_beta[forcing['velocity'] - 1] if forcing['velocity'] > 0 else self.beta
        side = 'rhs' if forcing['velocity'] > 0 else 'lhs'

        if self.internal_river_bc:
            river_bc_coef = self.forcing_river_bc_trial_functions[forcing['velocity'] - 1] if forcing['velocity'] > 0 else self.river_bc_trial_functions
            river_bc_coef_x = self.forcing_river_bc_trial_functions_x[forcing['velocity'] - 1] if forcing['velocity'] > 0 else self.river_bc_trial_functions_x

        G4 = self.vertical_basis.tensor_dict['G4']
        H = self.spatial_parameters['H'].Compile()
        Hx = self.spatial_parameters_grad['H'][0].Compile() / self.x_scaling
        Hy = self.spatial_parameters_grad['H'][1].Compile() / self.y_scaling

        for m in range(self.M):
            self.add_forcing(0.5 * G4(m) * Hx * alpha[m][l], self.DIC_test_functions[l], side)
            self.add_forcing(0.5 * G4(m) * Hy * beta[m][l], self.DIC_test_functions[l], side)
            self.add_horizontal_derivative(0.5 * G4(m) * H, 0, alpha[m][l], self.DIC_test_functions[l], side)
            self.add_horizontal_derivative(0.5 * G4(m) * H, 1, beta[m][l], self.DIC_test_functions[l], side)

            if self.internal_river_bc and l == 0:
                self.add_forcing(0.5 * G4(m) * Hx * river_bc_coef[m][l], self.DIC_test_functions[l], side)
                self.add_forcing(0.5 * G4(m) * H * river_bc_coef_x[m][l], self.DIC_test_functions[l], side)


    def add_stokes_transport(self, l: int, forcing={'surface': 0, 'velocity': 0}):
        """
        """

        alpha = self.forcing_alpha[forcing['velocity'] - 1] if forcing['velocity'] > 0 else self.alpha
        beta = self.forcing_beta[forcing['velocity'] - 1] if forcing['velocity'] > 0 else self.beta
        gamma = self.forcing_gamma[forcing['surface'] - 1] if forcing['surface'] > 0 else self.gamma
        side = 'rhs' if (forcing['surface'] > 0 and forcing['velocity'] > 0) else 'lhs'

        if self.internal_river_bc:
            river_bc_coef = self.forcing_river_bc_trial_functions[forcing['velocity'] - 1] if forcing['velocity'] > 0 else self.river_bc_trial_functions
            river_bc_coef_x = self.forcing_river_bc_trial_functions_x[forcing['velocity'] - 1] if forcing['velocity'] > 0 else self.river_bc_trial_functions_x

        H3 = self.time_basis.tensor_dict['H3']
        G4 = self.vertical_basis.tensor_dict['G4']
        eps = self.constant_parameters['surface_epsilon']

        for m in range(self.M):
            for i in range(-self.imax, self.imax + 1):
                for j in range(-self.imax, self.imax + 1):
                    if self.surface_matrix[abs(l), abs(i)] and self.surface_matrix[abs(l), abs(j)]:
                        self.add_double_product_horizontal_derivative(eps * self.ramp * H3(i, j, l) * G4(m), 0, alpha[m][j], gamma[i], self.DIC_test_functions[l], side) # product rule: (gamma * alpha)_x = gamma_x * alpha + gamma * alpha_x
                        self.add_double_product_horizontal_derivative(eps * self.ramp * H3(i, j, l) * G4(m), 0, gamma[i], alpha[m][j], self.DIC_test_functions[l], side)

                        self.add_double_product_horizontal_derivative(eps * self.ramp * H3(i, j, l) * G4(m), 1, beta[m][j], gamma[i], self.DIC_test_functions[l], side)
                        self.add_double_product_horizontal_derivative(eps * self.ramp * H3(i, j, l) * G4(m), 1, gamma[i], beta[m][j], self.DIC_test_functions[l], side)

                        if self.internal_river_bc and j == 0:
                            self.add_double_product_horizontal_derivative(eps * self.ramp * H3(i, j, l) * G4(m), 0, river_bc_coef[m][j], gamma[i], self.DIC_test_functions[l], side)
                            self.add_double_product(eps * self.ramp * H3(i, j, l) * G4(m), gamma[i], river_bc_coef_x[m][j], self.DIC_test_functions[l], side)


    def add_stokes_transport_linearised(self, l: int, Q0=None):
        H3 = self.time_basis.tensor_dict['H3']
        G4 = self.vertical_basis.tensor_dict['G4']
        eps = self.constant_parameters['surface_epsilon']
        
        if self.internal_river_bc:
            # river_bc_coef_0 = [{0: Q0[0] * self.normal_alpha[m] * self.river_interpolant} for m in range(self.M)]
            # river_bc_coef_0_x = [{0: Q0[0] * self.normal_alpha[m] * self.river_interpolant_x} for m in range(self.M)]
            river_bc_coef_0 = [{0: Q0[0] * self.river_interpolant} for m in range(self.M)]
            river_bc_coef_0_x = [{0: Q0[0] * self.river_interpolant_x} for m in range(self.M)]

        for m in range(self.M):
            for i in range(-self.imax, self.imax + 1):
                for j in range(-self.imax, self.imax + 1):
                    if self.surface_matrix[abs(l), abs(i)] and self.surface_matrix[abs(l), abs(j)]:
                        self.add_double_product_horizontal_derivative_linearised(eps * self.ramp * H3(i, j, l) * G4(m), 0, self.alpha[m][j], self.alpha0[m][j], self.gamma[i], self.gamma0[i], self.gamma0_grad[i], self.DIC_test_functions[l]) # product rule: (gamma * alpha)_x = gamma_x * alpha + gamma * alpha_x
                        self.add_double_product_horizontal_derivative_linearised(eps * self.ramp * H3(i, j, l) * G4(m), 0, self.gamma[i], self.gamma0[i], self.alpha[m][j], self.alpha0[m][j], self.alpha0_grad[m][j], self.DIC_test_functions[l])

                        self.add_double_product_horizontal_derivative_linearised(eps * self.ramp * H3(i, j, l) * G4(m), 1, self.beta[m][j], self.beta0[m][j], self.gamma[i], self.gamma0[i], self.gamma0_grad[i], self.DIC_test_functions[l])
                        self.add_double_product_horizontal_derivative_linearised(eps * self.ramp * H3(i, j, l) * G4(m), 1, self.gamma[i], self.gamma0[i], self.beta[m][j], self.beta0[m][j], self.beta0_grad[m][j], self.DIC_test_functions[l])

                        if self.internal_river_bc and j == 0:
                            self.add_double_product_horizontal_derivative_linearised(eps * self.ramp * H3(i, j, l) * G4(m), 0, self.river_bc_trial_functions[m][j], river_bc_coef_0[m][j], self.gamma[i], self.gamma0[i], self.gamma0_grad[i], self.DIC_test_functions[l])
                            self.add_double_product_linearised(eps * self.ramp * H3(i, j, l) * G4(m), self.gamma[i], self.gamma0[i], self.river_bc_trial_functions_x[m][j], river_bc_coef_0_x[m][j], self.DIC_test_functions[l])


    def add_DIC_time_derivative(self, l: int, forcing={'surface': 0, 'velocity': 0}):
        """l should be a positive integer!!"""

        gamma = self.forcing_gamma[forcing['surface'] - 1] if forcing['surface'] > 0 else self.gamma
        side = 'rhs' if forcing['surface'] > 0 else 'lhs'

        sigma = self.constant_parameters['sigma']
        self.add_time_derivative(sigma * np.pi * l, gamma[l], gamma[-l], self.DIC_test_functions[l], self.DIC_test_functions[-l], side)



    # terms for momentum equations ###################################

    def add_momentum_time_derivative(self, p: int, l: int, equation = 'u', forcing={'surface': 0, 'velocity': 0}):
        """l must be a positive integer!"""

        alpha = self.forcing_alpha[forcing['velocity'] - 1] if forcing['velocity'] > 0 else self.alpha
        beta = self.forcing_beta[forcing['velocity'] - 1] if forcing['velocity'] > 0 else self.beta
        side = 'rhs' if forcing['velocity'] > 0 else 'lhs'

        sigma = self.constant_parameters['sigma']
        H = self.spatial_parameters['H'].Compile()
        proj_coef = self.vertical_basis.inner_product(p, p) * np.pi * l

        flow_variable = alpha if equation == 'u' else beta
        test_function = self.umom_test_functions if equation == 'u' else self.vmom_test_functions

        self.add_time_derivative(proj_coef * H * sigma, flow_variable[p][l], flow_variable[p][-l], test_function[p][l], test_function[p][-l], side)


    def add_time_derivative_surface_interactions(self, p: int, l: int, equation = 'u', forcing={'surface': 0, 'velocity':0}):
        """l must be a positive integer!"""
        
        alpha = self.forcing_alpha[forcing['velocity'] - 1] if forcing['velocity'] > 0 else self.alpha
        beta = self.forcing_beta[forcing['velocity'] - 1] if forcing['velocity'] > 0 else self.beta
        gamma = self.forcing_gamma[forcing['surface'] - 1] if forcing['surface'] > 0 else self.gamma
        side = 'rhs' if (forcing['surface'] > 0 and forcing['velocity'] > 0) else 'lhs'

        if self.internal_river_bc:
            river_bc_coef = self.forcing_river_bc_trial_functions[forcing['velocity'] - 1] if forcing['velocity'] > 0 else self.river_bc_trial_functions

        eps = self.constant_parameters['surface_epsilon']
        sigma = self.constant_parameters['sigma']

        vertical_proj_coef = self.vertical_basis.inner_product(p, p)
        H4 = self.time_basis.tensor_dict['H4']

        flow_variable = alpha if equation == 'u' else beta
        test_function = self.umom_test_functions if equation == 'u' else self.vmom_test_functions

        for i in range(-self.imax, self.imax + 1):
            for j in range(-self.imax, self.imax + 1):
                if self.surface_matrix[l, abs(i)] and self.surface_matrix[l, abs(j)]:
                    if self.internal_river_bc and equation == 'u' and i == 0:
                        self.add_double_product(eps * self.ramp * sigma * vertical_proj_coef * H4(i,j,-l), flow_variable[p][i] + river_bc_coef[p][i], gamma[j], test_function[p][-l], side)
                        self.add_double_product(eps * self.ramp * sigma * vertical_proj_coef * H4(i,j,l), flow_variable[p][i] + river_bc_coef[p][i], gamma[j], test_function[p][l], side)
                    else:
                        self.add_double_product(eps * self.ramp * sigma * vertical_proj_coef * H4(i,j,-l), flow_variable[p][i], gamma[j], test_function[p][-l], side)
                        self.add_double_product(eps * self.ramp * sigma * vertical_proj_coef * H4(i,j,l), flow_variable[p][i], gamma[j], test_function[p][l], side)


    def add_time_derivative_surface_interactions_linearised(self, p: int, l: int, equation='u', Q0=None):
        eps = self.constant_parameters['surface_epsilon']
        sigma = self.constant_parameters['sigma']

        vertical_proj_coef = self.vertical_basis.inner_product(p, p)
        H4 = self.time_basis.tensor_dict['H4']

        flow_variable = self.alpha if equation == 'u' else self.beta
        flow_variable_0 = self.alpha0 if equation == 'u' else self.beta0
        test_function = self.umom_test_functions if equation == 'u' else self.vmom_test_functions
        
        if self.internal_river_bc:
            # river_bc_coef_0 = [{0: Q0[0] * self.normal_alpha[m] * self.river_interpolant} for m in range(self.M)]
            river_bc_coef_0 = [{0: Q0[0] * self.river_interpolant} for m in range(self.M)]

        for i in range(-self.imax, self.imax + 1):
            for j in range(-self.imax, self.imax + 1):
                if self.surface_matrix[l, abs(i)] and self.surface_matrix[l, abs(j)]:
                    if self.internal_river_bc and equation == 'u' and i == 0:
                        self.add_double_product_linearised(eps * self.ramp * sigma * vertical_proj_coef * H4(i,j,-l), flow_variable[p][i] + self.river_bc_trial_functions[p][i], flow_variable_0[p][i] + river_bc_coef_0[p][i],
                                                            self.gamma[j], self.gamma0[j], test_function[p][-l])
                        self.add_double_product_linearised(eps * self.ramp * sigma * vertical_proj_coef * H4(i,j,l), flow_variable[p][i] + self.river_bc_trial_functions[p][i], flow_variable_0[p][i] + river_bc_coef_0[p][i],
                                                            self.gamma[j], self.gamma0[j], test_function[p][l])
                    else:
                        self.add_double_product_linearised(eps * self.ramp * sigma * vertical_proj_coef * H4(i,j,-l), flow_variable[p][i], flow_variable_0[p][i], self.gamma[j], self.gamma0[j], test_function[p][-l])
                        self.add_double_product_linearised(eps * self.ramp * sigma * vertical_proj_coef * H4(i,j,l), flow_variable[p][i], flow_variable_0[p][i], self.gamma[j], self.gamma0[j], test_function[p][l])


    def add_coriolis(self, p: int, l: int, equation='u', forcing={'surface': 0, 'velocity':0}):

        alpha = self.forcing_alpha[forcing['velocity'] - 1] if forcing['velocity'] > 0 else self.alpha
        beta = self.forcing_beta[forcing['velocity'] - 1] if forcing['velocity'] > 0 else self.beta
        side = 'rhs' if forcing['velocity'] > 0 else 'lhs'

        if self.internal_river_bc:
            river_bc_coef = self.forcing_river_bc_trial_functions[forcing['velocity'] - 1] if forcing['velocity'] > 0 else self.river_bc_trial_functions

        parameter = -self.constant_parameters['f'] if equation == 'u' else self.constant_parameters['f']
        flow_variable = beta[p][l] if equation == 'u' else alpha[p][l]
        test_function = self.umom_test_functions[p][l] if equation == 'u' else self.vmom_test_functions[p][l]

        H = self.spatial_parameters['H'].Compile()
        proj_coef = 0.5 * self.vertical_basis.inner_product(p, p) # product of time- and vertical projection coefficient

        self.add_forcing(H * proj_coef * parameter * flow_variable, test_function, side)
        
        if self.internal_river_bc and equation == 'v' and l == 0:
            self.add_forcing(H * proj_coef * parameter * river_bc_coef[p][l], test_function, side)

    
    def add_coriolis_surface_interaction(self, p: int, l: int, equation='u', forcing={'surface': 0, 'velocity':0}):

        alpha = self.forcing_alpha[forcing['velocity'] - 1] if forcing['velocity'] > 0 else self.alpha
        beta = self.forcing_beta[forcing['velocity'] - 1] if forcing['velocity'] > 0 else self.beta
        gamma = self.forcing_gamma[forcing['surface'] -1] if forcing['surface'] > 0 else self.gamma
        side = 'rhs' if (forcing['surface'] > 0 and forcing['velocity'] > 0) else 'lhs'

        if self.internal_river_bc:
            river_bc_coef = self.forcing_river_bc_trial_functions[forcing['velocity'] - 1] if forcing['velocity'] > 0 else self.river_bc_trial_functions

        parameter = -self.constant_parameters['f'] if equation == 'u' else self.constant_parameters['f']
        eps = self.constant_parameters['surface_epsilon']
        flow_variable = beta if equation == 'u' else alpha
        test_function = self.umom_test_functions[p][l] if equation == 'u' else self.vmom_test_functions[p][l]

        H3 = self.time_basis.tensor_dict['H3']
        H3_is_zero = self.time_basis.tensor_dict['H3_iszero']
        vertical_proj_coef = self.vertical_basis.inner_product(p, p) # only vertical projection coefficient; time-projection coefficient changes for every combination (i, j)

        for i in range(-self.imax, self.imax + 1):
            for j in range(-self.imax, self.imax + 1):
                if self.surface_matrix[abs(l), abs(i)] and self.surface_matrix[abs(l), abs(j)] and not H3_is_zero(i,j,l):
                    self.add_double_product(H3(i, j, l) * self.ramp * vertical_proj_coef * parameter * eps, gamma[i], flow_variable[p][j], test_function, side)

                    if self.internal_river_bc and equation == 'v' and j == 0:
                        self.add_double_product(H3(i, j, l) * self.ramp * vertical_proj_coef * parameter * eps, gamma[i], river_bc_coef[p][j], test_function, side)

    
    def add_coriolis_surface_interaction_linearised(self, p: int, l: int, Q0=None, equation='u'):
        parameter = -self.constant_parameters['f'] if equation == 'u' else self.constant_parameters['f']
        eps = self.constant_parameters['surface_epsilon']
        flow_variable = self.beta if equation == 'u' else self.alpha
        flow_variable_0 = self.beta0 if equation == 'u' else self.alpha0

        test_function = self.umom_test_functions[p][l] if equation == 'u' else self.vmom_test_functions[p][l]
        
        if self.internal_river_bc:
            river_bc_coef_0 = [{0: Q0[0] * self.river_interpolant} for m in range(self.M)]

        H3 = self.time_basis.tensor_dict['H3']
        H3_is_zero = self.time_basis.tensor_dict['H3_iszero']
        vertical_proj_coef = self.vertical_basis.inner_product(p, p) # only vertical projection coefficient; time-projection coefficient changes for every combination (i, j)

        for i in range(-self.imax, self.imax + 1):
            for j in range(-self.imax, self.imax + 1):
                if self.surface_matrix[abs(l), abs(i)] and self.surface_matrix[abs(l), abs(j)] and not H3_is_zero(i,j,l):
                    self.add_double_product_linearised(H3(i, j, l) * self.ramp * vertical_proj_coef * parameter * eps, self.gamma[i], self.gamma0[i], flow_variable[p][j], flow_variable_0[p][j], test_function)

                    if self.internal_river_bc and equation == 'v' and j == 0:
                        self.add_double_product_linearised(H3(i, j, l) * self.ramp * vertical_proj_coef * parameter * eps, self.gamma[i], self.gamma0[i], self.river_bc_trial_functions[p][j], river_bc_coef_0[p][j], test_function)


    def add_barotropic_pressure_gradient(self, p: int, l: int, equation='u', forcing={'surface': 0, 'velocity':0}):
        """"""

        gamma = self.forcing_gamma[forcing['surface'] - 1] if forcing['surface'] > 0 else self.gamma
        side = 'rhs' if forcing['surface'] > 0 else 'lhs'

        direction = 0 if equation == 'u' else 1
        proj_coef = 0.5 * self.vertical_basis.tensor_dict['G4'](p)
        # proj_coef = self.vertical_basis.tensor_dict['G4'](p)
        g = self.constant_parameters['g']
        H = self.spatial_parameters['H'].Compile()
        Hx = self.spatial_parameters_grad['H'][0].Compile() / self.x_scaling
        Hy = self.spatial_parameters_grad['H'][1].Compile() / self.y_scaling

        test_function = self.umom_test_functions[p][l] if equation == 'u' else self.vmom_test_functions[p][l]

        self.add_horizontal_derivative(-g * H * proj_coef, direction, test_function, gamma[l], side)
        if equation == 'u':
            # self.weak_form += -g * side_factor * proj_coef * test_function * gamma[l] * Hx * ngsolve.dx
            self.add_forcing(-g * proj_coef * gamma[l] * Hx, test_function, side)
        else:
            # self.weak_form += -g * side_factor * proj_coef * test_function * gamma[l] * Hy * ngsolve.dx
            self.add_forcing(-g * proj_coef * gamma[l] * Hy, test_function, side)
        # -\int gH zeta * div(v)

        
    def add_natural_sea_bc(self, p: int, l: int, forcing={'surface': 0, 'velocity':0}):
        proj_coef = 0.5 * self.vertical_basis.tensor_dict['G4'](p)

        g = self.constant_parameters['g']
        H = self.spatial_parameters['H'].Compile()
        side = 'rhs' if forcing['surface'] > 0 else 'lhs'

        test_function = self.umom_test_functions[p][l]

        # -\int_{Gamma_s} gH A * u * nds (u points inward, so outward normal adds extra minus sign)
        if self.internal_sea_bc and forcing['surface'] == 0:
            self.add_boundary_forcing(-g * H * proj_coef * 1/self.x_scaling * self.A[l], test_function, BOUNDARY_DICT[SEA], side)
        elif forcing['surface'] == 0:
            if l > 0:
                A = self.constant_parameters['seaward_amplitudes'][l] * ngsolve.cos(self.constant_parameters['seaward_phases'][l-1])
            elif l == 0:
                A = self.constant_parameters['seaward_amplitudes'][0]
            else:
                A = -self.constant_parameters['seaward_amplitudes'][l] * ngsolve.sin(self.constant_parameters['seaward_phases'][l-1])
            
            self.add_boundary_forcing(-g * H * 1/self.x_scaling * proj_coef * A, test_function, BOUNDARY_DICT[SEA], side)
        else:
            self.add_boundary_forcing(-g * H * 1/self.x_scaling * proj_coef * self.forcing_gamma[forcing['surface'] - 1][l], test_function, BOUNDARY_DICT[SEA], side)
        # check if this is really only integrating along the sea boundary!!!!


    def add_barotropic_pressure_gradient_surface_interaction(self, p: int, l: int, equation='u', forcing={'surface_1': 0, 'surface_2':0}):
        """"""
        # Do not integrate the surface interaction term by parts; leads to more complicated weak form.
        gamma1 = self.forcing_gamma[forcing['surface_1'] - 1] if forcing['surface_1'] > 0 else self.gamma
        gamma2 = self.forcing_gamma[forcing['surface_2'] - 1] if forcing['surface_2'] > 0 else self.gamma
        side = 'rhs' if (forcing['surface_1'] > 0 and forcing['surface_2'] > 0) else 'lhs'

        direction = 0 if equation == 'u' else 1
        vertical_proj_coef = self.vertical_basis.tensor_dict['G4'](p)
        H3 = self.time_basis.tensor_dict['H3']
        H3_is_zero = self.time_basis.tensor_dict['H3_iszero']
        g = self.constant_parameters['g']
        eps = self.constant_parameters['surface_epsilon']

        test_function = self.umom_test_functions[p][l] if equation == 'u' else self.vmom_test_functions[p][l]

        for i in range(-self.imax, self.imax + 1):
            for j in range(-self.imax, self.imax + 1):
                if self.surface_matrix[abs(l), abs(i)] and self.surface_matrix[abs(l), abs(j)] and not H3_is_zero(i,j,l):
                    self.add_double_product_horizontal_derivative(g * eps * self.ramp * vertical_proj_coef * H3(i, j, l), direction, gamma1[i], gamma2[j], test_function, side)

    
    def add_barotropic_pressure_gradient_surface_interaction_linearised(self, p: int, l: int, equation='u'):
        """"""
        direction = 0 if equation == 'u' else 1
        vertical_proj_coef = self.vertical_basis.tensor_dict['G4'](p)
        H3 = self.time_basis.tensor_dict['H3']
        H3_is_zero = self.time_basis.tensor_dict['H3_iszero']
        g = self.constant_parameters['g']
        eps = self.constant_parameters['surface_epsilon']

        test_function = self.umom_test_functions[p][l] if equation == 'u' else self.vmom_test_functions[p][l]

        for i in range(-self.imax, self.imax + 1):
            for j in range(-self.imax, self.imax + 1):
                if self.surface_matrix[abs(l), abs(i)] and self.surface_matrix[abs(l), abs(j)] and not H3_is_zero(i,j,l):
                    self.add_double_product_horizontal_derivative_linearised(g * eps * self.ramp * vertical_proj_coef * H3(i, j, l), direction, self.gamma[i], self.gamma0[i], self.gamma[j], self.gamma0[j], self.gamma0_grad[j], test_function)


    def add_baroclinic_pressure_gradient(self, p: int, l: int, equation='u', forcing={'surface': 0, 'velocity':0}):
        """
        Only use as_forcing if you use it in the linear decomposed model; otherwise, leave at False (sign will change).
        """
        if l != 0: # prescribed density gradient cannot be time-dependent in this model so this term will have no effect on non-residual constituents
            return
        
        side = 'rhs' if forcing['surface'] > 0 or forcing['velocity'] > 0 else 'lhs' # not tested; test if we start adding baroclinic gradient
        
        density_gradient = self.spatial_parameters_grad['rho'][0].Compile() / self.spatial_parameters['rho'] / self.x_scaling if equation == 'u' \
                           else self.spatial_parameters_grad['rho'][1].Compile() / self.spatial_parameters['rho'] / self.y_scaling
        
        proj_coef = 0.5 * np.sqrt(2) * self.vertical_basis.tensor_dict['G5'](p)
        H = self.spatial_parameters['H'].Compile()

        test_function = self.umom_test_functions[p][l] if equation == 'u' else self.vmom_test_functions[p][l]

        self.add_forcing(proj_coef * H**2 * density_gradient, test_function, side)


    def add_baroclinic_pressure_gradient_surface_interaction(self, p: int, l: int, equation='u', forcing={'surface': 0, 'velocity':0}):
        """Both linear and non-linear surface interaction terms here: (H + zeta)^2 = H^2 (see add_baroclinic_pressure_gradient) + 2*H*zeta + zeta**2"""

        gamma = self.forcing_gamma[forcing['surface'] - 1] if forcing['surface'] > 0 else self.gamma
        side = 'rhs' if forcing['surface'] > 0 else 'lhs'

        density_gradient = self.spatial_parameters_grad['rho'][0].Compile() / self.spatial_parameters['rho'] / self.x_scaling if equation == 'u' \
                           else self.spatial_parameters_grad['rho'][1].Compile() / self.spatial_parameters['rho'] / self.y_scaling
        
        vertical_proj_coef = self.vertical_basis.tensor_dict['G5'](p)
        H = self.spatial_parameters['H']
        eps = self.constant_parameters['surface_epsilon'] 

        H3 = self.time_basis.tensor_dict['H3']

        test_function = self.umom_test_functions[p][l] if equation == 'u' else self.vmom_test_functions[p][l]

        # linear surface interaction 2 * H * zeta
        if self.surface_matrix[abs(l), abs(l)]: 
            self.add_forcing(vertical_proj_coef * eps * H * density_gradient * gamma[l], test_function, side)

        # non-linear surface interaction zeta**2
        for i in range(-self.imax, self.imax + 1):
            for j in range(-self.imax, self.imax + 1):
                if self.surface_matrix[abs(l), abs(i)] and self.surface_matrix[abs(l), abs(j)]:
                    self.add_double_product(eps * self.ramp * vertical_proj_coef * H3(i,j,l) * density_gradient, gamma[i], gamma[j], test_function, side)


    def add_baroclinic_pressure_gradient_surface_interaction_linearised(self, p: int, l: int, equation='u', A0=None):
        density_gradient = self.spatial_parameters_grad['rho'][0].Compile() / self.spatial_parameters['rho'] / self.x_scaling if equation == 'u' \
                           else self.spatial_parameters_grad['rho'][1].Compile() / self.spatial_parameters['rho'] / self.y_scaling
        
        vertical_proj_coef = self.vertical_basis.tensor_dict['G5'](p)
        H = self.spatial_parameters['H']
        eps = self.constant_parameters['surface_epsilon'] 

        H3 = self.time_basis.tensor_dict['H3']

        test_function = self.umom_test_functions[p][l] if equation == 'u' else self.vmom_test_functions[p][l]

        # linear surface interaction 2 * H * zeta; unchanged here in linearisation because already linear
        if self.surface_matrix[abs(l), abs(l)]: 
            self.add_forcing(vertical_proj_coef * eps * H * density_gradient * self.gamma[l], test_function)

        # non-linear surface interaction zeta**2
        for i in range(-self.imax, self.imax + 1):
            for j in range(-self.imax, self.imax + 1):
                if self.surface_matrix[abs(l), abs(i)] and self.surface_matrix[abs(l), abs(j)]:
                    self.add_double_product_linearised(eps * self.ramp * vertical_proj_coef * H3(i, j, l) * density_gradient, self.gamma[i], self.gamma0[i], self.gamma[j], self.gamma0[j], test_function)


    def add_vertical_eddy_viscosity(self, p: int, l: int, equation='u', forcing={'surface': 0, 'velocity':0}):

        alpha = self.forcing_alpha[forcing['velocity'] - 1] if forcing['velocity'] > 0 else self.alpha
        beta = self.forcing_beta[forcing['velocity'] - 1] if forcing['velocity'] > 0 else self.beta
        if self.internal_river_bc:
            river_bc_coef = self.forcing_river_bc_trial_functions[forcing['velocity'] - 1] if forcing['velocity'] > 0 else self.river_bc_trial_functions
        side = 'rhs' if forcing['velocity'] > 0 else 'lhs'

        Av = self.constant_parameters['Av']
        proj_coef = 0.5 * self.vertical_basis.tensor_dict['G3'](p, p)

        flow_variable = alpha[p][l] if equation == 'u' else beta[p][l]
        test_function = self.umom_test_functions[p][l] if equation == 'u' else self.vmom_test_functions[p][l]

        if self.model_options['veddy_viscosity_assumption'] == 'depth-scaled&constantprofile':
            self.add_forcing(-proj_coef * Av * flow_variable, test_function, side)
            if self.internal_river_bc and l == 0 and equation == 'u':
                self.add_forcing(-proj_coef * Av * river_bc_coef[p][l], test_function, side)
        elif self.model_options['veddy_viscosity_assumption'] == 'constant':
            H = self.spatial_parameters['H'].Compile()
            self.add_forcing(-proj_coef * Av * flow_variable / H, test_function, side)
            if self.internal_river_bc and l == 0 and equation == 'u':
                self.add_forcing(-proj_coef * Av * river_bc_coef[p][l] / H, test_function, side)


    def add_horizontal_eddy_viscosity(self, p: int, l: int, equation='u', forcing={'surface': 0, 'velocity': 0}):
        """"""

        proj_coef = 0.5 * self.vertical_basis.inner_product(p, p)
        Ah = self.constant_parameters['Ah']
        H = self.spatial_parameters['H'].Compile()
        Hx = self.spatial_parameters_grad['H'][0].Compile() / self.x_scaling
        Hy = self.spatial_parameters_grad['H'][1].Compile() / self.y_scaling
        side = 'rhs' if forcing['velocity'] > 0 else 'lhs'

        alpha = self.forcing_alpha[forcing['velocity'] - 1] if forcing['velocity'] > 0 else self.alpha
        beta = self.forcing_beta[forcing['velocity'] - 1] if forcing['velocity'] > 0 else self.beta
        flow_variable = alpha[p][l] if equation == 'u' else beta[p][l]
        test_function = self.umom_test_functions[p][l] if equation == 'u' else self.vmom_test_functions[p][l]
        if self.internal_river_bc:
            river_bc_coef_x = self.forcing_river_bc_trial_functions_x[forcing['velocity'] - 1] if forcing['velocity'] > 0 else self.river_bc_trial_functions_x

        self.add_horizontal_second_derivative(proj_coef * Ah * H, 0, flow_variable, test_function, side) # product rule: (H * phi)_x = Hx * phi + H * phi_x
        self.add_horizontal_derivative(proj_coef * Ah * Hx, 0, flow_variable, test_function, side)

        self.add_horizontal_second_derivative(proj_coef * Ah * H, 1, flow_variable, test_function, side)
        self.add_horizontal_derivative(proj_coef * Ah * Hy, 1, flow_variable, test_function, side)

        if self.internal_river_bc and equation == 'u' and l == 0:
            self.add_horizontal_derivative(proj_coef * Ah * H, 0, test_function, river_bc_coef_x[p][l], side) # sub test function at trial function spot because test function needs to be differentiated and trial function is analytically differentiated, so no differentiation in the assembly.
            self.add_forcing(proj_coef * Ah * Hx * river_bc_coef_x[p][l], test_function, side)

    
    def add_horizontal_eddy_viscosity_surface_interactions(self, p: int, l: int, equation='u', forcing={'surface': 0, 'velocity': 0}):
        Ah = self.constant_parameters['Ah']
        eps = self.constant_parameters['surface_epsilon']
        H3 = self.time_basis.tensor_dict['H3']
        
        alpha = self.forcing_alpha[forcing['velocity'] - 1] if forcing['velocity'] > 0 else self.alpha
        beta = self.forcing_beta[forcing['velocity'] - 1] if forcing['velocity'] > 0 else self.beta
        gamma = self.forcing_gamma[forcing['surface'] - 1] if forcing['surface'] > 0 else self.gamma
        side = 'rhs' if (forcing['velocity'] > 0 and forcing['surface'] > 0) else 'lhs'

        flow_variable = alpha if equation == 'u' else beta
        test_function = self.umom_test_functions[p][l] if equation == 'u' else self.vmom_test_functions[p][l]
        if self.internal_river_bc:
            river_bc_coef_x = self.forcing_river_bc_trial_functions_x[forcing['velocity'] - 1] if forcing['velocity'] > 0 else self.river_bc_trial_functions_x

        for i in range(-self.imax, self.imax + 1):
            for j in range(-self.imax, self.imax + 1):
                proj_coef = self.vertical_basis.inner_product(p, p) * H3(i, j, l)
                self.add_double_product_horizontal_second_derivative(eps * self.ramp * proj_coef * Ah, 0, gamma[i], flow_variable[p][j], test_function, side)
                self.add_double_product_horizontal_second_derivative(eps * self.ramp * proj_coef * Ah, 0, test_function, flow_variable[p][j], gamma[i], side)
                self.add_double_product_horizontal_second_derivative(eps * self.ramp * proj_coef * Ah, 1, gamma[i], flow_variable[p][j], test_function, side)
                self.add_double_product_horizontal_second_derivative(eps * self.ramp * proj_coef * Ah, 1, test_function, flow_variable[p][j], gamma[i], side)

                if self.internal_river_bc and equation == 'u' and j == 0:
                    self.add_double_product_horizontal_derivative(eps * self.ramp * proj_coef * Ah, 0, gamma[i], test_function, river_bc_coef_x[p][j], side)
                    self.add_double_product_horizontal_derivative(eps * self.ramp * proj_coef * Ah, 0, river_bc_coef_x[p][j], gamma[i], test_function, side)


    def add_horizontal_eddy_viscosity_surface_interactions_linearised(self, p: int, l: int, equation='u', Q0=None):
        Ah = self.constant_parameters['Ah']
        eps = self.constant_parameters['surface_epsilon']
        H3 = self.time_basis.tensor_dict['H3']

        flow_variable = self.alpha if equation == 'u' else self.beta
        flow_variable_0 = self.alpha0 if equation == 'u' else self.beta0
        flow_variable_0_grad = self.alpha0_grad if equation == 'u' else self.beta0_grad

        test_function = self.umom_test_functions[p][l] if equation == 'u' else self.vmom_test_functions[p][l]
        if self.internal_river_bc:
            river_bc_coef_0_x = [{0: Q0[0] * self.river_interpolant_x} for m in range(self.M)]
        
        for i in range(-self.imax, self.imax + 1):
            for j in range(-self.imax, self.imax + 1):
                proj_coef = self.vertical_basis.inner_product(p, p) * H3(i, j, l)
                self.add_double_product_horizontal_second_derivative_linearised(eps * self.ramp * proj_coef * Ah, 0, self.gamma[i], self.gamma0[i], flow_variable[p][j], flow_variable_0[p][j], flow_variable_0_grad[p][j], test_function)
                self.weak_form += eps * self.ramp * proj_coef * Ah * test_function * ngsolve.grad(flow_variable[p][j])[0] * self.gamma0_grad[i][0] / self.x_scaling**2 * ngsolve.dx
                self.weak_form += eps * self.ramp * proj_coef * Ah * test_function * flow_variable_0_grad[p][j][0] * ngsolve.grad(self.gamma[i])[0] / self.x_scaling**2 * ngsolve.dx
                self.add_double_product_horizontal_second_derivative_linearised(eps * self.ramp * proj_coef * Ah, 1, self.gamma[i], self.gamma0[i], flow_variable[p][j], flow_variable_0[p][j], flow_variable_0_grad[p][j], test_function)
                self.weak_form += eps * self.ramp * proj_coef * Ah * test_function * ngsolve.grad(flow_variable[p][j])[1] * self.gamma0_grad[i][1] / self.y_scaling**2 * ngsolve.dx
                self.weak_form += eps * self.ramp * proj_coef * Ah * test_function * flow_variable_0_grad[p][j][1] * ngsolve.grad(self.gamma[i])[1] / self.y_scaling**2 * ngsolve.dx

                if self.internal_river_bc and equation == 'u' and j == 0:
                    self.add_double_product_horizontal_derivative_linearised(eps * self.ramp * proj_coef * Ah, 0, self.river_bc_trial_functions_x[p][j], river_bc_coef_0_x[p][j], self.gamma[i], self.gamma0[i], self.gamma0_grad[i], test_function)
                    self.weak_form += eps * self.ramp * proj_coef * Ah * self.gamma[i] * river_bc_coef_0_x[p][j] * ngsolve.grad(test_function)[0] / self.x_scaling * ngsolve.dx
                    self.weak_form += eps * self.ramp * proj_coef * Ah * self.gamma0[i] * self.river_bc_trial_functions_x[p][j] * ngsolve.grad(test_function)[0] / self.x_scaling * ngsolve.dx        


    def add_momentum_advection(self, p: int, l: int, equation='u', forcing={'surface': 0, 'velocity':0}):
        """"""

        alpha = self.forcing_alpha[forcing['velocity'] - 1] if forcing['velocity'] > 0 else self.alpha
        beta = self.forcing_beta[forcing['velocity'] - 1] if forcing['velocity'] > 0 else self.beta
        side = 'rhs' if forcing['velocity'] > 0 else 'lhs'

        if self.internal_river_bc:
            river_bc_coef = self.forcing_river_bc_trial_functions[forcing['velocity'] - 1] if forcing['velocity'] > 0 else self.river_bc_trial_functions
            river_bc_coef_x = self.forcing_river_bc_trial_functions_x[forcing['velocity'] - 1] if forcing['velocity'] > 0 else self.river_bc_trial_functions_x
            river_bc_coef_y = self.forcing_river_bc_trial_functions_y[forcing['velocity'] - 1] if forcing['velocity'] > 0 else self.river_bc_trial_functions_y

        eps = self.constant_parameters['advection_epsilon']
        H = self.spatial_parameters['H'].Compile()
        Hx = self.spatial_parameters_grad['H'][0].Compile() / self.x_scaling
        Hy = self.spatial_parameters_grad['H'][1].Compile() / self.y_scaling

        H3 = self.time_basis.tensor_dict['H3']
        H3_is_zero = self.time_basis.tensor_dict['H3_iszero']
        G1 = self.vertical_basis.tensor_dict['G1']
        G2 = self.vertical_basis.tensor_dict['G2']

        test_function = self.umom_test_functions[p][l] if equation == 'u' else self.vmom_test_functions[p][l]
        flow_variable = alpha if equation == 'u' else beta

        
        for i in range(-self.imax, self.imax+1):
            for j in range(-self.imax, self.imax+1):
                if self.advection_matrix[abs(l), abs(i)] and self.advection_matrix[abs(l), abs(j)] and not H3_is_zero(i,j,l):
                    # do not add a zero term if the projection coefficient vanishes; this takes a lot of unnecessary computational effort
                    for m in range(self.M):
                        for n in range(self.M):
                            # longitudinal
                            self.add_double_product_horizontal_derivative(eps * self.ramp * H * H3(i,j,l) * G1(m,n,p), 0, alpha[m][i], flow_variable[n][j], test_function, side)
                            # lateral
                            self.add_double_product_horizontal_derivative(eps * self.ramp * H * H3(i,j,l) * G1(m,n,p), 1, beta[m][i], flow_variable[n][j], test_function, side)
                            # vertical
                            self.add_double_product(-eps * self.ramp * H3(i,j,l) * G2(m,n,p) * Hx, flow_variable[m][i], alpha[n][j], test_function, side)
                            self.add_double_product(-eps * self.ramp * H3(i,j,l) * G2(m,n,p) * Hy, flow_variable[m][i], beta[n][j], test_function, side)
                            self.add_double_product_horizontal_derivative(-eps * self.ramp * H3(i,j,l) * G2(m,n,p) * H, 0, flow_variable[m][i], alpha[n][j], test_function, side)
                            self.add_double_product_horizontal_derivative(-eps * self.ramp * H3(i,j,l) * G2(m,n,p) * H, 1, flow_variable[m][i], beta[n][j], test_function, side)

                            # add boundary coefficients
                            if self.internal_river_bc:
                                # longitudinal
                                if i == 0: # very ugly, but only river_bc_coef[m][0] exist. Might refactor later
                                    self.add_double_product_horizontal_derivative(eps * self.ramp * H * H3(i,j,l) * G1(m,n,p), 0, river_bc_coef[m][i], flow_variable[n][j], test_function, side)
                                if equation == 'u': # then flow_variable = alpha
                                    if j == 0:
                                        self.add_double_product(eps * self.ramp * H * H3(i,j,l) * G1(m,n,p), alpha[m][i], river_bc_coef_x[n][j], test_function, side)
                                    if i == 0 and j == 0:
                                        self.add_double_product(eps * self.ramp * H * H3(i,j,l) * G1(m,n,p), river_bc_coef[m][i], river_bc_coef_x[n][j], test_function, side)
                                # lateral; only if equation is u-momentum (lateral flow does not need the extra boundary coefficients)
                                    if j == 0:
                                        self.add_double_product(eps * self.ramp * H * H3(i,j,l) * G1(m,n,p), beta[m][i], river_bc_coef_y[n][j], test_function, side)
                                # vertical
                                if j == 0:
                                    self.add_double_product(-eps * self.ramp * H3(i,j,l) * G2(m,n,p) * Hx, flow_variable[m][i], river_bc_coef[n][j], test_function, side)
                                    self.add_double_product(-eps * self.ramp * H3(i,j,l) * G2(m,n,p) * H, flow_variable[m][i], river_bc_coef_x[n][j], test_function, side)
                                if equation == 'u':
                                    if i == 0:
                                        self.add_double_product(-eps * self.ramp * H3(i,j,l) * G2(m,n,p) * Hx, river_bc_coef[m][i], alpha[n][j], test_function, side)
                                    if i == 0 and j == 0:
                                        self.add_double_product(-eps * self.ramp * H3(i,j,l) * G2(m,n,p) * Hx, river_bc_coef[m][i], river_bc_coef[n][j], test_function, side)
                                    if i == 0:
                                        self.add_double_product(-eps * self.ramp * H3(i,j,l) * G2(m,n,p) * Hy, river_bc_coef[m][i], beta[n][j], test_function, side)
                                        self.add_double_product_horizontal_derivative(-eps * self.ramp * H3(i,j,l) * G2(m,n,p) * H, 0, river_bc_coef[m][i], alpha[n][j], test_function, side)
                                    if i == 0 and j == 0:
                                        self.add_double_product(-eps * self.ramp * H3(i,j,l) * G2(m,n,p) * H, river_bc_coef[m][i], river_bc_coef_x[n][j], test_function, side)
                                    if i == 0:
                                        self.add_double_product_horizontal_derivative(-eps * self.ramp * H3(i,j,l) * G2(m,n,p) * H, 1, river_bc_coef[m][i], beta[n][j], test_function, side)
                                

    def add_momentum_advection_linearised(self, p: int, l: int, equation='u', Q0=None):
        eps = self.constant_parameters['advection_epsilon']
        H = self.spatial_parameters['H'].Compile()
        Hx = self.spatial_parameters_grad['H'][0].Compile() / self.x_scaling
        Hy = self.spatial_parameters_grad['H'][1].Compile() / self.y_scaling

        H3 = self.time_basis.tensor_dict['H3']
        H3_is_zero = self.time_basis.tensor_dict['H3_iszero']
        G1 = self.vertical_basis.tensor_dict['G1']
        G2 = self.vertical_basis.tensor_dict['G2']

        test_function = self.umom_test_functions[p][l] if equation == 'u' else self.vmom_test_functions[p][l]
        flow_variable = self.alpha if equation == 'u' else self.beta
        flow_variable_0 = self.alpha0 if equation == 'u' else self.beta0
        flow_variable_0_grad = self.alpha0_grad if equation == 'u' else self.beta0_grad

        if self.internal_river_bc:
            # river_bc_coef_0 = [{0: Q0[0] * self.normal_alpha[m] * self.river_interpolant} for m in range(self.M)]
            # river_bc_coef_0_x = [{0: Q0[0] * self.normal_alpha[m] * self.river_interpolant_x} for m in range(self.M)]
            # river_bc_coef_0_y = [{0: Q0[0] * self.normal_alpha_y[m] * self.river_interpolant / self.y_scaling} for m in range(self.M)]
            river_bc_coef_0 = [{0: Q0[0] * self.river_interpolant} for m in range(self.M)]
            river_bc_coef_0_x = [{0: Q0[0] * self.river_interpolant_x} for m in range(self.M)]
            river_bc_coef_0_y = [{0: 0} for m in range(self.M)]

        for i in range(-self.imax, self.imax+1):
            for j in range(-self.imax, self.imax+1):
                if self.advection_matrix[abs(l), abs(i)] and self.advection_matrix[abs(l), abs(j)] and not H3_is_zero(i,j,l):
                    # do not add a zero term if the projection coefficient vanishes; this takes a lot of unnecessary computational effort
                    for m in range(self.M):
                        for n in range(self.M):
                        # longitudinal
                            self.add_double_product_horizontal_derivative_linearised(eps * self.ramp * H * H3(i,j,l) * G1(m,n,p), 0, self.alpha[m][i], self.alpha0[m][i], flow_variable[n][j], flow_variable_0[n][j], flow_variable_0_grad[n][j], test_function)
                            # lateral
                            self.add_double_product_horizontal_derivative_linearised(eps * self.ramp * H * H3(i,j,l) * G1(m,n,p), 1, self.beta[m][i], self.beta0[m][i], flow_variable[n][j], flow_variable_0[n][j], flow_variable_0_grad[n][j], test_function)
                            # vertical
                            self.add_double_product_linearised(-eps * self.ramp * H3(i,j,l) * G2(m,n,p) * Hx, flow_variable[m][i], flow_variable_0[m][i], self.alpha[n][j], self.alpha0[n][j], test_function)
                            self.add_double_product_linearised(-eps * self.ramp * H3(i,j,l) * G2(m,n,p) * Hy, flow_variable[m][i], flow_variable_0[m][i], self.beta[n][j], self.beta0[n][j], test_function)
                            self.add_double_product_horizontal_derivative_linearised(-eps * self.ramp * H3(i,j,l) * G2(m,n,p) * H, 0, flow_variable[m][i], flow_variable_0[m][i], self.alpha[n][j], self.alpha0[n][j], self.alpha0_grad[n][j], test_function)
                            self.add_double_product_horizontal_derivative_linearised(-eps * self.ramp * H3(i,j,l) * G2(m,n,p) * H, 1, flow_variable[m][i], flow_variable_0[m][i], self.beta[n][j], self.beta0[n][j], self.beta0_grad[n][j], test_function)

                            # add boundary coefficients
                            if self.internal_river_bc:
                                # longitudinal
                                if i == 0:
                                    self.add_double_product_horizontal_derivative_linearised(eps * self.ramp * H * H3(i,j,l) * G1(m,n,p), 0, self.river_bc_trial_functions[m][i], river_bc_coef_0[m][i], flow_variable[n][j], flow_variable_0[n][j], flow_variable_0_grad[n][j], test_function)
                                if equation == 'u': # then flow_variable = alpha
                                    if j == 0:
                                        self.add_double_product_linearised(eps * self.ramp * H * H3(i,j,l) * G1(m,n,p), self.alpha[m][i], self.alpha0[m][i], self.river_bc_trial_functions_x[n][j], river_bc_coef_0_x[n][j], test_function)
                                    if i == 0 and j == 0:
                                        self.add_double_product_linearised(eps * self.ramp * H * H3(i,j,l) * G1(m,n,p), self.river_bc_trial_functions[m][i], river_bc_coef_0[m][i], self.river_bc_trial_functions_x[n][j], river_bc_coef_0_x[n][j], test_function)
                                # lateral; only if equation is u-momentum (lateral flow does not need the extra boundary coefficients)
                                if j == 0:
                                    self.add_double_product_linearised(eps * self.ramp * H * H3(i,j,l) * G1(m,n,p), self.beta[m][i], self.beta0[m][i], self.river_bc_trial_functions_y[n][j], river_bc_coef_0_y[n][j], test_function)
                                # vertical
                                if j == 0:
                                    self.add_double_product_linearised(-eps * self.ramp * H3(i,j,l) * G2(m,n,p) * Hx, flow_variable[m][i], flow_variable_0[m][i], self.river_bc_trial_functions[n][j], river_bc_coef_0[n][j], test_function)
                                    self.add_double_product_linearised(-eps * self.ramp * H3(i,j,l) * G2(m,n,p) * H, flow_variable[m][i], flow_variable_0[m][i], self.river_bc_trial_functions_x[n][j], river_bc_coef_0_x[n][j], test_function)
                                if equation == 'u':
                                    if i == 0:
                                        self.add_double_product_linearised(-eps * self.ramp * H3(i,j,l) * G2(m,n,p) * Hx, self.river_bc_trial_functions[m][i], river_bc_coef_0[m][i], self.alpha[n][j], self.alpha0[n][j], test_function)
                                    if i == 0 and j == 0:
                                        self.add_double_product_linearised(-eps * self.ramp * H3(i,j,l) * G2(m,n,p) * Hx, self.river_bc_trial_functions[m][i], river_bc_coef_0[m][i], self.river_bc_trial_functions[n][j], river_bc_coef_0[n][j], test_function)
                                    if i == 0:
                                        self.add_double_product_linearised(-eps * self.ramp * H3(i,j,l) * G2(m,n,p) * Hy, self.river_bc_trial_functions[m][i], river_bc_coef_0[m][i], self.beta[n][j], self.beta0[n][j], test_function)
                                        self.add_double_product_horizontal_derivative_linearised(-eps * self.ramp * H3(i,j,l) * G2(m,n,p) * H, 0, self.river_bc_trial_functions[m][i], river_bc_coef_0[m][i], self.alpha[n][j], self.alpha0[n][j], self.alpha0_grad[n][j], test_function)
                                    if i == 0 and j == 0:
                                        self.add_double_product_linearised(-eps * self.ramp * H3(i,j,l) * G2(m,n,p) * H, self.river_bc_trial_functions[m][i], river_bc_coef_0[m][i], self.river_bc_trial_functions_x[n][j], river_bc_coef_0_x[n][j], test_function)
                                    if i == 0:
                                        self.add_double_product_horizontal_derivative_linearised(-eps * self.ramp * H3(i,j,l) * G2(m,n,p) * H, 1, self.river_bc_trial_functions[m][i], river_bc_coef_0[m][i], self.beta[n][j], self.beta0[n][j], self.beta0_grad[n][j], test_function)


    def add_advection_surface_interactions(self, p: int, l: int, equation='u', forcing={'surface': 0, 'velocity':0}):
        """Prepare for nested for-loops :)"""

        alpha = self.forcing_alpha[forcing['velocity'] - 1] if forcing['velocity'] > 0 else self.alpha
        beta = self.forcing_beta[forcing['velocity'] - 1] if forcing['velocity'] > 0 else self.beta
        gamma = self.forcing_gamma[forcing['surface'] - 1] if forcing['surface'] > 0 else self.gamma

        if self.internal_river_bc:
            river_bc_coef = self.forcing_river_bc_trial_functions[forcing['velocity'] - 1] if forcing['velocity'] > 0 else self.river_bc_trial_functions
            river_bc_coef_x = self.forcing_river_bc_trial_functions_x[forcing['velocity'] - 1] if forcing['velocity'] > 0 else self.river_bc_trial_functions_x
            river_bc_coef_y = self.forcing_river_bc_trial_functions_y[forcing['velocity'] - 1] if forcing['velocity'] > 0 else self.river_bc_trial_functions_y

            # also ugly but (hopefully) temporary solution
            for m in range(self.M):
                for l in range(-self.imax, self.imax+1):
                    if l == 0:
                        continue
                    river_bc_coef[m][l] = 0
                    river_bc_coef_x[m][l] = 0
                    river_bc_coef_y[m][l] = 0

        side = 'rhs' if (forcing['velocity'] > 0 and forcing['surface'] > 0) else 'lhs'

        surface_eps = self.constant_parameters['surface_epsilon']
        advection_eps = self.constant_parameters['advection_epsilon']
        sigma = self.constant_parameters['sigma']

        G1 = self.vertical_basis.tensor_dict['G1']
        G2 = self.vertical_basis.tensor_dict['G2']
        G7 = self.vertical_basis.tensor_dict['G7']
        H4 = self.time_basis.tensor_dict['H4']
        H5 = self.time_basis.tensor_dict['H5']

        flow_variable = alpha if equation == 'u' else beta
        test_function = self.umom_test_functions[p][l] if equation == 'u' else self.vmom_test_functions[p][l]
        
        # surface interactions with double products
        for i in range(-self.imax, self.imax + 1):
            for j in range(-self.imax, self.imax + 1):
                for m in range(self.M):
                    if self.surface_matrix[abs(l), abs(i)] and self.surface_matrix[abs(l), abs(j)] and self.advection_matrix[abs(l), abs(i)] and self.advection_matrix[abs(l), abs(j)]:
                        proj_coef = H4(i, j, l) * G7(m, p)
                        self.add_double_product(-surface_eps * advection_eps * self.ramp * proj_coef * sigma, gamma[i], flow_variable[m][j], test_function, side)
                        if self.internal_river_bc and equation == 'u' and j == 0:
                            self.add_double_product(-surface_eps * advection_eps * self.ramp * proj_coef * sigma, gamma[i], river_bc_coef[m][j], test_function, side)

        # surface interactions with triple products
        for i in range(-self.imax, self.imax + 1):
            for j in range(-self.imax, self.imax + 1):
                for k in range(-self.imax, self.imax + 1):
                    if self.surface_matrix[abs(l), abs(i)] and self.surface_matrix[abs(l), abs(j)] and self.surface_matrix[abs(l), abs(k)] and self.advection_matrix[abs(l), abs(i)] and self.advection_matrix[abs(l), abs(j)]:
                        time_proj_coef = H5(i,j,k,l)
                        for m in range(self.M):
                            for n in range(self.M): 
                                if self.internal_river_bc:
                                    # longitudinal
                                    self.add_triple_product_horizontal_derivative(surface_eps * advection_eps * self.ramp * time_proj_coef * G1(m,n,p), 0, 
                                                                                  gamma[k], alpha[m][i] + river_bc_coef[m][i], flow_variable[n][j], test_function, side)
                                    if equation == 'u':
                                        self.add_triple_product(surface_eps * advection_eps * self.ramp * time_proj_coef * G1(m,n,p), 
                                                                gamma[k], alpha[m][i] + river_bc_coef[m][i], river_bc_coef_x[n][j], test_function, side)
                                    # lateral
                                    self.add_triple_product_horizontal_derivative(surface_eps * advection_eps * self.ramp * time_proj_coef * G1(m,n,p), 1, 
                                                                                  gamma[k], beta[m][i], flow_variable[n][j], test_function, side)
                                    if equation == 'u':
                                        self.add_triple_product(surface_eps * advection_eps * self.ramp * time_proj_coef * G1(m,n,p),
                                                                gamma[k], beta[m][i], river_bc_coef_y[n][j], test_function, side)
                                    # vertical (in the sense of sigma coordinates and transformed vertical velocity)
                                    self.add_triple_product_horizontal_derivative(surface_eps * advection_eps * self.ramp * time_proj_coef * G2(m,n,p), 0,
                                                                                  flow_variable[m][i], alpha[n][j] + river_bc_coef[n][j], gamma[k], test_function, side)
                                    self.add_triple_product_horizontal_derivative(surface_eps * advection_eps * self.ramp * time_proj_coef * G2(m,n,p), 0,
                                                                                  flow_variable[m][i], gamma[k], alpha[n][j], test_function, side)
                                    self.add_triple_product(surface_eps * advection_eps * self.ramp * time_proj_coef * G2(m,n,p),
                                                            flow_variable[m][i], gamma[k], river_bc_coef_x[n][j], test_function, side)
                                    self.add_triple_product_horizontal_derivative(surface_eps * advection_eps * self.ramp * time_proj_coef * G2(m,n,p), 1,
                                                                                  flow_variable[m][i], beta[n][j], gamma[k], test_function, side)
                                    self.add_triple_product_horizontal_derivative(surface_eps * advection_eps * self.ramp * time_proj_coef * G2(m,n,p), 1,
                                                                                  flow_variable[m][i], gamma[k], beta[n][j], test_function, side)
                                    if equation == 'u':
                                        self.add_triple_product_horizontal_derivative(surface_eps * advection_eps * self.ramp * time_proj_coef * G2(m,n,p), 0,
                                                                                      river_bc_coef[m][i], alpha[n][j] + river_bc_coef[n][j], gamma[k], test_function, side)
                                        self.add_triple_product_horizontal_derivative(surface_eps * advection_eps * self.ramp * time_proj_coef * G2(m,n,p), 0,
                                                                                      river_bc_coef[m][i], gamma[k], alpha[n][j], test_function, side)
                                        self.add_triple_product(surface_eps * advection_eps * self.ramp * time_proj_coef * G2(m,n,p),
                                                                river_bc_coef[m][i], gamma[k], river_bc_coef_x[n][j], test_function, side)
                                        self.add_triple_product_horizontal_derivative(surface_eps * advection_eps * self.ramp * time_proj_coef * G2(m,n,p), 1,
                                                                                      river_bc_coef[m][i], beta[n][j], gamma[k], test_function, side)
                                        self.add_triple_product_horizontal_derivative(surface_eps * advection_eps * self.ramp * time_proj_coef * G2(m,n,p), 1,
                                                                                      river_bc_coef[m][i], gamma[k], beta[n][j], test_function, side)   
                                else:
                                    # longitudinal
                                    self.add_triple_product_horizontal_derivative(surface_eps * advection_eps * self.ramp * time_proj_coef * G1(m,n,p), 0, 
                                                                                  gamma[k], alpha[m][i], flow_variable[n][j], test_function, side)
                                    # lateral
                                    self.add_triple_product_horizontal_derivative(surface_eps * advection_eps * self.ramp * time_proj_coef * G1(m,n,p), 1, 
                                                                                  gamma[k], beta[m][i], flow_variable[n][j], test_function, side)
                                    # vertical (in the sense of sigma coordinates and transformed vertical velocity)
                                    self.add_triple_product_horizontal_derivative(surface_eps * advection_eps * self.ramp * time_proj_coef * G2(m,n,p), 0,
                                                                                  flow_variable[m][i], alpha[n][j], gamma[k], test_function, side)
                                    self.add_triple_product_horizontal_derivative(surface_eps * advection_eps * self.ramp * time_proj_coef * G2(m,n,p), 0,
                                                                                  flow_variable[m][i], gamma[k], alpha[n][j], test_function, side)
                                    self.add_triple_product_horizontal_derivative(surface_eps * advection_eps * self.ramp * time_proj_coef * G2(m,n,p), 1,
                                                                                  flow_variable[m][i], beta[n][j], gamma[k], test_function, side)
                                    self.add_triple_product_horizontal_derivative(surface_eps * advection_eps * self.ramp * time_proj_coef * G2(m,n,p), 1,
                                                                                  flow_variable[m][i], gamma[k], beta[n][j], test_function, side)


    def add_advection_surface_interactions_linearised(self, p: int, l: int, equation='u', Q0=None):
        surface_eps = self.constant_parameters['surface_epsilon']
        advection_eps = self.constant_parameters['advection_epsilon']
        sigma = self.constant_parameters['sigma']

        G1 = self.vertical_basis.tensor_dict['G1']
        G2 = self.vertical_basis.tensor_dict['G2']
        G7 = self.vertical_basis.tensor_dict['G7']
        H4 = self.time_basis.tensor_dict['H4']
        H5 = self.time_basis.tensor_dict['H5']


        flow_variable = self.alpha if equation == 'u' else self.beta
        flow_variable_0 = self.alpha0 if equation == 'u' else self.beta0
        flow_variable_0_grad = self.alpha0_grad if equation == 'u' else self.beta0_grad
        test_function = self.umom_test_functions[p][l] if equation == 'u' else self.vmom_test_functions[p][l]
        
        if self.internal_river_bc:
            river_bc_coef_0 = [{0: Q0[0] * self.river_interpolant} for m in range(self.M)]
            river_bc_coef_0_x = [{0: Q0[0] * self.river_interpolant_x} for m in range(self.M)]
            river_bc_coef_0_y = [{0: 0} for m in range(self.M)]

            for m in range(self.M):
                for l in range(-self.imax, self.imax + 1):
                    if l == 0:
                        continue
                    river_bc_coef_0[m][l] = 0
                    river_bc_coef_0_x[m][l] = 0
                    river_bc_coef_0_y[m][l] = 0
                    self.river_bc_trial_functions[m][l] = 0
                    self.river_bc_trial_functions_x[m][l] = 0
                    self.river_bc_trial_functions_y[m][l] = 0
        

        # double product interactions
        for i in range(-self.imax, self.imax + 1):
            for j in range(-self.imax, self.imax + 1):
                for m in range(self.M):
                    if self.surface_matrix[abs(l), abs(i)] and self.surface_matrix[abs(l), abs(j)] and self.advection_matrix[abs(l), abs(i)] and self.advection_matrix[abs(l), abs(j)]:
                        proj_coef = H4(i, j, l) * G7(m, p)
                        self.add_double_product_linearised(-surface_eps * advection_eps * self.ramp * proj_coef * sigma, self.gamma[i], self.gamma0[i], flow_variable[m][j], flow_variable_0[m][j], test_function)
                        if self.internal_river_bc and equation == 'u':
                            self.add_double_product_linearised(-surface_eps * advection_eps * self.ramp * proj_coef * sigma, self.gamma[i], self.gamma0[i], self.river_bc_trial_functions[m][j], river_bc_coef_0[m][j], test_function)


        # triple product interactions
        for i in range(-self.imax, self.imax + 1):
            for j in range(-self.imax, self.imax + 1):
                for k in range(-self.imax, self.imax + 1):
                    if self.surface_matrix[abs(l), abs(i)] and self.surface_matrix[abs(l), abs(j)] and self.surface_matrix[abs(l), abs(k)] and self.advection_matrix[abs(l), abs(i)] and self.advection_matrix[abs(l), abs(j)]:
                        time_proj_coef = H5(i,j,k,l)
                        for m in range(self.M):
                            for n in range(self.M):  
                                if self.internal_river_bc:
                                    # longitudinal
                                    self.add_triple_product_horizontal_derivative_linearised(surface_eps * advection_eps * self.ramp * time_proj_coef * G1(m,n,p), 0, 
                                                                                             self.gamma[k], self.gamma0[k],
                                                                                             self.alpha[m][i] + self.river_bc_trial_functions[m][i], self.alpha0[m][i] + river_bc_coef_0[m][i],
                                                                                             flow_variable[n][j], flow_variable_0[n][j], flow_variable_0_grad[n][j], test_function)
                                    if equation == 'u':
                                        self.add_triple_product_linearised(surface_eps * advection_eps * self.ramp * time_proj_coef * G1(m,n,p), 
                                                                           self.gamma[k], self.gamma0[k],
                                                                           self.alpha[m][i] + self.river_bc_trial_functions[m][i], self.alpha0[m][i] + river_bc_coef_0[m][i],
                                                                           self.river_bc_trial_functions_x[n][j], river_bc_coef_0_x[n][j], test_function)
                                    # lateral
                                    self.add_triple_product_horizontal_derivative_linearised(surface_eps * advection_eps * self.ramp * time_proj_coef * G1(m,n,p), 1, 
                                                                                             self.gamma[k], self.gamma0[k], 
                                                                                             self.beta[m][i], self.beta0[m][i],
                                                                                             flow_variable[n][j], flow_variable_0[n][j], flow_variable_0_grad[n][j], test_function)
                                    if equation == 'u':
                                        self.add_triple_product_linearised(surface_eps * advection_eps * self.ramp * time_proj_coef * G1(m,n,p),
                                                                           self.gamma[k], self.gamma0[k],
                                                                           self.beta[m][i], self.beta0[m][i],
                                                                           self.river_bc_trial_functions_y[n][j], river_bc_coef_0_y[n][j], test_function)
                                    # vertical (in the sense of sigma coordinates and transformed vertical velocity)
                                    self.add_triple_product_horizontal_derivative_linearised(surface_eps * advection_eps * self.ramp * time_proj_coef * G2(m,n,p), 0,
                                                                                             flow_variable[m][i], flow_variable_0[m][i],
                                                                                             self.alpha[n][j] + self.river_bc_trial_functions[n][j], self.alpha0[n][j] + river_bc_coef_0[n][j],
                                                                                             self.gamma[k], self.gamma0[k], self.gamma0_grad[k], test_function)
                                    self.add_triple_product_horizontal_derivative_linearised(surface_eps * advection_eps * self.ramp * time_proj_coef * G2(m,n,p), 0,
                                                                                             flow_variable[m][i], flow_variable_0[m][i],
                                                                                             self.gamma[k], self.gamma0[k], 
                                                                                             self.alpha[n][j], self.alpha0[n][j], self.alpha0_grad[n][j], test_function)
                                    self.add_triple_product_linearised(surface_eps * advection_eps * self.ramp * time_proj_coef * G2(m,n,p),
                                                                       flow_variable[m][i], flow_variable_0[m][i], 
                                                                       self.gamma[k], self.gamma0[k],
                                                                       self.river_bc_trial_functions_x[n][j], river_bc_coef_0_x[n][j], test_function)
                                    self.add_triple_product_horizontal_derivative_linearised(surface_eps * advection_eps * self.ramp * time_proj_coef * G2(m,n,p), 1,
                                                                                             flow_variable[m][i], flow_variable_0[m][i],
                                                                                             self.beta[n][j], self.beta0[n][j],
                                                                                             self.gamma[k], self.gamma0[k], self.gamma0_grad[k], test_function)
                                    self.add_triple_product_horizontal_derivative_linearised(surface_eps * advection_eps * self.ramp * time_proj_coef * G2(m,n,p), 1,
                                                                                             flow_variable[m][i], flow_variable_0[m][i],
                                                                                             self.gamma[k], self.gamma0[k],
                                                                                             self.beta[n][j], self.beta0[n][j], self.beta0_grad[n][j], test_function)
                                    if equation == 'u':
                                        self.add_triple_product_horizontal_derivative_linearised(surface_eps * advection_eps * self.ramp * time_proj_coef * G2(m,n,p), 0,
                                                                                                 self.river_bc_trial_functions[m][i], river_bc_coef_0[m][i],
                                                                                                 self.alpha[n][j] + self.river_bc_trial_functions[n][j], self.alpha0[n][j] + river_bc_coef_0[n][j],
                                                                                                 self.gamma[k], self.gamma0[k], self.gamma0_grad[k], test_function)
                                        self.add_triple_product_horizontal_derivative_linearised(surface_eps * advection_eps * self.ramp * time_proj_coef * G2(m,n,p), 0,
                                                                                                 self.river_bc_trial_functions[m][i], river_bc_coef_0[m][i],
                                                                                                 self.gamma[k], self.gamma0[k], 
                                                                                                 self.alpha[n][j], self.alpha0[n][j], self.alpha0_grad[n][j], test_function)
                                        self.add_triple_product_linearised(surface_eps * advection_eps * self.ramp * time_proj_coef * G2(m,n,p),
                                                                           self.river_bc_trial_functions[m][i], river_bc_coef_0[m][i], 
                                                                           self.gamma[k], self.gamma0[k], 
                                                                           self.river_bc_trial_functions_x[n][j], river_bc_coef_0_x[n][j], test_function)
                                        self.add_triple_product_horizontal_derivative_linearised(surface_eps * advection_eps * self.ramp * time_proj_coef * G2(m,n,p), 1,
                                                                                                 self.river_bc_trial_functions[m][i], river_bc_coef_0[m][i],
                                                                                                 self.beta[n][j], self.beta0[n][j], 
                                                                                                 self.gamma[k], self.gamma0[k], self.gamma0_grad[k], test_function)
                                        self.add_triple_product_horizontal_derivative_linearised(surface_eps * advection_eps * self.ramp * time_proj_coef * G2(m,n,p), 1,
                                                                                                 self.river_bc_trial_functions[m][i], river_bc_coef_0[m][i],
                                                                                                 self.gamma[k], self.gamma0[k],
                                                                                                 self.beta[n][j], self.beta0[n][j], self.beta0_grad[n][j], test_function)                                        
                                else:
                                    # longitudinal
                                    self.add_triple_product_horizontal_derivative_linearised(surface_eps * advection_eps * self.ramp * time_proj_coef * G1(m,n,p), 0, 
                                                                                             self.gamma[k], self.gamma0[k], 
                                                                                             self.alpha[m][i], self.alpha0[m][i],
                                                                                             flow_variable[n][j], flow_variable_0[n][j], flow_variable_0_grad[n][j], test_function)
                                    # lateral
                                    self.add_triple_product_horizontal_derivative_linearised(surface_eps * advection_eps * self.ramp * time_proj_coef * G1(m,n,p), 1, 
                                                                                             self.gamma[k], self.gamma0[k],
                                                                                             self.beta[m][i], self.beta0[m][i],
                                                                                             flow_variable[n][j], flow_variable_0[n][j], flow_variable_0_grad[n][j], test_function)
                                    # vertical (in the sense of sigma coordinates and transformed vertical velocity)
                                    self.add_triple_product_horizontal_derivative_linearised(surface_eps * advection_eps * self.ramp * time_proj_coef * G2(m,n,p), 0,
                                                                                             flow_variable[m][i], flow_variable_0[m][i],
                                                                                             self.alpha[n][j], self.alpha0[n][j],
                                                                                             self.gamma[k], self.gamma0[k], self.gamma0_grad[k], test_function)
                                    self.add_triple_product_horizontal_derivative_linearised(surface_eps * advection_eps * self.ramp * time_proj_coef * G2(m,n,p), 0,
                                                                                             flow_variable[m][i], flow_variable_0[m][i],
                                                                                             self.gamma[k], self.gamma0[k],
                                                                                             self.alpha[n][j], self.alpha0[n][j], self.alpha0_grad[n][j], test_function)
                                    self.add_triple_product_horizontal_derivative_linearised(surface_eps * advection_eps * self.ramp * time_proj_coef * G2(m,n,p), 1,
                                                                                             flow_variable[m][i], flow_variable_0[m][i],
                                                                                             self.beta[n][j], self.beta0[n][j],
                                                                                             self.gamma[k], self.gamma0[k], self.gamma0_grad[k], test_function)
                                    self.add_triple_product_horizontal_derivative_linearised(surface_eps * advection_eps * self.ramp * time_proj_coef * G2(m,n,p), 1,
                                                                                             flow_variable[m][i], flow_variable_0[m][i],
                                                                                             self.gamma[k], self.gamma0[k],
                                                                                             self.beta[n][j], self.beta0[n][j], self.beta0_grad[n][j], test_function)


    # functions to add full equations #########################################

    def add_depth_integrated_continuity_equation(self):
        """"""

        if 'transport_divergence' in self.forcing_instruction.keys():
            self.add_transport_divergence(0, forcing=self.forcing_instruction['transport_divergence'])

        if self.surface_in_sigma and 'Stokes_transport_divergence' in self.forcing_instruction.keys():
            self.add_stokes_transport(0, forcing=self.forcing_instruction['Stokes_transport_divergence'])

        for l in range(1, self.imax + 1):
            if 'DIC_time_derivative' in self.forcing_instruction.keys():
                self.add_DIC_time_derivative(l, forcing=self.forcing_instruction['DIC_time_derivative'])

            if 'transport_divergence' in self.forcing_instruction.keys():
                self.add_transport_divergence(l, forcing=self.forcing_instruction['transport_divergence'])
                self.add_transport_divergence(-l, forcing=self.forcing_instruction['transport_divergence'])

            if self.surface_in_sigma and 'Stokes_transport_divergence' in self.forcing_instruction.keys():
                self.add_stokes_transport(l, forcing=self.forcing_instruction['Stokes_transport_divergence'])
                self.add_stokes_transport(-l, forcing=self.forcing_instruction['Stokes_transport_divergence'])


    def add_depth_integrated_continuity_equation_linearised(self, Q0=None):
        """ """

        if 'transport_divergence' in self.forcing_instruction.keys():
            self.add_transport_divergence(0)

        if self.surface_in_sigma and 'Stokes_transport_divergence' in self.forcing_instruction.keys():
            self.add_stokes_transport_linearised(0, Q0=Q0)

        for l in range(1, self.imax + 1):
            if 'DIC_time_derivative' in self.forcing_instruction.keys():
                self.add_DIC_time_derivative(l)

            if 'transport_divergence' in self.forcing_instruction.keys():
                self.add_transport_divergence(l)
                self.add_transport_divergence(-l)

            if self.surface_in_sigma and 'Stokes_transport_divergence' in self.forcing_instruction.keys():
                self.add_stokes_transport_linearised(l, Q0=Q0)
                self.add_stokes_transport_linearised(-l, Q0=Q0)


    def add_internal_sea_boundary_condition(self, dirac_delta_width=0.05):
        """"""

        amplitudes = self.constant_parameters['seaward_amplitudes']
        phases = self.constant_parameters['seaward_phases']

        dirac_delta_sea = ngsolve.IfPos((ngsolve.x - dirac_delta_width/2) * (-dirac_delta_width/2 - ngsolve.x),
                                        (4/(dirac_delta_width**2)) * (dirac_delta_width / 2 - ngsolve.sqrt((ngsolve.x)**2)),
                                        0) # Hat function Dirac Delta
        
        self.add_forcing((self.gamma[0]) * dirac_delta_sea, self.sea_bc_test_functions[0])
        self.add_forcing(-amplitudes[0] * dirac_delta_sea * np.sqrt(2), self.sea_bc_test_functions[0])

        for l in range(1, self.imax + 1):
            if amplitudes[l] == 0:
                self.add_forcing((self.gamma[l]) * dirac_delta_sea, self.sea_bc_test_functions[l])
                self.add_forcing((self.gamma[-l]) * dirac_delta_sea, self.sea_bc_test_functions[-l])
            else:
                # add highly non-linear weak form that we do not have a ready-made function for already
                # amplitude equation
                self.weak_form += ngsolve.sqrt((self.gamma[-l])**2 +
                                               (self.gamma[l])**2) * dirac_delta_sea * self.sea_bc_test_functions[-l] * ngsolve.dx
                self.add_forcing(-amplitudes[l] * dirac_delta_sea, self.sea_bc_test_functions[-l])
                # phase equation
                self.weak_form += ngsolve.atan2(-(self.gamma[-l]), self.gamma[l]) * \
                                  dirac_delta_sea * self.sea_bc_test_functions[l] * ngsolve.dx
                self.add_forcing(-phases[l - 1] * dirac_delta_sea, self.sea_bc_test_functions[l])


    def add_internal_sea_boundary_condition_linearised(self, dirac_delta_width=0.05):
        """"""
        amplitudes = self.constant_parameters['seaward_amplitudes']

        dirac_delta_sea = ngsolve.IfPos((ngsolve.x - dirac_delta_width/2) * (-dirac_delta_width/2 - ngsolve.x),
                                        (4/(dirac_delta_width**2)) * (dirac_delta_width / 2 - ngsolve.sqrt((ngsolve.x)**2)),
                                        0) # Hat function Dirac Delta
        
        self.add_forcing((self.gamma[0]) * dirac_delta_sea, self.sea_bc_test_functions[0])

        # sea_bc_coef_0 = {l: A0[l] * self.sea_interpolant for l in range(-self.imax, self.imax + 1)}

        for l in range(1, self.imax + 1):
            if amplitudes[l] == 0:
                self.add_forcing((self.gamma[l]) * dirac_delta_sea, self.sea_bc_test_functions[l])
                self.add_forcing((self.gamma[-l]) * dirac_delta_sea, self.sea_bc_test_functions[-l])
            else:
                # amplitude equation linearisation
                self.add_forcing(dirac_delta_sea / ngsolve.sqrt((self.gamma0[-l])**2 + (self.gamma0[l])**2) * 
                                 (self.gamma0[l]) * (self.gamma[l]), self.sea_bc_test_functions[-l])
                self.add_forcing(dirac_delta_sea / ngsolve.sqrt((self.gamma0[-l])**2 + (self.gamma0[l])**2) * 
                                 (self.gamma0[-l]) * (self.gamma[-l]), self.sea_bc_test_functions[-l])
                # phase equation linearisation
                self.add_forcing(-dirac_delta_sea / ((self.gamma0[l]) * (1 + ((self.gamma0[-l])**2 / (self.gamma0[l])**2))) * 
                                 (self.gamma[-l]), self.sea_bc_test_functions[l])
                self.add_forcing(dirac_delta_sea * (self.gamma0[-l]) / ((self.gamma0[-l])**2 + (self.gamma0[l])**2) * 
                                 (self.gamma[l]), self.sea_bc_test_functions[l])


    def add_internal_river_boundary_condition(self, dirac_delta_width=0.05):
        """"""
        dirac_delta_river = ngsolve.IfPos((ngsolve.x - dirac_delta_width/2 - (self.geometric_information['riverine_boundary_x']/self.x_scaling)) * ((self.geometric_information['riverine_boundary_x']/self.x_scaling) - dirac_delta_width/2 - ngsolve.x),
                                          (4/(dirac_delta_width**2)) * (dirac_delta_width / 2 - ngsolve.sqrt((ngsolve.x-(self.geometric_information['riverine_boundary_x']/self.x_scaling))**2)),
                                          0) # Hat function Dirac Delta
        
        discharge = self.constant_parameters['discharge']
        eps = self.constant_parameters['surface_epsilon']
        H = self.spatial_parameters['H']

        G4 = self.vertical_basis.tensor_dict['G4']
     
        for m in range(self.M):
            proj_coef = 0.5 * np.sqrt(2) * G4(m)
            self.add_forcing(self.y_scaling * H * dirac_delta_river * proj_coef * (self.alpha[m][0] + self.river_bc_trial_functions[m][0]), self.river_bc_test_functions[0])

        self.add_forcing(discharge * dirac_delta_river, self.river_bc_test_functions[0]) 

        # non-linear part (transport through cross-section from z=0 till z=zeta)

        for i in range(-self.imax, self.imax + 1):
            if self.surface_matrix[0, abs(i)] and eps != 0:
                for m in range(self.M):
                    if i == 0:
                        proj_coef = 0.5 * G4(m)
                        self.add_double_product(proj_coef * self.y_scaling * dirac_delta_river, self.gamma[i], self.alpha[m][i] + self.river_bc_trial_functions[m][i], self.river_bc_test_functions[0])
                    else:
                        proj_coef = 0.5 * G4(m)
                        self.add_double_product(proj_coef * self.y_scaling * dirac_delta_river, self.gamma[i], self.alpha[m][i], self.river_bc_test_functions[0])


    def add_internal_river_boundary_condition_linearised(self, dirac_delta_width=0.05, Q0=None):
        """"""
        dirac_delta_river = ngsolve.IfPos((ngsolve.x - dirac_delta_width/2 - (self.geometric_information['riverine_boundary_x']/self.x_scaling)) * ((self.geometric_information['riverine_boundary_x']/self.x_scaling) - dirac_delta_width/2 - ngsolve.x),
                                          (4/(dirac_delta_width**2)) * (dirac_delta_width / 2 - ngsolve.sqrt((ngsolve.x-(self.geometric_information['riverine_boundary_x']/self.x_scaling))**2)),
                                          0) # Hat function Dirac Delta
        
        H = self.spatial_parameters['H']
        eps = self.constant_parameters['surface_epsilon']

        G4 = self.vertical_basis.tensor_dict['G4']
        H3 = self.time_basis.tensor_dict['H3']

        # river_bc_coef_0 = [{0: Q0[0] * self.normal_alpha[m] * self.river_interpolant} for m in range(self.M)]
        river_bc_coef_0 = [{0: Q0[0] * self.river_interpolant} for m in range(self.M)]
        # if self.internal_sea_bc:
        #     sea_bc_coef_0 = {l: A0[l] * self.sea_interpolant for l in range(-self.imax, self.imax + 1)}

        # linear part (transport through cross-section from z=-H till z=0; nothing changes in linearisation

        for m in range(self.M):
            proj_coef = 0.5 * np.sqrt(2) * self.vertical_basis.tensor_dict['G4'](m)
            self.add_forcing(self.y_scaling * H * dirac_delta_river * proj_coef * (self.alpha[m][0] + self.river_bc_trial_functions[m][0]), self.river_bc_test_functions[0])

                
     

        # non-linear part (transport through cross-section from z=0 till z=zeta)

        for i in range(-self.imax, self.imax + 1):
            if self.surface_matrix[0, abs(i)] and eps != 0:
                for m in range(self.M):
                    proj_coef = 0.5 * G4(m)
                    if i == 0:
                        self.add_double_product_linearised(proj_coef * self.y_scaling * dirac_delta_river, self.gamma[i], self.gamma0[i],
                                                       self.alpha[m][i] + self.river_bc_trial_functions[m][i], self.alpha0[m][i] + river_bc_coef_0[m][i], self.river_bc_test_functions[0])
                    else:
                        self.add_double_product_linearised(proj_coef * self.y_scaling * dirac_delta_river, self.gamma[i], self.gamma0[i],
                                                       self.alpha[m][i], self.alpha0[m][i], self.river_bc_test_functions[0])
                    # if self.internal_sea_bc:
                    #     self.add_double_product_linearised(proj_coef * self.y_scaling * dirac_delta_river, self.sea_bc_trial_functions[i], sea_bc_coef_0[i],
                    #                                        self.alpha[m][i] + self.river_bc_trial_functions[m][i], self.alpha0[m][i] + river_bc_coef_0[m][i], self.river_bc_test_functions[0])

        # for l in range(-self.imax, self.imax + 1):
        #     if l == 0:
        #         continue

        #     for i in range(-self.imax, self.imax + 1):
        #         for j in range(-self.imax, self.imax + 1):
        #             if self.surface_matrix[abs(l), abs(i)] and self.surface_matrix[abs(l), abs(j)]:
        #                 for m in range(self.M):
        #                     proj_coef = H3(i,j,l) * G4(m)
        #                     if j == 0:
        #                         self.add_double_product_linearised(proj_coef * self.y_scaling * dirac_delta_river, self.gamma[i], self.gamma0[i], 
        #                                                         self.alpha[m][j] + self.river_bc_trial_functions[m][j], self.alpha0[m][j] + river_bc_coef_0[m][j], self.river_bc_test_functions[l])
        #                     else:    
        #                         self.add_double_product_linearised(proj_coef * self.y_scaling * dirac_delta_river, self.gamma[i], self.gamma0[i], 
        #                                                         self.alpha[m][j], self.alpha0[m][j], self.river_bc_test_functions[l])
                            # if self.internal_sea_bc:
                            #     self.add_double_product_linearised(proj_coef * self.y_scaling * dirac_delta_river, self.sea_bc_trial_functions[i], sea_bc_coef_0[i],
                            #                                        self.alpha[m][j] + self.river_bc_trial_functions[m][j], self.alpha0[m][j] + river_bc_coef_0[m][j], self.river_bc_test_functions[l])


    def add_momentum_equation(self, equation='u'):
        """"""

        for p in range(self.M):
            
            if 'momentum_advection' in self.forcing_instruction.keys():
                self.add_momentum_advection(p, 0, equation=equation, forcing=self.forcing_instruction['momentum_advection'])

            if 'Coriolis' in self.forcing_instruction.keys():
                self.add_coriolis(p, 0, equation=equation, forcing=self.forcing_instruction['Coriolis'])

            if 'barotropic_pressure_gradient' in self.forcing_instruction.keys():
                self.add_barotropic_pressure_gradient(p, 0, equation=equation, forcing=self.forcing_instruction['barotropic_pressure_gradient'])
            if 'tide_bc' in self.forcing_instruction.keys() and equation == 'u':
                self.add_natural_sea_bc(p, 0, forcing=self.forcing_instruction['tide_bc'])

            if 'baroclinic_pressure_gradient' in self.forcing_instruction.keys():
                self.add_baroclinic_pressure_gradient(p, 0, equation=equation, forcing=self.forcing_instruction['baroclinic_pressure_gradient'])

            if 'vertical_eddy_viscosity' in self.forcing_instruction.keys():
                self.add_vertical_eddy_viscosity(p, 0, equation=equation, forcing=self.forcing_instruction['vertical_eddy_viscosity'])

            if 'horizontal_eddy_viscosity' in self.forcing_instruction.keys():
                self.add_horizontal_eddy_viscosity(p, 0, equation=equation, forcing=self.forcing_instruction['horizontal_eddy_viscosity'])

            if self.surface_in_sigma:
                if self.model_options['include_advection_surface_interactions'] and 'momentum_advection_surface_interactions' in self.forcing_instruction.keys():
                    self.add_advection_surface_interactions(p, 0, equation=equation, forcing=self.forcing_instruction['momentum_advection_surface_interactions'])

                if 'Coriolis_surface_interactions' in self.forcing_instruction.keys():
                    self.add_coriolis_surface_interaction(p, 0, equation=equation, forcing=self.forcing_instruction['Coriolis_surface_interactions'])

                if 'barotropic_pressure_gradient_surface_interactions' in self.forcing_instruction.keys():
                    self.add_barotropic_pressure_gradient_surface_interaction(p, 0, equation=equation, 
                                                                              forcing=self.forcing_instruction['barotropic_pressure_gradient_surface_interactions'])

                if 'baroclinic_pressure_gradient_surface_interactions' in self.forcing_instruction.keys():
                    self.add_baroclinic_pressure_gradient_surface_interaction(p, 0, equation=equation, 
                                                                              forcing=self.forcing_instruction['baroclinic_pressure_gradient_surface_interactions'])
                    
                if 'horizontal_eddy_viscosity_surface_interactions' in self.forcing_instruction.keys():
                    self.add_horizontal_eddy_viscosity_surface_interactions(p, 0, equation=equation, forcing=self.forcing_instruction['horizontal_eddy_viscosity_surface_interactions'])

            for l in range(1, self.imax + 1):

                if 'MOM_time_derivative' in self.forcing_instruction.keys():
                    self.add_momentum_time_derivative(p, l, equation=equation, forcing=self.forcing_instruction['MOM_time_derivative'])

                if 'momentum_advection' in self.forcing_instruction.keys():
                    self.add_momentum_advection(p, -l, equation=equation, forcing=self.forcing_instruction['momentum_advection'])
                    self.add_momentum_advection(p, l, equation=equation, forcing=self.forcing_instruction['momentum_advection'])

                if 'Coriolis' in self.forcing_instruction.keys():
                    self.add_coriolis(p, -l, equation=equation, forcing=self.forcing_instruction['Coriolis'])
                    self.add_coriolis(p, l, equation=equation, forcing=self.forcing_instruction['Coriolis'])

                if 'barotropic_pressure_gradient' in self.forcing_instruction.keys():
                    self.add_barotropic_pressure_gradient(p, -l, equation=equation, forcing=self.forcing_instruction['barotropic_pressure_gradient'])
                    
                    self.add_barotropic_pressure_gradient(p, l, equation=equation, forcing=self.forcing_instruction['barotropic_pressure_gradient'])

                if 'tide_bc' in self.forcing_instruction.keys() and equation == 'u':
                    self.add_natural_sea_bc(p, -l, forcing=self.forcing_instruction['tide_bc'])
                    self.add_natural_sea_bc(p, l, forcing=self.forcing_instruction['tide_bc'])

                if 'baroclinic_pressure_gradient' in self.forcing_instruction.keys():
                    self.add_baroclinic_pressure_gradient(p, -l, equation=equation, forcing=self.forcing_instruction['baroclinic_pressure_gradient'])
                    self.add_baroclinic_pressure_gradient(p, l, equation=equation, forcing=self.forcing_instruction['baroclinic_pressure_gradient'])

                if 'vertical_eddy_viscosity' in self.forcing_instruction.keys():
                    self.add_vertical_eddy_viscosity(p, -l, equation=equation, forcing=self.forcing_instruction['vertical_eddy_viscosity'])
                    self.add_vertical_eddy_viscosity(p, l, equation=equation, forcing=self.forcing_instruction['vertical_eddy_viscosity'])

                if 'horizontal_eddy_viscosity' in self.forcing_instruction.keys():
                    self.add_horizontal_eddy_viscosity(p, -l, equation=equation, forcing=self.forcing_instruction['horizontal_eddy_viscosity'])
                    self.add_horizontal_eddy_viscosity(p, l, equation=equation, forcing=self.forcing_instruction['horizontal_eddy_viscosity'])

                if self.surface_in_sigma:

                    if 'MOM_time_derivative_surface_interactions' in self.forcing_instruction.keys():
                        self.add_time_derivative_surface_interactions(p, l, equation=equation, forcing=self.forcing_instruction['MOM_time_derivative_surface_interactions'])

                    if self.model_options['include_advection_surface_interactions'] and 'momentum_advection_surface_interactions' in self.forcing_instruction.keys():
                        self.add_advection_surface_interactions(p, -l, equation=equation, forcing=self.forcing_instruction['momentum_advection_surface_interactions'])
                        self.add_advection_surface_interactions(p, l, equation=equation, forcing=self.forcing_instruction['momentum_advection_surface_interactions'])

                    if 'Coriolis_surface_interactions' in self.forcing_instruction.keys():
                        self.add_coriolis_surface_interaction(p, -l, equation=equation, forcing=self.forcing_instruction['Coriolis_surface_interactions'])
                        self.add_coriolis_surface_interaction(p, l, equation=equation, forcing=self.forcing_instruction['Coriolis_surface_interactions'])

                    if 'barotropic_pressure_gradient_surface_interactions' in self.forcing_instruction.keys():
                        self.add_barotropic_pressure_gradient_surface_interaction(p, -l, equation=equation, 
                                                                                  forcing=self.forcing_instruction['barotropic_pressure_gradient_surface_interactions'])
                        self.add_barotropic_pressure_gradient_surface_interaction(p, l, equation=equation,
                                                                                  forcing=self.forcing_instruction['barotropic_pressure_gradient_surface_interactions'])

                    if 'baroclinic_pressure_gradient_surface_interactions' in self.forcing_instruction.keys():
                        self.add_baroclinic_pressure_gradient_surface_interaction(p, -l, equation=equation,
                                                                                  forcing=self.forcing_instruction['baroclinic_pressure_gradient_surface_interactions'])
                        self.add_baroclinic_pressure_gradient_surface_interaction(p, l, equation=equation, 
                                                                                  forcing=self.forcing_instruction['baroclinic_pressure_gradient_surface_interactions'])
                        
                    if 'horizontal_eddy_viscosity_surface_interactions' in self.forcing_instruction.keys():
                        self.add_horizontal_eddy_viscosity_surface_interactions(p, -l, equation=equation, forcing=self.forcing_instruction['horizontal_eddy_viscosity_surface_interactions'])
                        self.add_horizontal_eddy_viscosity_surface_interactions(p, l, equation=equation, forcing=self.forcing_instruction['horizontal_eddy_viscosity_surface_interactions'])


    def add_momentum_equation_linearised(self, equation='u', Q0=None):
        
        """"""

        for p in range(self.M):
            
            if 'momentum_advection' in self.forcing_instruction.keys():
                self.add_momentum_advection_linearised(p, 0, equation=equation, Q0=Q0)

            if 'Coriolis' in self.forcing_instruction.keys():
                self.add_coriolis(p, 0, equation=equation)

            if 'barotropic_pressure_gradient' in self.forcing_instruction.keys():
                self.add_barotropic_pressure_gradient(p, 0, equation=equation)

            if 'tide_bc' in self.forcing_instruction.keys() and equation == 'u':
                self.add_natural_sea_bc(p, 0)

            # the standard baroclinic pressure gradient term is always a forcing, so must be left out of the linearised equations

            if 'vertical_eddy_viscosity' in self.forcing_instruction.keys():
                self.add_vertical_eddy_viscosity(p, 0, equation=equation)

            if 'horizontal_eddy_viscosity' in self.forcing_instruction.keys():
                self.add_horizontal_eddy_viscosity(p, 0, equation=equation)

            if self.surface_in_sigma:

                if self.model_options['include_advection_surface_interactions'] and 'momentum_advection_surface_interactions' in self.forcing_instruction.keys():
                    self.add_advection_surface_interactions_linearised(p, 0, equation=equation, Q0=Q0)

                if 'Coriolis_surface_interactions' in self.forcing_instruction.keys():
                    self.add_coriolis_surface_interaction_linearised(p, 0, equation=equation, Q0=Q0)

                if 'barotropic_pressure_gradient_surface_interactions' in self.forcing_instruction.keys():
                    self.add_barotropic_pressure_gradient_surface_interaction_linearised(p, 0, equation=equation)

                if 'baroclinic_pressure_gradient_surface_interactions' in self.forcing_instruction.keys():
                    self.add_baroclinic_pressure_gradient_surface_interaction_linearised(p, 0, equation=equation)

                if 'horizontal_eddy_viscosity_surface_interactions' in self.forcing_instruction.keys():
                    self.add_horizontal_eddy_viscosity_surface_interactions_linearised(p, 0, equation=equation, Q0=Q0)

            for l in range(1, self.imax + 1):

                if 'MOM_time_derivative' in self.forcing_instruction.keys():
                    self.add_momentum_time_derivative(p, l, equation=equation)

                if 'momentum_advection' in self.forcing_instruction.keys():
                    self.add_momentum_advection_linearised(p, -l, equation=equation, Q0=Q0)
                    self.add_momentum_advection_linearised(p, l, equation=equation, Q0=Q0)

                if 'Coriolis' in self.forcing_instruction.keys():
                    self.add_coriolis(p, -l, equation=equation)
                    self.add_coriolis(p, l, equation=equation)

                if 'barotropic_pressure_gradient' in self.forcing_instruction.keys():
                    self.add_barotropic_pressure_gradient(p, -l, equation=equation)
                    self.add_barotropic_pressure_gradient(p, l, equation=equation)

                if 'tide_bc' in self.forcing_instruction.keys() and equation == 'u':
                    self.add_natural_sea_bc(p, -l)
                    self.add_natural_sea_bc(p, l)

                if 'vertical_eddy_viscosity' in self.forcing_instruction.keys():
                    self.add_vertical_eddy_viscosity(p, -l, equation=equation)
                    self.add_vertical_eddy_viscosity(p, l, equation=equation)

                if 'horizontal_eddy_viscosity' in self.forcing_instruction.keys():
                    self.add_horizontal_eddy_viscosity(p, -l, equation=equation)
                    self.add_horizontal_eddy_viscosity(p, l, equation=equation)

                if self.surface_in_sigma:
                    if 'MOM_time_derivative_surface_interactions' in self.forcing_instruction.keys():
                        self.add_time_derivative_surface_interactions_linearised(p, l, equation=equation, Q0=Q0)

                    if self.model_options['include_advection_surface_interactions'] and 'momentum_advection_surface_interactions' in self.forcing_instruction.keys():
                            self.add_advection_surface_interactions_linearised(p, -l, equation=equation, Q0=Q0)
                            self.add_advection_surface_interactions_linearised(p, l, equation=equation, Q0=Q0)

                    if 'Coriolis_surface_interactions' in self.forcing_instruction.keys():
                        self.add_coriolis_surface_interaction_linearised(p, -l, equation=equation, Q0=Q0)
                        self.add_coriolis_surface_interaction_linearised(p, l, equation=equation, Q0=Q0)

                    if 'barotropic_pressure_gradient_surface_interactions' in self.forcing_instruction.keys():
                        self.add_barotropic_pressure_gradient_surface_interaction_linearised(p, -l, equation=equation)
                        self.add_barotropic_pressure_gradient_surface_interaction_linearised(p, l, equation=equation)

                    if 'baroclinic_pressure_gradient_surface_interactions' in self.forcing_instruction.keys():
                        self.add_baroclinic_pressure_gradient_surface_interaction_linearised(p, -l, equation=equation)
                        self.add_baroclinic_pressure_gradient_surface_interaction_linearised(p, l, equation=equation)

                    if 'horizontal_eddy_viscosity_surface_interactions' in self.forcing_instruction.keys():
                        self.add_horizontal_eddy_viscosity_surface_interactions_linearised(p, -l, equation =equation, Q0=Q0)
                        self.add_horizontal_eddy_viscosity_surface_interactions_linearised(p, l, equation =equation, Q0=Q0)
                    
