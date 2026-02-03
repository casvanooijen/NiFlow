import numpy as np
import os
import json
import cloudpickle
import ngsolve
import timeit
from contextlib import nullcontext
from copy import deepcopy

import NiFlow.truncationbasis.truncationbasis as truncationbasis
from NiFlow.geometry.create_geometry import parametric_geometry, RIVER, SEA, WALL, WALLUP, WALLDOWN, BOUNDARY_DICT
from NiFlow.geometry.meshing import generate_mesh
from NiFlow.geometry.geometries import *
import NiFlow.define_weak_forms as weakforms
from NiFlow.utils import *
from NiFlow.linear_solver import *


def select_model_options(bed_bc:str = 'no_slip', veddy_viscosity_assumption:str = 'constant',
                         advection_influence_matrix: np.ndarray = None, sea_boundary_treatment:str = 'exact',
                         river_boundary_treatment:str = 'simple', surface_interaction_influence_matrix: np.ndarray = None,
                         include_advection_surface_interactions:bool = True):
    
    """
    
    Returns a dictionary of all the available model options. Safer than manually creating this dictionary.
    
    Arguments: ('...'  means that there will possibly be future options added)

        - bed_bc:                                   indicates what type of boundary condition is used at the river bed ('no_slip' or 'partial_slip');
        - veddy_viscosity_assumption:               structure of the vertical eddy viscosity parameter ('constant' or 'depth-scaled&constantprofile');
        - advection_influence_matrix (np.ndarray):  (imax+1) x (imax+1) - boolean matrix where element (i,j) indicates whether constituent i is influenced by constituent j through momentum advection (if possible);
                                                    more precisely, in the equations for constituent i, any product of constituents that includes constituent j will not be present in the advective terms
                                                    if element (i, j) is False, even if that product *should* physically be present;
        - sea_boundary_treatment:                   indicates how the seaward boundary condition is handled: 'simple' means that the seaward boundary condition is just applied at the boundary of the computational domain,
                                                    'linear_guess' means that we choose a boundary condition for the computational domain such that an amplitude/phase condition holds for the linear version of the model
                                                    at the boundary of the interpretable domain, 'exact' means that the (non-linear!) interpretable boundary condition is included in the model equations.
        - river_boundary_treatment:                 indicates how the riverine boundary condition is handled: 'simple' means that the riverine boundary condition is just applied at the computational boundary;
                                                    'exact' means that the (linear) interpretable boundary condition is included in the model equations. If 'simple' is chosen, any ramping is performed on the inside of the interpretable
                                                    domain where the ramping zone is [L - L_R_river - L_BL_river, L - L_BL_river] and the boundary layer adjustment zone is [L - L_BL_river, L].
        - surface_interaction_influence_matrix:     (imax + 1) x (imax + 1) - boolean matrix where entry (i, j) indicates whether constituent i is influenced by constituent j through non-linear surface interactions (if possible).
        - include_advection_surface_interactions:   if True, then non-linear interaction between the surface and the advective forcing will be included.
        """
    
    if bed_bc == 'partial_slip' and veddy_viscosity_assumption == 'constant':
        raise ValueError("Partial-slip condition and constant vertical eddy viscosity are incompatible")

    options = {
            'bed_bc': bed_bc,
            'veddy_viscosity_assumption': veddy_viscosity_assumption,
            'advection_influence_matrix': advection_influence_matrix, # the validity of this matrix is checked when imax is know, i.e. when the hydrodynamics object is initialised
            'sea_boundary_treatment': sea_boundary_treatment,
            'river_boundary_treatment': river_boundary_treatment,
            'surface_interaction_influence_matrix': surface_interaction_influence_matrix,
            'include_advection_surface_interactions': include_advection_surface_interactions
        }
    

    return options


def set_geometric_information(domain_shape: str, shape_parameters: tuple, x_scaling: float, y_scaling: float, 
                              boundary_layer_transition_length_sea: float, ramp_length_sea: float, ramp_adjustment_length_sea: float,
                              boundary_layer_transition_length_river: float = 0, ramp_length_river: float = 0, ramp_adjustment_length_river: float=0):

    """
    Returns a dictionary of geometric information, including the model geometry, scale parameters, and parameters associated with the ramping of non-linear terms.

    Arguments:
        - domain_shape: shape of the domain; choices currently available include 'rectangle'.
        - shape_parameters: parameters associated with your chosen domain shape; for 'rectangle', the shape parameters are (estuary length, estuary width) in that order.
        - x_scaling: parameter by which the computational geometry is scaled in the x-direction for better mesh generation.
        - y_scaling: parameter by which the computational geometry is scaled in the y-direction for better mesh generation.
        - boundary_layer_transition_length_sea: size of the boundary layer transition zone at the seaward boundary (where the internal dynamics adjust to the boundary condition); recommended to be at least the tidal excursion length.
        - ramp_length_sea: size of the ramping zone at the seaward boundary (where the strength of non-linear terms is gradually increased); recommended to be at least the tidal excursion length.
        - ramp_adjustment_length_sea: size of the ramp adjustment zone at the seaward boundary (where the internal dynamics adjust to the modified dynamics induced by the ramping function); recommended to be at least the tidal excursion length.
        - boundary_layer_transition_length_river: size of the boundary layer transition zone at the riverine boundary (where the internal dynamics adjust to the boundary condition); recommended to be at least the tidal excursion length.
        - ramp_length_river: size of the ramping zone at the riverine boundary (where the strength of non-linear terms is gradually increased); recommended to be at least the tidal excursion length.
        - ramp_adjustment_length_river: size of the ramp adjustment zone at the riverine boundary (where the internal dynamics adjust to the modified dynamics induced by the ramping function); recommended to be at least the tidal excursion length.
    
    """

    if domain_shape == 'rectangle':
        riverine_boundary_x = shape_parameters[0]
    
    geometric_information = {
        'shape': domain_shape,
        'shape_parameters': shape_parameters,
        'x_scaling': x_scaling,
        'y_scaling': y_scaling,
        'L_BL_sea': boundary_layer_transition_length_sea,
        'L_R_sea': ramp_length_sea,
        'L_RA_sea': ramp_adjustment_length_sea,
        'riverine_boundary_x': riverine_boundary_x,
        'L_BL_river': boundary_layer_transition_length_river,
        'L_R_river': ramp_length_river,
        'L_RA_river': ramp_adjustment_length_river
    }

    return geometric_information


def set_numerical_information(M, imax, order, grid_size, mesh_generation_method, element_type, continuation_parameters={'advection_epsilon': [1], 'Av': [1], 'Ah': [1]}):
    """Returns dictionary of numerical information, associated to the meshing and the Finite Element formulation
    
    Arguments:
    
    - M:                                number of vertical basis functions that should be taken into account.
    - imax:                             number of tidal constituents (excluding M0) that should be taken into account.
    - order:                            order of the finite element basis.
    - grid_size:                        grid size used for mesh; if a structured approach is used, then supply a tuple for this variable, which leads to grid_size[0] cells in the y-direction and grid_size[1] cells in the x-direction.                                                         
    - mesh_generation_method:           method by which the mesh is generated; options: 'structured_quads', 'structured_tri_AQ', 'structured_tri_DQ', 'structured_tri_CCQ', 'unstructured'.
    - element_type:                     type of finite element formulation used: 'simple', 'taylor-hood', or 'MINI'.
    - continuation_parameters:          dictionary specifying how continuation will be applied for the Newton method; continuation is possible with
                                        advection_epsilon, vertical eddy viscosity, and horizontal eddy viscosity.
    
    """

    if element_type == 'MINI' and mesh_generation_method == 'structured_quads':
        raise ValueError("Cannot use the MINI element for quadrilateral mesh. Please use 'taylor-hood' instead.")
    
    if mesh_generation_method == 'unstructured':
        if isinstance(grid_size, tuple):
            raise ValueError("For unstructured grids, provide a single float that denotes the size of gridcells.")
    else:
        if not isinstance(grid_size, tuple):
            raise ValueError("For structuref grids, provide a tuple denoting the number of grid cells in each direction")

    info = {
        'M': M,
        'imax': imax,
        'order': order,
        'grid_size': grid_size,
        'mesh_generation_method': mesh_generation_method,
        'element_type': element_type,
        'continuation_parameters': continuation_parameters
    }

    return info


def set_constant_physical_parameters(f: float, Av: float, Ah: float, seaward_amplitudes: list, seaward_phases: list,
                                     river_discharge: float = 0., g: float = 9.81, sigma: float = 2 / 89428.32720,
                                     sf: float=1000000., advection_epsilon: float=1., surface_epsilon: float=1.):
    """Returns a dictionary of constant physical parameters for the model.
    
    Arguments:

    - f: Coriolis frequency.
    - Av: vertical eddy viscosity; if 'depth-scaled&constantprofile' is chosen in model_options['veddy_viscosity_assumption'], then this value will be multiplied by the local water depth in the equations.
    - Ah: horizontal eddy viscosity.
    - seaward_amplitudes: list of water level amplitudes for each tidal constituent that are satisfied (in a width-averaged sense) at the (interpretable) seaward boundary.
                          the first entry - corresponding to the tidally averaged water level - does not contain an amplitude, but rather the actual
                          value of the tidally averaged water level.
    - seaward_phases: list of water level phases for each tidal constituent **except the residual component** that must be satisfied (in a width-averaged sense) at the (interpretable)
                      seaward boundary. This list starts at the principle non-constant tidal component.
    - river_discharge: stationary river discharge in m^3/s (default 0).
    - g: gravitational acceleration (default 9.81).
    - sigma: (non-angular) frequency of the principle tidal component (default 2 / 89428.32720 from Table 3.2 in Gerkema (2019)).
    - sf: partial-slip parameter; only used if the partial-slip boundary condition is selected.
    - advection_epsilon: non-physical parameter that is a factor in front of the advective terms, controlling their strength (default 1, which is the physical case). 
    - surface_epsilon: non-physical parameter that is a factor in front of the non-linear surface interaction terms, controlling their strength (default 1, which is the physical case). 
    
    """

    params = {
        'f': f,
        'Av': Av,
        'Ah': Ah,
        'seaward_amplitudes': seaward_amplitudes,
        'seaward_phases': seaward_phases,
        'discharge': river_discharge,
        'g': g,
        'sigma': sigma,
        'sf': sf,
        'advection_epsilon': advection_epsilon,
        'surface_epsilon': surface_epsilon
    }

    return params


def set_spatial_physical_parameters(H, R=ngsolve.CF(0), rho=ngsolve.CF(1)):
    """Returns a dictionary of spatially varying parameters. These parameters must be defined as ngsolve CoefficientFunctions. Please **do not** use outer scope variables in the definitions of these functions;
    this will mess up the saving/loading of models through pickling. This can always be avoided by using function closures.

    Arguments: 

    - H: water depth below 0.
    - R: reference water level.
    - rho: vertically constant reference level.
    
    """

    H.spacedim=2
    R.spacedim=2
    rho.spacedim=2

    params = {
        'H': H,
        'R': R,
        'rho': rho
    }

    return params


def get_spatial_parameter_gradients(params, mesh):
    param_gradients = {}
    temp_fes = ngsolve.H1(mesh, order=6)
    for name, param in params.items():
        temp_gf = ngsolve.GridFunction(temp_fes)
        temp_gf.Set(param)
        param_gradients[name] = ngsolve.grad(temp_gf)

    return param_gradients

class Hydrodynamics(object):

    """
    Class that stores all information about a model simulation, including the chosen physical processes, parameters, numerical method, geometry, ...

    Attributes:

    Methods:


    """

    def __init__(self, model_options: dict, geometric_information: dict, numerical_information: dict,
                 constant_physical_parameters: dict, spatial_physical_parameters: dict):
        
        """Creates Hydrodynamics object from chosen parameters/processes

        Arguments:
        
        """
        

        # Update advection_influence_matrix to default if it was set to None
        if model_options['advection_influence_matrix'] is None:
            model_options['advection_influence_matrix'] = np.full((numerical_information['imax'] + 1, numerical_information['imax'] + 1), True)

        if model_options['surface_interaction_influence_matrix'] is None:
            model_options['surface_interaction_influence_matrix'] = np.full((numerical_information['imax'] + 1, numerical_information['imax'] + 1), True)

        if np.any(model_options['surface_interaction_influence_matrix']) and model_options['veddy_viscosity_assumption'] == 'constant':
            raise ValueError("Currently impossible to use constant eddy viscosity with non-linear surface interactions. Please use depth-scaled eddy viscosity instead.")

        self.model_options = model_options
        self.geometric_information = geometric_information
        self.numerical_information = numerical_information
        self.constant_physical_parameters = constant_physical_parameters
        

        # MAKE THE GEOMETRY FROM GEOMETRIC_INFORMATION

        if geometric_information['shape'] == 'rectangle':
            # if model_options['river_boundary_treatment'] == 'exact':
            geomcurves = parametric_rectangle(geometric_information['shape_parameters'][0] / geometric_information['x_scaling'], 
                                                geometric_information['shape_parameters'][1] / geometric_information['y_scaling'],
                                                geometric_information['L_BL_sea'] / geometric_information['x_scaling'],
                                                geometric_information['L_R_sea'] / geometric_information['x_scaling'],
                                                geometric_information['L_RA_sea'] / geometric_information['x_scaling'],
                                                geometric_information['L_BL_river'] / geometric_information['x_scaling'],
                                                geometric_information['L_R_river'] / geometric_information['x_scaling'],
                                                geometric_information['L_RA_river'] / geometric_information['x_scaling'])
            display_geomcurves = parametric_rectangle(geometric_information['shape_parameters'][0] / geometric_information['x_scaling'], 
                                              geometric_information['shape_parameters'][1] / geometric_information['y_scaling'],
                                              0, 0, 0, 0, 0, 0) # geometry without ramping zone for postprocessing purposes
        else:
            raise ValueError(f"'{geometric_information['shape']}'-shape not implemented.")
        
        self.geom = parametric_geometry(geomcurves)
        self.display_geom = parametric_geometry(display_geomcurves)

        # CREATE THE MESH

        if numerical_information['mesh_generation_method'] == 'unstructured':
            self.mesh = generate_mesh(self.geom, method='unstructured', maxh_unstructured=numerical_information['grid_size'])
            self.display_mesh = generate_mesh(self.display_geom, method='unstructured', maxh_unstructured=numerical_information['grid_size'])
        else:
            self.mesh = ngsolve.Mesh(generate_mesh(self.geom, method=numerical_information['mesh_generation_method'], num_cells=numerical_information['grid_size']))
            self.display_mesh = ngsolve.Mesh(generate_mesh(self.display_geom, method=numerical_information['mesh_generation_method'], num_cells=numerical_information['grid_size']))

        # SET TRUNCATION BASES
        self.time_basis = truncationbasis.unit_harmonic_time_basis
        if model_options['bed_bc'] == 'no_slip':
            self.vertical_basis = truncationbasis.eigbasis_constantAv
        elif model_options['bed_bc'] == 'partial_slip':
            self.vertical_basis = truncationbasis.eigbasis_partialslip(numerical_information['M'], constant_physical_parameters['sf'], constant_physical_parameters['Av'])

        self._setup_fem_space()            
        self._setup_TnT()

        # CONVERT SPATIAL PARAMETERS TO COEFFICIENTFUNCTIONS

        self.spatial_parameters = spatial_physical_parameters
        self.spatial_parameters_grad = get_spatial_parameter_gradients(spatial_physical_parameters, self.mesh)
        # SET BOUNDARY CONDITIONS

        self._set_seaward_boundary_condition()
        self._set_riverine_boundary_condition()

        # INITIALISE FINITE ELEMENT SPACE AND TEST-/TRIALFUNCTIONS

        self.solution_gf = ngsolve.GridFunction(self.femspace)
        self.restructure_solution()

        self.nfreedofs = count_free_dofs(self.femspace)

        # these are only used in the child class DecomposedHydrodynamics and are defined here to prevent exceptions
        self.is_decomposed = False
        self.as_forcing_list = []
        self.forcing_alpha = None
        self.forcing_beta = None
        self.forcing_gamma = None
        self.forcing_A = None
        self.forcing_Q = None


    # Private methods

    def _set_seaward_boundary_condition(self):
        self.seaward_forcing = SeawardForcing(self)


    def _set_riverine_boundary_condition(self, **kwargs):
        """Sets riverine boundary condition assuming that the depth-averaged along-channel velocity scales linearly with local depth. Only total river discharge (dimensional)
        needs to be provided. If in **kwargs, manual is set to True, a user-provided lateral distribution of the discharge is used instead.
        
        """
        self.riverine_forcing = RiverineForcing(self, **kwargs)


    def _setup_fem_space(self):
        

        # shorthands for long variable names
        M = self.numerical_information['M']
        imax = self.numerical_information['imax']
        
        self.U = ngsolve.H1(self.mesh, order= 2 if (self.numerical_information['element_type'] == 'taylor-hood' and self.numerical_information['order']==1) else self.numerical_information['order'], dirichlet=f"{BOUNDARY_DICT[RIVER]}")  # make sure that zero-th order is not used for the free surface
        self.V = ngsolve.H1(self.mesh, order= 2 if (self.numerical_information['element_type'] == 'taylor-hood' and self.numerical_information['order']==1) else self.numerical_information['order'], dirichlet=f"{BOUNDARY_DICT[WALLDOWN]}|{BOUNDARY_DICT[WALLUP]}")

        # add interior bubble functions if MINI-elements are used
        if self.numerical_information['element_type'] == 'MINI':
            self.U.SetOrder(ngsolve.TRIG, 3 if self.numerical_information['order'] == 1 else self.numerical_information['order'] + 1)
            self.V.SetOrder(ngsolve.TRIG, 3 if self.numerical_information['order'] == 1 else self.numerical_information['order'] + 1)

            self.U.Update()
            self.V.Update()

        

        # define Z-space with order one less than velocity space in case of Taylor-Hood elements or MINI (k>1) elements
        if ((self.numerical_information['element_type'] == 'taylor-hood') or (self.numerical_information['element_type'] == 'MINI')) and self.numerical_information['order'] > 1:
            self.Z = ngsolve.H1(self.mesh, order=self.numerical_information['order'] - 1)
        else:
            self.Z = ngsolve.H1(self.mesh, order=self.numerical_information['order'])

        self.ndofs_alpha = self.U.ndof
        self.ndofs_beta = self.V.ndof
        self.ndofs_gamma = self.Z.ndof

        self.nfreedofs_alpha = count_free_dofs(self.U)
        self.nfreedofs_beta = count_free_dofs(self.V)
        self.nfreedofs_gamma = count_free_dofs(self.Z)

        self.ndofs_u = self.ndofs_alpha * M * (2 * imax + 1)
        self.ndofs_v = self.ndofs_beta * M * (2 * imax + 1)
        self.ndofs_z = self.ndofs_gamma * (2 * imax + 1)
        
        self.nfreedofs_u = self.nfreedofs_alpha * M * (2 * imax + 1)
        self.nfreedofs_v = self.nfreedofs_beta * M * (2 * imax + 1)
        self.nfreedofs_z = self.nfreedofs_gamma * (2 * imax + 1)

        if self.model_options['sea_boundary_treatment'] == 'exact' or self.model_options['river_boundary_treatment'] == 'exact': # take into account floating point errors
            scalarFESpace = ngsolve.NumberSpace(self.mesh)
        
        list_of_spaces = [self.U for _ in range(M*(2*imax + 1))]
        for _ in range(M*(2*imax + 1)): 
            list_of_spaces.append(self.V)
        for _ in range(2*imax + 1):
            list_of_spaces.append(self.Z)

        if self.model_options['sea_boundary_treatment'] == 'exact' and self.model_options['river_boundary_treatment'] == 'exact': # if we treat the boundary on both sides
            for _ in range(2 * imax + 2): # only one dimension for scalar Q
                list_of_spaces.append(scalarFESpace)
        elif self.model_options['sea_boundary_treatment'] != 'exact' and self.model_options['river_boundary_treatment'] != 'exact': # if we do not treat the boundaries
            pass
        elif self.model_options['sea_boundary_treatment'] == 'exact' and self.model_options['river_boundary_treatment'] != 'exact':
            for _ in range(2 * imax + 1): # if we have ramping on only one side
                list_of_spaces.append(scalarFESpace)
        elif self.model_options['sea_boundary_treatment'] != 'exact' and self.model_options['river_boundary_treatment'] == 'exact':
            # for _ in range(2 * imax + 1): # if we have ramping on only one side
            list_of_spaces.append(scalarFESpace) 

        X = ngsolve.FESpace(list_of_spaces)
        self.femspace = X



    def _setup_TnT(self):
        """Sorts the ngsolve Trial and Test functions into intuitive dictionaries"""

        trialtuple = self.femspace.TrialFunction()
        testtuple = self.femspace.TestFunction()

        M = self.numerical_information['M']
        imax = self.numerical_information['imax']
        num_time_components = 2 * imax + 1


        alpha_trialfunctions = [dict() for _ in range(M)]
        umom_testfunctions = [dict() for _ in range(M)] # test functions for momentum equation u

        beta_trialfunctions = [dict() for _ in range(M)]
        vmom_testfunctions = [dict() for _ in range(M)] # test functions for momentum equation v

        gamma_trialfunctions = dict()
        DIC_testfunctions = dict() # test functions for Depth-Integrated Continuity equation

        for m in range(M):

            alpha_trialfunctions[m][0] = trialtuple[m * num_time_components]
            umom_testfunctions[m][0] = testtuple[m * num_time_components]

            beta_trialfunctions[m][0] = trialtuple[(M + m) * num_time_components]
            vmom_testfunctions[m][0] = testtuple[(M + m) * num_time_components]
            for i in range(1, imax + 1):
                alpha_trialfunctions[m][-i] = trialtuple[m * num_time_components + i]
                alpha_trialfunctions[m][i] = trialtuple[m * num_time_components + imax + i]

                umom_testfunctions[m][-i] = testtuple[m * num_time_components + i]
                umom_testfunctions[m][i] = testtuple[m * num_time_components + imax + i]

                beta_trialfunctions[m][-i] = trialtuple[(M + m) * num_time_components + i]
                beta_trialfunctions[m][i] = trialtuple[(M + m) * num_time_components + imax + i]

                vmom_testfunctions[m][-i] = testtuple[(M + m) * num_time_components + i]
                vmom_testfunctions[m][i] = testtuple[(M + m) * num_time_components + imax + i]
        
        gamma_trialfunctions[0] = trialtuple[2*M*num_time_components]
        DIC_testfunctions[0] = testtuple[2*M*num_time_components]

        for i in range(1, imax + 1):
            gamma_trialfunctions[-i] = trialtuple[2*M*num_time_components + i]
            gamma_trialfunctions[i] = trialtuple[2*M*num_time_components + imax + i]

            DIC_testfunctions[-i] = testtuple[2*M*num_time_components + i]
            DIC_testfunctions[i] = testtuple[2*M*num_time_components + imax + i]

        if self.model_options['sea_boundary_treatment'] == 'exact': 
            A_trialfunctions = dict()
            sea_boundary_testfunctions = dict()

            A_trialfunctions[0] = trialtuple[2*M*num_time_components + num_time_components]
            sea_boundary_testfunctions[0] = testtuple[2*M*num_time_components + num_time_components]

            for i in range(1, imax + 1):
                A_trialfunctions[-i] = trialtuple[2*M*num_time_components + num_time_components + i]
                A_trialfunctions[i] = trialtuple[2*M*num_time_components + num_time_components + imax + i]

                sea_boundary_testfunctions[-i] = testtuple[2*M*num_time_components + num_time_components + i]
                sea_boundary_testfunctions[i] = testtuple[2*M*num_time_components + num_time_components + imax + i]

            self.A_trialfunctions = A_trialfunctions
            self.sea_boundary_testfunctions = sea_boundary_testfunctions

            if self.model_options['river_boundary_treatment'] == 'exact': # if both sides are ramped
                Q_trialfunctions = dict()
                river_boundary_testfunctions = dict()

                Q_trialfunctions[0] = trialtuple[2*M*num_time_components + 2*num_time_components]
                river_boundary_testfunctions[0] = testtuple[2*M*num_time_components + 2*num_time_components]

                self.Q_trialfunctions = Q_trialfunctions
                self.river_boundary_testfunctions = river_boundary_testfunctions

        elif self.model_options['river_boundary_treatment'] == 'exact': # if only the river side is ramped
                Q_trialfunctions = dict()
                river_boundary_testfunctions = dict()

                Q_trialfunctions[0] = trialtuple[2*M*num_time_components + num_time_components]
                river_boundary_testfunctions[0] = testtuple[2*M*num_time_components + num_time_components]

                self.Q_trialfunctions = Q_trialfunctions
                self.river_boundary_testfunctions = river_boundary_testfunctions

        self.alpha_trialfunctions = alpha_trialfunctions
        self.umom_testfunctions = umom_testfunctions
        self.beta_trialfunctions = beta_trialfunctions
        self.vmom_testfunctions = vmom_testfunctions
        self.gamma_trialfunctions = gamma_trialfunctions
        self.DIC_testfunctions = DIC_testfunctions

    # Public methods


    def setup_weak_form(self, static_condensation=False):
        a_total = ngsolve.BilinearForm(self.femspace, condense=static_condensation)

        if self.model_options['sea_boundary_treatment'] == 'exact':
            A_trial_functions, sea_bc_test_functions = self.A_trialfunctions, self.sea_boundary_testfunctions
        else:
            A_trial_functions, sea_bc_test_functions = None, None

        if self.model_options['river_boundary_treatment'] == 'exact':
            Q_trial_functions, river_bc_test_functions = self.Q_trialfunctions, self.river_boundary_testfunctions
            normal_alpha, normal_alpha_y = self.riverine_forcing.normal_alpha, self.riverine_forcing.normal_alpha_y
        else:
            Q_trial_functions, river_bc_test_functions, normal_alpha, normal_alpha_y = None, None, None, None

        weakforms.construct_non_linear_weak_form(a_total, self.model_options, self.geometric_information, self.numerical_information,
                                                 self.constant_physical_parameters, self.spatial_parameters, self.spatial_parameters_grad,
                                                 self.time_basis, self.vertical_basis,
                                                 self.alpha_trialfunctions, self.beta_trialfunctions, self.gamma_trialfunctions,
                                                 self.umom_testfunctions, self.vmom_testfunctions, self.DIC_testfunctions,
                                                 A_trial_functions=A_trial_functions, Q_trial_functions=Q_trial_functions,
                                                 sea_bc_test_functions=sea_bc_test_functions, river_bc_test_functions=river_bc_test_functions,
                                                 normal_alpha=normal_alpha, normal_alpha_y=normal_alpha_y, operator='full',
                                                 as_forcing_list=[], forcing_alpha=self.forcing_alpha, forcing_beta=self.forcing_beta,
                                                 forcing_gamma=self.forcing_gamma, forcing_Q=self.forcing_Q)

        self.total_bilinearform = a_total

    def restructure_solution(self):
        """Associates each part of the solution gridfunction vector to a Fourier and vertical eigenfunction pair."""

        M = self.numerical_information['M']
        imax = self.numerical_information['imax']

        self.alpha_solution = [dict() for _ in range(M)]
        self.beta_solution = [dict() for _ in range(M)]
        self.gamma_solution = dict()
        self.A_solution = dict()
        self.Q_solution = dict()

        for m in range(M):
            self.alpha_solution[m][0] = self.solution_gf.components[m * (2*imax + 1)]
            self.beta_solution[m][0] = self.solution_gf.components[(M + m) * (2*imax + 1)]
            
            for q in range(1, imax + 1):
                self.alpha_solution[m][-q] = self.solution_gf.components[m * (2*imax + 1) + q]
                self.alpha_solution[m][q] = self.solution_gf.components[m * (2*imax + 1) + imax + q]

                self.beta_solution[m][-q] = self.solution_gf.components[(M + m) * (2*imax + 1) + q]
                self.beta_solution[m][q] = self.solution_gf.components[(M + m) * (2*imax + 1) + imax + q]
        
        self.gamma_solution[0] = self.solution_gf.components[2*(M)*(2*imax+1)]

        for q in range(1, imax + 1):
            self.gamma_solution[-q] = self.solution_gf.components[2*(M)*(2*imax+1) + q]
            self.gamma_solution[q] = self.solution_gf.components[2*(M)*(2*imax+1) + imax + q]


        if self.model_options['sea_boundary_treatment'] == 'exact':
            self.A_solution[0] = self.solution_gf.components[2*(M)*(2*imax+1) + (2*imax + 1)]
            for q in range(1, imax + 1):
                self.A_solution[-q] = self.solution_gf.components[2*(M)*(2*imax + 1) + (2*imax + 1) + q]
                self.A_solution[q] = self.solution_gf.components[2*(M)*(2*imax + 1) + (2*imax + 1) + imax + q]

            if self.model_options['river_boundary_treatment'] == 'exact': # if both layers are handled, then river follows the sea
                self.Q_solution[0] = self.solution_gf.components[2*(M)*(2*imax+1) + 2*(2*imax + 1)]
                for q in range(1, imax + 1):
                    # self.Q_solution[-q] = self.solution_gf.components[2*(M)*(2*imax + 1) + 2*(2*imax + 1) + q]
                    # self.Q_solution[q] = self.solution_gf.components[2*(M)*(2*imax + 1) + 2*(2*imax + 1) + imax + q]
                    self.Q_solution[-q] = 0
                    self.Q_solution[q] = 0
        elif self.model_options['river_boundary_treatment'] == 'exact':
            self.Q_solution[0] = self.solution_gf.components[2*(M)*(2*imax+1) + (2*imax + 1)]
            for q in range(1, imax + 1):
                # self.Q_solution[-q] = self.solution_gf.components[2*(M)*(2*imax + 1) + (2*imax + 1) + q]
                # self.Q_solution[q] = self.solution_gf.components[2*(M)*(2*imax + 1) + (2*imax + 1) + imax + q]
                self.Q_solution[-q] = 0
                self.Q_solution[q] = 0


    def get_gradients(self, compiled=True):
        M = self.numerical_information['M']
        imax = self.numerical_information['imax']

        self.alpha_grad = [dict() for _ in range(M)]
        self.beta_grad = [dict() for _ in range(M)]
        self.gamma_grad = dict()
        
        for i in range(-imax, imax + 1):
            self.gamma_grad[i] = ngsolve.grad(self.gamma_solution[i]).Compile() if compiled else ngsolve.grad(self.gamma_solution[i])
            for m in range(M):
                self.alpha_grad[m][i] = ngsolve.grad(self.alpha_solution[m][i]).Compile() if compiled else ngsolve.grad(self.alpha_solution[m][i])
                self.beta_grad[m][i] = ngsolve.grad(self.beta_solution[m][i]).Compile() if compiled else ngsolve.grad(self.beta_solution[m][i])


    def save(self, name: str, solution_format='npy'):
        """Saves the hydrodynamics object. The folder to which the object is saved contains:
        
        - geometric_information.json:           json-formatted version of the geometric_information dictionary
        - numerical_information.json:           json-formatted version of the numerical_information dictionary
        - model_options.json:                   json-formatted version of the model_options dictionary
        - constant_physical_parameters.json:    json-formatted version of the constant_physical_parameters dictionary
        - solution.npy (or solution.txt):       solution vector (in .npy-format (memory-efficient) or .txt-format (human-readable))
        - spatial_parameters (folder):          folder containing pickled python functions for each spatial parameter          

        Arguments:

        - name: name of the folder the hydrodynamics-object will be saved to;
        - solution_format: format to which the solution vector is saved ('npy' or 'txt')

        """
        os.makedirs(name, exist_ok=True)

        # model options

        model_options = {}
        model_options.update(self.model_options)

        model_options['advection_influence_matrix'] = model_options['advection_influence_matrix'].tolist() # reformat advection_influence_matrix for json conversion
        model_options['surface_interaction_influence_matrix'] = model_options['surface_interaction_influence_matrix'].tolist()

        with open(f"{name}/model_options.json", 'x') as f_options:
            json.dump(model_options, f_options, indent=4)

        # geometric information

        with open(f"{name}/geometric_information.json", 'x') as f_geom:
            json.dump(self.geometric_information, f_geom, indent=4)

        # numerical information

        with open(f"{name}/numerical_information.json", 'x') as f_numeric:
            json.dump(self.numerical_information, f_numeric, indent=4)

        # constant physical parameters

        with open(f"{name}/constant_physical_parameters.json", 'x') as f_params:
            json.dump(self.constant_physical_parameters, f_params, indent=4)

        # spatial parameters

        os.makedirs(f'{name}/spatial_parameters')

        for paramname, value in self.spatial_parameters.items():
            with open(f'{name}/spatial_parameters/{paramname}.pkl', 'wb') as file:
                # cloudpickle.dump(value.fh, file, protocol=4)
                cloudpickle.dump(value, file, protocol=4)


        # solution

        save_gridfunction(self.solution_gf, f"{name}/solution", format=solution_format)


    def hrefine(self, threshold: float, numits: int = 1, based_on = 'bathygrad'):
        """
        Refines the mesh a number of iterations based on the following rule: if the integrated 'based_on'-quantity in a particular element exceeds 
        a threshold times the overall arithmetic average (of all elements) integrated 'based_on'-quantity in the mesh, that element is marked for
        refinement. This procedure is performed a user-specified number of times.

        Arguments:

            - threshold (float):    factor larger than one that is used for the refinement rule;
            - numits (int):         number of times the mesh is refined;
            - based_on (str):       integrable quantity the rule is based on; options are 'bathygrad' which bases the rule on the norm of the bathymetry gradient; 
        
        """
        if based_on == 'bathygrad':
            bathy_gradnorm = ngsolve.sqrt(self.spatial_parameters_grad['H'][0] * self.spatial_parameters_grad['H'][0] + 
                                          self.spatial_parameters_grad['H'][1] * self.spatial_parameters_grad['H'][1])
        else:
            raise ValueError("Invalid value for 'based_on'. Please choose from the following options: 'bathygrad'.")
            
        for _ in range(numits):

            num_refined = refine_mesh_by_elemental_integration(self.mesh, bathy_gradnorm, threshold)

           
            for name, param in self.spatial_parameters.items(): # SpatialParameter-objects need to be redefined on the new mesh
                self.spatial_parameters[name] = param

            bathy_gradnorm = ngsolve.sqrt(self.spatial_parameters_grad['H'][0] * self.spatial_parameters_grad['H'][0] + 
                                          self.spatial_parameters_grad['H'][1] * self.spatial_parameters_grad['H'][1])
                
            if num_refined == 0:
                break

        self.nfreedofs = count_free_dofs(self.femspace)
        

def load_hydrodynamics(name, solution_format='npy'):
    """Creates a Hydrodynamics object from a folder generated by the save-method of the Hydrodynamics object. This object can *only* be used for postprocessing.
    
    Arguments:
        - name:       name of the folder the data may be found in
        - solution_format: file extension of the solution vector; 'npy' or 'txt'.
        
    """

    with open(f"{name}/model_options.json", 'rb') as options_file:
        model_options = json.load(options_file)

    model_options['advection_influence_matrix'] = np.array(model_options['advection_influence_matrix'])
    model_options['surface_interaction_influence_matrix'] = np.array(model_options['surface_interaction_influence_matrix'])

    with open(f"{name}/geometric_information.json", 'rb') as geom_file:
        geometric_information = json.load(geom_file)

    with open(f"{name}/numerical_information.json", 'rb') as numeric_file:
        numerical_information = json.load(numeric_file)

    with open(f"{name}/constant_physical_parameters.json", 'rb') as params_file:
        constant_physical_parameters = json.load(params_file)

    spatial_parameters = {}
    for param in os.listdir(f'{name}/spatial_parameters'):
        filename = os.fsdecode(param)
        param_name = filename[:-4] # ignore file extension
        with open(f'{name}/spatial_parameters/{filename}', 'rb') as spatial_parameter_file:
            spatial_parameters[param_name] = cloudpickle.load(spatial_parameter_file)
            spatial_parameters[param_name].spacedim = 2 # cloudpickle forgets space dimension of CFs

    
    hydro = Hydrodynamics(model_options, geometric_information, numerical_information, constant_physical_parameters, spatial_parameters)
    hydro.solution_gf = ngsolve.GridFunction(hydro.femspace)
    load_basevector(hydro.solution_gf.vec, f"{name}/solution", format=solution_format)
    hydro.restructure_solution()
    hydro.get_gradients(compiled=True)

    return hydro



class DecomposedHydrodynamics(Hydrodynamics):


    def __init__(self, parent_hydro: Hydrodynamics, forcing_instruction: dict[dict]):

        self.parent_hydro = parent_hydro

        self.model_options = deepcopy(parent_hydro.model_options)
        self.model_options['sea_boundary_treatment'] = 'simple' # do not use internal boundary conditions for this model; these equations are non-linear and ruin the decomposition
        self.model_options['river_boundary_treatment'] = 'simple'

        self.geometric_information = deepcopy(parent_hydro.geometric_information) # don't use internal boundary conditions.
        # self.geometric_information['L_BL_sea'] = 0
        # self.geometric_information['L_R_sea'] = 0
        # self.geometric_information['L_R_sea'] = 0
        # self.geometric_information['L_BL_river'] = 0
        # self.geometric_information['L_R_river'] = 0
        # self.geometric_information['L_R_river'] = 0

        self.forcing_instruction = forcing_instruction

        self.numerical_information = deepcopy(parent_hydro.numerical_information)
        self.constant_physical_parameters = deepcopy(parent_hydro.constant_physical_parameters)
        self.spatial_parameters = set_spatial_physical_parameters(self.parent_hydro.spatial_parameters['H'])
        self.spatial_parameters_grad = get_spatial_parameter_gradients(self.spatial_parameters, self.parent_hydro.mesh)

        self.geom = parent_hydro.geom
        self.mesh = parent_hydro.mesh

        self.display_geom = parent_hydro.display_geom
        self.display_mesh = parent_hydro.display_mesh

        self.time_basis = parent_hydro.time_basis
        self.vertical_basis = parent_hydro.vertical_basis

        self._setup_fem_space()
        self._setup_TnT()
        self.solution_gf = ngsolve.GridFunction(self.femspace)
        self.restructure_solution()

        self.nfreedofs = count_free_dofs(self.femspace)
        self.is_decomposed = True

        self.forcing_alpha, self.forcing_beta, self.forcing_gamma, self.forcing_Q, self.forcing_A = self.collect_forcings()
        self.decomposition_depth = len(self.forcing_alpha)

        # self.merge_depth_and_residual_setup()


    def collect_forcings(self):
        """Collect solutions of hydrodynamics all the way up to the non-linear model using recursion"""
        if not self.parent_hydro.is_decomposed: # to do: make sure copies are made instead of references copied
            forcing_alpha = [self.parent_hydro.alpha_solution]
            forcing_beta = [self.parent_hydro.beta_solution]
            forcing_gamma = [{l: self.parent_hydro.gamma_solution[l] for l in range(-self.parent_hydro.numerical_information['imax'], self.parent_hydro.numerical_information['imax'] + 1)}]
            forcing_Q = [self.parent_hydro.Q_solution]
            forcing_A = [self.parent_hydro.A_solution]
        else:
            forcing_alpha = [self.parent_hydro.alpha_solution] + self.parent_hydro.collect_forcings()[0]
            forcing_beta = [self.parent_hydro.beta_solution] + self.parent_hydro.collect_forcings()[1]
            forcing_gamma = [self.parent_hydro.gamma_solution] + self.parent_hydro.collect_forcings()[2]
            forcing_Q = [self.parent_hydro.Q_solution] + self.parent_hydro.collect_forcings()[3]
            forcing_A = [self.parent_hydro.A_solution] + self.parent_hydro.collect_forcings()[4]

        return forcing_alpha, forcing_beta, forcing_gamma, forcing_Q, forcing_A
    

    def merge_depth_and_residual_setup(self):
        self.spatial_parameters['H'] += self.forcing_gamma[0][0]
        self.spatial_parameters_grad = get_spatial_parameter_gradients(self.spatial_parameters, self.mesh)
        self.forcing_gamma[0][0] = ngsolve.CF(0)
        self.forcing_gamma[0][0].spacedim = 2


    def setup_weak_form(self, static_condensation=True):
        a_total = ngsolve.BilinearForm(self.femspace, condense=static_condensation)
        rhs_total = ngsolve.LinearForm(self.femspace)

        if self.model_options['sea_boundary_treatment'] == 'exact':
            A_trial_functions, sea_bc_test_functions = self.A_trialfunctions, self.sea_boundary_testfunctions
        else:
            A_trial_functions, sea_bc_test_functions = None, None

        if self.model_options['river_boundary_treatment'] == 'exact':
            Q_trial_functions, river_bc_test_functions = self.Q_trialfunctions, self.river_boundary_testfunctions
            normal_alpha, normal_alpha_y = self.riverine_forcing.normal_alpha, self.riverine_forcing.normal_alpha_y
        else:
            Q_trial_functions, river_bc_test_functions, normal_alpha, normal_alpha_y = None, None, None, None

        weakforms.construct_non_linear_weak_form(a_total, self.model_options, self.geometric_information, self.numerical_information,
                                                 self.constant_physical_parameters, self.spatial_parameters, self.spatial_parameters_grad,
                                                 self.time_basis, self.vertical_basis,
                                                 self.alpha_trialfunctions, self.beta_trialfunctions, self.gamma_trialfunctions,
                                                 self.umom_testfunctions, self.vmom_testfunctions, self.DIC_testfunctions,
                                                 A_trial_functions=A_trial_functions, Q_trial_functions=Q_trial_functions,
                                                 sea_bc_test_functions=sea_bc_test_functions, river_bc_test_functions=river_bc_test_functions,
                                                 normal_alpha=normal_alpha, normal_alpha_y=normal_alpha_y, operator='linear_for_decomposition',
                                                 forcing_instruction=self.forcing_instruction, forcing_alpha=self.forcing_alpha, forcing_beta=self.forcing_beta,
                                                 forcing_gamma=self.forcing_gamma, forcing_Q=self.forcing_Q, linear_form=rhs_total)

        self.total_bilinearform = a_total
        self.total_linearform = rhs_total


    def solve(self, linear_solver='pardiso', print_log=True, num_threads=12, static_condensation=True):
        """User is resposible that every DecomposedHydrodynamics leads to a linear model!"""
        if print_log:
            print(f"Initiating solution procedure for the linear model forced by the following collection of terms: {self.forcing_instruction}. The total number of free degrees of freedom is {self.nfreedofs}.\n")

        if num_threads > 1:
            ngsolve.SetHeapSize(200_000_000)
            ngsolve.SetNumThreads(num_threads)

        context = ngsolve.TaskManager() if num_threads > 1 else nullcontext()

        # set up weak form
        weak_form_start = timeit.default_timer()
        with context:
            self.setup_weak_form()
        weak_form_time = timeit.default_timer() - weak_form_start
        if print_log:
            print(f"Setting up weak form took {np.round(weak_form_time, 3)} seconds.")

        # assemble vector and matrix
        assembly_start = timeit.default_timer()
        with context:
            self.total_bilinearform.Assemble()
        
            self.total_linearform.Assemble()
        assembly_time = timeit.default_timer() - assembly_start
        if print_log:
            print(f"Assembling weak form took {np.round(assembly_time, 3)} seconds.")

        # handle riverine boundary condition if the river bc is part of the forcing
        if self.parent_hydro.model_options['river_boundary_treatment'] == 'exact':
            self.river_interpolant = ((self.parent_hydro.geometric_information['L_BL_sea']/self.parent_hydro.geometric_information['x_scaling']) + (self.parent_hydro.geometric_information['L_R_sea']/self.parent_hydro.geometric_information['x_scaling']) + \
                                (self.parent_hydro.geometric_information['L_RA_sea']/self.parent_hydro.geometric_information['x_scaling']) + ngsolve.x) / \
                                ((self.parent_hydro.geometric_information['riverine_boundary_x']+self.parent_hydro.geometric_information['L_BL_river']+self.parent_hydro.geometric_information['L_R_river']+self.parent_hydro.geometric_information['L_RA_river'] +
                                self.parent_hydro.geometric_information['L_BL_sea'] + self.parent_hydro.geometric_information['L_R_sea'] + self.parent_hydro.geometric_information['L_RA_sea']) / self.parent_hydro.geometric_information['x_scaling'])

        if 'river_bc' in self.forcing_instruction.keys():
            if self.forcing_instruction['river_bc']['velocity'] > 0: 
                essential_bc_start = timeit.default_timer()
                for m in range(self.numerical_information['M']):
                    if self.parent_hydro.model_options['river_boundary_treatment'] == 'exact':
                        self.solution_gf.components[m * (2*self.numerical_information['imax'] + 1)].Set(self.forcing_alpha[self.forcing_instruction['river_bc']['velocity'] - 1][m][0] +
                                                                                                        self.forcing_Q[self.forcing_instruction['river_bc']['velocity'] - 1][0] * self.river_interpolant, ngsolve.BND)
                    else:
                        self.solution_gf.components[m * (2*self.numerical_information['imax'] + 1)].Set(self.parent_hydro.constant_physical_parameters['discharge'] * self.parent_hydro.riverine_forcing.normal_alpha[m], ngsolve.BND)

                rhs = self.total_linearform.vec.CreateVector()
                with context:
                    rhs.data = self.total_linearform.vec - self.total_bilinearform.mat * self.solution_gf.vec
                essential_bc_time = timeit.default_timer() - essential_bc_start
                if print_log:
                    print(f"Handling essential BC took {np.round(essential_bc_time, 3)} seconds.")
        else:
            rhs = self.total_linearform.vec.CreateVector()
            rhs.data = self.total_linearform.vec

        # convert to scipy-sparse matrix if we don't use the built-in solver
        if linear_solver != 'pardiso':
            conversion_start = timeit.default_timer()
            # Extract matrix and vector from ngsolve
            freedof_list = get_freedof_list(self.femspace.FreeDofs())
            mat = remove_fixeddofs_from_csr(basematrix_to_csr_matrix(self.total_bilinearform.mat), freedof_list)
            rhs_arr = rhs.FV().NumPy()[freedof_list]
            conversion_time = timeit.default_timer() - conversion_start
            if print_log:
                print(f"Conversion of equation to scipy-sparse took {np.round(conversion_time, 3)} seconds.")

        # invert system
        inversion_start = timeit.default_timer()
        if linear_solver == 'pardiso':
            with context:
                if static_condensation:
                    invS = self.total_bilinearform.mat.Inverse(freedofs=self.femspace.FreeDofs(coupling=True), inverse='pardiso')
                    ext = ngsolve.IdentityMatrix() + self.total_bilinearform.harmonic_extension
                    extT = ngsolve.IdentityMatrix() + self.total_bilinearform.harmonic_extension_trans
                    invA = ext @ invS @ extT + self.total_bilinearform.inner_solve
                    self.solution_gf.vec.data += invA * rhs
                else:
                    self.solution_gf.vec.data += self.total_bilinearform.mat.Inverse(freedofs=self.femspace.FreeDofs(), inverse='pardiso') * rhs
        elif linear_solver == 'scipy_direct':
            solver = scipyLU_solver
            sol = solver.solve(mat, rhs_arr, rcm=True)
            self.solution_gf.vec.FV().NumPy()[freedof_list] = sol
        elif linear_solver == 'pypardiso':
            solver = pypardiso_spsolve
            sol = solver.solve(mat, rhs_arr, rcm=True)
            self.solution_gf.vec.FV().NumPy()[freedof_list] = sol
        inversion_time = timeit.default_timer() - inversion_start

        self.restructure_solution()
        self.get_gradients()

        if print_log:
            print(f"Inversion of system took {np.round(inversion_time, 3)} seconds.")
            print("Solution procedure complete.\n")
        

def decompose_hydro(hydro: Hydrodynamics, forcings, **kwargs):
    contributions = {}
    for name, forcing in forcings.items():
        linear_hydro = DecomposedHydrodynamics(hydro, forcing)
        linear_hydro.solve(**kwargs)
        contributions[name] = linear_hydro
    return contributions


def save_decompositions(contributions: dict[str, Hydrodynamics], folder_name):
    os.makedirs(folder_name, exist_ok=True)
    for forcing_name, hydro in contributions.items():
        hydro.save(f'{folder_name}/{forcing_name}')


class RiverineForcing(object):

    
    def __init__(self, hydro: Hydrodynamics, discharge=None):

        """Sets up the riverine boundary condition for the residual component of the along-channel velocity, given
        a stationary river discharge, or a lateral structure of the (stationary) river discharge.
        
        Arguments:

        - hydro: Hydrodynamics-object this boundary condition is associated to
        - discharge: if this argument is None, the constant value of the river discharge from hydro.constant_physical_parameters is used;
          otherwise, the user can provide a manual lateral structure for the river discharge as an ngsolve.CoefficientFunction. A manual
          vertical distribution is not (yet) implemented.
        
        """

        # initialise parameters
        
        self.hydro = hydro

        manual = True
        if discharge is None:
            discharge = hydro.constant_physical_parameters['discharge']
            manual = False

        Av = hydro.constant_physical_parameters['Av']

        if self.hydro.model_options['bed_bc'] == 'partial_slip':
            sf = hydro.constant_physical_parameters['sf']

        H = self.hydro.spatial_parameters['H']
        Hy = self.hydro.spatial_parameters_grad['H'][1]
        R = self.hydro.spatial_parameters['R']
        Ry = self.hydro.spatial_parameters_grad['H'][1]

        if manual:
            self.discharge_cf = discharge
        else:
            self.discharge = discharge * np.sqrt(2)

            # integrate (H+R)^2 over width
            y = np.linspace(-0.5, 0.5, 1001)
            dy = y[1] - y[0]

            eval_integrand = evaluate_CF_range((H+R)*(H+R), hydro.mesh, np.ones_like(y), y) # currently only works for unit square domains!
            integral = dy * eval_integrand.sum() # numerical integration with leftpoint rule

            self.discharge_cf = np.sqrt(2) * (H+R) * (H+R) / (hydro.geometric_information['y_scaling'] * integral)

        # project vertical structure onto vertical basis

        if self.hydro.model_options['bed_bc'] == 'no_slip':

            def vertical_structure(z):
                return 1 - z**2
        
        elif self.hydro.model_options['bed_bc'] == 'partial_slip':

            def vertical_structure(z):
                return 2*Av/sf + 1 - z**2
            
        projection = truncationbasis.Projection(vertical_structure, hydro.vertical_basis, hydro.numerical_information['M'])
        projection.construct_analytical_massmatrix()
        projection.project_galerkin(10, 30, sparse=False) # matrix is so small that a sparse structure is completely unnecessary

        self.normal_alpha = []
        self.normal_alpha_y = []

        if self.hydro.model_options['bed_bc'] == 'no_slip':
            for m in range(hydro.numerical_information['M']):
                self.normal_alpha.append(-projection.coefficients[m] * 1.5 * self.discharge_cf / (H + R))
                self.normal_alpha_y.append(-projection.coefficients[m] * 1.5 * np.sqrt(2) * (Hy + Ry) / (hydro.geometric_information['y_scaling'] * integral))

        elif self.hydro.model_options['bed_bc'] == 'partial_slip':
            for m in range(hydro.numerical_information['M']):
                self.normal_alpha.append(-projection.coefficients[m] * (3 * sf * self.discharge_cf) / ((H+R)*(2*sf + 6*Av)))
                self.normal_alpha_y.append(-projection.coefficients[m] * (3 * sf * (np.sqrt(2) * (Hy + Ry)/(hydro.geometric_information['y_scaling']*integral)) / (2*sf + 6*Av)))


class SeawardForcing(object):

    def __init__(self, hydro: Hydrodynamics):
        """Object containing information about the seaward boundary condition.
        
        Arguments:

        - hydro: Hydrodynamics-object this boundary condition is associated to.
        """
        self.hydro = hydro

        # Fill amplitudes and phases with zeros in the places where they are not prescribed
        self.amplitudes = hydro.constant_physical_parameters['seaward_amplitudes']
        self.phases = hydro.constant_physical_parameters['seaward_phases']
        for _ in range(hydro.numerical_information['imax'] + 1 - len(self.amplitudes)):
            self.amplitudes.append(0)
            self.phases.append(0)

        self.cfdict = {0: self.amplitudes[0] * np.sqrt(2)}
        self.boundaryCFdict = {0: hydro.mesh.BoundaryCF({BOUNDARY_DICT[SEA]: self.cfdict[0]}, default=0)}
        for i in range(1, hydro.numerical_information['imax'] + 1):
            self.cfdict[i] = self.amplitudes[i] * ngsolve.cos(self.phases[i-1]) # phase for residual flow is not defined, therefore list index i - 1
            self.cfdict[-i] = -self.amplitudes[i] * ngsolve.sin(self.phases[i-1])

            self.boundaryCFdict[i] = hydro.mesh.BoundaryCF({BOUNDARY_DICT[SEA]: self.cfdict[i]}, default=0)
            self.boundaryCFdict[-i] = hydro.mesh.BoundaryCF({BOUNDARY_DICT[SEA]: self.cfdict[-i]}, default=0)






            



        
        
