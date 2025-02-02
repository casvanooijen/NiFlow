import numpy as np
import os
import json
import cloudpickle
import ngsolve

import NiFlow.truncationbasis.truncationbasis as truncationbasis
from NiFlow.geometry.create_geometry import parametric_geometry, RIVER, SEA, WALL, WALLUP, WALLDOWN, BOUNDARY_DICT
from NiFlow.geometry.meshing import generate_mesh
from NiFlow.geometry.geometries import *
from NiFlow.spatial_parameter.boundary_fitted_coordinates import generate_bfc
from NiFlow.spatial_parameter.spatial_parameter import SpatialParameter
import NiFlow.define_weak_forms as weakforms
from NiFlow.utils import *


def select_model_options(bed_bc:str = 'no_slip', veddy_viscosity_assumption:str = 'constant',
                         advection_influence_matrix: np.ndarray = None, sea_boundary_treatment:str = 'exact',
                         river_boundary_treatment:str = 'simple'):
    
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
        """
    
    if bed_bc == 'partial_slip' and veddy_viscosity_assumption == 'constant':
        raise ValueError("Partial-slip condition and constant vertical eddy viscosity are incompatible")

    options = {
            'bed_bc': bed_bc,
            'veddy_viscosity_assumption': veddy_viscosity_assumption,
            'advection_influence_matrix': advection_influence_matrix, # the validity of this matrix is checked when imax is know, i.e. when the hydrodynamics object is initialised
            'sea_boundary_treatment': sea_boundary_treatment,
            'river_boundary_treatment': river_boundary_treatment
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
                                     sf: float=1000000., advection_epsilon: float=1.):
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
        'advection_epsilon': advection_epsilon
    }

    return params


def set_spatial_physical_parameters(H, R, rho):
    """Returns a dictionary of spatially varying parameters. These parameters must be defined as python functions in terms of the curvilinear
    coordinates xi and eta, 'xi' being the along-channel coordinate running from 0 to 1 (this also includes the ramping zone), and 'eta' being
    the cross-channel coordinate running from -0.5 to 0.5. The function must make use of sympy-functions to define the parameter. Please **do not** use outer scope variables in the definitions of these functions;
    this will mess up the saving/loading of models through pickling. This can always be avoided by using function closures.

    These functions are converted to ngsolve.CoefficientFunction-objects in the initialisation of the Hydrodynamics-object.

    Arguments: 

    - H: water depth below 0.
    - R: reference water level.
    - rho: vertically constant reference level.
    
    """

    param_function_handles = {
        'H': H,
        'R': R,
        'rho': rho
    }
    return param_function_handles



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

        self.model_options = model_options
        self.geometric_information = geometric_information
        self.numerical_information = numerical_information
        self.constant_physical_parameters = constant_physical_parameters

        # MAKE THE GEOMETRY FROM GEOMETRIC_INFORMATION

        if geometric_information['shape'] == 'rectangle':
            if model_options['river_boundary_treatment'] == 'exact':
                geomcurves = parametric_rectangle(geometric_information['shape_parameters'][0] / geometric_information['x_scaling'], 
                                                geometric_information['shape_parameters'][1] / geometric_information['y_scaling'],
                                                geometric_information['L_BL_sea'] / geometric_information['x_scaling'],
                                                geometric_information['L_R_sea'] / geometric_information['x_scaling'],
                                                geometric_information['L_RA_sea'] / geometric_information['x_scaling'],
                                                geometric_information['L_BL_river'] / geometric_information['x_scaling'],
                                                geometric_information['L_R_river'] / geometric_information['x_scaling'],
                                                geometric_information['L_RA_river'] / geometric_information['x_scaling'])
            elif model_options['river_boundary_treatment'] == 'simple': # ramping is done inside the domain if river_boundary_treatment is 'simple'
                geomcurves = parametric_rectangle(geometric_information['shape_parameters'][0] / geometric_information['x_scaling'], 
                                                geometric_information['shape_parameters'][1] / geometric_information['y_scaling'],
                                                geometric_information['L_BL_sea'] / geometric_information['x_scaling'],
                                                geometric_information['L_R_sea'] / geometric_information['x_scaling'],
                                                geometric_information['L_RA_sea'] / geometric_information['x_scaling'], 0, 0, 0)
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

        # CONVERT SPATIAL PARAMETERS TO COEFFICIENTFUNCTIONS

        self.spatial_parameters = {}
        bfc = generate_bfc(self.mesh, numerical_information['order'], method='laplace')

        self.spatial_parameters['H'] = SpatialParameter(spatial_physical_parameters['H'], bfc)
        self.spatial_parameters['R'] = SpatialParameter(spatial_physical_parameters['R'], bfc)
        self.spatial_parameters['rho'] = SpatialParameter(spatial_physical_parameters['rho'], bfc)

        self.sympy_spatial_parameters = spatial_physical_parameters

        # SET BOUNDARY CONDITIONS

        self._set_seaward_boundary_condition()
        self._set_riverine_boundary_condition()

        # INITIALISE FINITE ELEMENT SPACE AND TEST-/TRIALFUNCTIONS

        self._setup_fem_space()
        self._setup_TnT()

        self.nfreedofs = count_free_dofs(self.femspace)


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
        
        U = ngsolve.H1(self.mesh, order= 2 if (self.numerical_information['element_type'] == 'taylor-hood' and self.numerical_information['order']==1) else self.numerical_information['order'], dirichlet=f"{BOUNDARY_DICT[RIVER]}")  # make sure that zero-th order is not used for the free surface
        V = ngsolve.H1(self.mesh, order= 2 if (self.numerical_information['element_type'] == 'taylor-hood' and self.numerical_information['order']==1) else self.numerical_information['order'], dirichlet=f"{BOUNDARY_DICT[WALLDOWN]}|{BOUNDARY_DICT[WALLUP]}")

        # add interior bubble functions if MINI-elements are used
        if self.numerical_information['element_type'] == 'MINI':
            U.SetOrder(ngsolve.TRIG, 3 if self.numerical_information['order'] == 1 else self.numerical_information['order'] + 1)
            V.SetOrder(ngsolve.TRIG, 3 if self.numerical_information['order'] == 1 else self.numerical_information['order'] + 1)

            U.Update()
            V.Update()

        # define Z-space with order one less than velocity space in case of Taylor-Hood elements or MINI (k>1) elements
        if ((self.numerical_information['element_type'] == 'taylor-hood') or (self.numerical_information['element_type'] == 'MINI')) and self.numerical_information['order'] > 1:
            Z = ngsolve.H1(self.mesh, order=self.numerical_information['order'] - 1, dirichlet=BOUNDARY_DICT[SEA])
        else:
            Z = ngsolve.H1(self.mesh, order=self.numerical_information['order'], dirichlet=BOUNDARY_DICT[SEA])

        if self.model_options['sea_boundary_treatment'] == 'exact' or self.model_options['river_boundary_treatment'] == 'exact': # take into account floating point errors
            scalarFESpace = ngsolve.NumberSpace(self.mesh)
        
        list_of_spaces = [U for _ in range(M*(2*imax + 1))]
        for _ in range(M*(2*imax + 1)): 
            list_of_spaces.append(V)
        for _ in range(2*imax + 1):
            list_of_spaces.append(Z)

        if self.model_options['sea_boundary_treatment'] == 'exact' and self.model_options['river_boundary_treatment'] == 'exact': # if we treat the boundary on both sides
            for _ in range(2 * (2 * imax + 1)):
                list_of_spaces.append(scalarFESpace)
        elif self.model_options['sea_boundary_treatment'] != 'exact' and self.model_options['river_boundary_treatment'] != 'exact': # if we do not treat the boundaries
            pass
        elif self.model_options['sea_boundary_treatment'] == 'exact' and self.model_options['river_boundary_treatment'] != 'exact':
            for _ in range(2 * imax + 1): # if we have ramping on only one side
                list_of_spaces.append(scalarFESpace)
        elif self.model_options['sea_boundary_treatment'] != 'exact' and self.model_options['river_boundary_treatment'] == 'exact':
            for _ in range(2 * imax + 1): # if we have ramping on only one side
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

                for i in range(1, imax + 1):
                    Q_trialfunctions[-i] = trialtuple[2*M*num_time_components + 2*num_time_components + i]
                    Q_trialfunctions[i] = trialtuple[2*M*num_time_components + 2*num_time_components + imax + i]

                    river_boundary_testfunctions[-i] = testtuple[2*M*num_time_components + 2*num_time_components + i]
                    river_boundary_testfunctions[i] = testtuple[2*M*num_time_components + 2*num_time_components + imax + i]

                self.Q_trialfunctions = Q_trialfunctions
                self.river_boundary_testfunctions = river_boundary_testfunctions

        elif self.model_options['river_boundary_treatment'] == 'exact': # if only the river side is ramped
                Q_trialfunctions = dict()
                river_boundary_testfunctions = dict()

                Q_trialfunctions[0] = trialtuple[2*M*num_time_components + 2*num_time_components]
                river_boundary_testfunctions[0] = testtuple[2*M*num_time_components + 2*num_time_components]

                for i in range(1, imax + 1):
                    Q_trialfunctions[-i] = trialtuple[2*M*num_time_components + 2*num_time_components + i]
                    Q_trialfunctions[i] = trialtuple[2*M*num_time_components + 2*num_time_components + imax + i]

                    river_boundary_testfunctions[-i] = testtuple[2*M*num_time_components + 2*num_time_components + i]
                    river_boundary_testfunctions[i] = testtuple[2*M*num_time_components + 2*num_time_components + imax + i]

                self.Q_trialfunctions = Q_trialfunctions
                self.river_boundary_testfunctions = river_boundary_testfunctions

        self.alpha_trialfunctions = alpha_trialfunctions
        self.umom_testfunctions = umom_testfunctions
        self.beta_trialfunctions = beta_trialfunctions
        self.vmom_testfunctions = vmom_testfunctions
        self.gamma_trialfunctions = gamma_trialfunctions
        self.DIC_testfunctions = DIC_testfunctions

    # Public methods

    def setup_weak_form(self):
        a_total = ngsolve.BilinearForm(self.femspace)

        if self.model_options['sea_boundary_treatment'] == 'exact':
            if self.model_options['river_boundary_treatment'] == 'exact':
                weakforms.add_weak_form(a_total, self.model_options, self.numerical_information, self.geometric_information, self.constant_physical_parameters, self.spatial_parameters,
                                        self.alpha_trialfunctions, self.beta_trialfunctions, self.gamma_trialfunctions,
                                        self.umom_testfunctions, self.vmom_testfunctions, self.DIC_testfunctions,
                                        self.vertical_basis, self.time_basis, self.riverine_forcing.normal_alpha, self.riverine_forcing.normal_alpha_y, only_linear=False,
                                        A_trialfunctions=self.A_trialfunctions, sea_boundary_testfunctions=self.sea_boundary_testfunctions,
                                        Q_trialfunctions=self.Q_trialfunctions, river_boundary_testfunctions=self.river_boundary_testfunctions)
            else:
                weakforms.add_weak_form(a_total, self.model_options, self.numerical_information, self.geometric_information, self.constant_physical_parameters, self.spatial_parameters,
                                        self.alpha_trialfunctions, self.beta_trialfunctions, self.gamma_trialfunctions,
                                        self.umom_testfunctions, self.vmom_testfunctions, self.DIC_testfunctions,
                                        self.vertical_basis, self.time_basis, self.riverine_forcing.normal_alpha, self.riverine_forcing.normal_alpha_y, only_linear=False,
                                        A_trialfunctions=self.A_trialfunctions, sea_boundary_testfunctions=self.sea_boundary_testfunctions)
        elif self.model_options['river_boundary_treatment'] == 'exact':
            weakforms.add_weak_form(a_total, self.model_options, self.numerical_information, self.geometric_information, self.constant_physical_parameters, self.spatial_parameters,
                                    self.alpha_trialfunctions, self.beta_trialfunctions, self.gamma_trialfunctions,
                                    self.umom_testfunctions, self.vmom_testfunctions, self.DIC_testfunctions,
                                    self.vertical_basis, self.time_basis, self.riverine_forcing.normal_alpha, self.riverine_forcing.normal_alpha_y, only_linear=False,
                                    Q_trialfunctions=self.Q_trialfunctions, river_boundary_testfunctions=self.river_boundary_testfunctions)
        else:
            weakforms.add_weak_form(a_total, self.model_options, self.numerical_information, self.geometric_information, self.constant_physical_parameters, self.spatial_parameters,
                                    self.alpha_trialfunctions, self.beta_trialfunctions, self.gamma_trialfunctions,
                                    self.umom_testfunctions, self.vmom_testfunctions, self.DIC_testfunctions,
                                    self.vertical_basis, self.time_basis, self.riverine_forcing.normal_alpha, only_linear=False)
        
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
                    self.Q_solution[-q] = self.solution_gf.components[2*(M)*(2*imax + 1) + 2*(2*imax + 1) + q]
                    self.Q_solution[q] = self.solution_gf.components[2*(M)*(2*imax + 1) + 2*(2*imax + 1) + imax + q]
        elif self.model_options['river_boundary_treatment'] == 'exact':
            self.Q_solution[0] = self.solution_gf.components[2*(M)*(2*imax+1) + (2*imax + 1)]
            for q in range(1, imax + 1):
                self.Q_solution[-q] = self.solution_gf.components[2*(M)*(2*imax + 1) + (2*imax + 1) + q]
                self.Q_solution[q] = self.solution_gf.components[2*(M)*(2*imax + 1) + (2*imax + 1) + imax + q]


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
                cloudpickle.dump(value.fh, file, protocol=4)

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
            bathy_gradnorm = ngsolve.sqrt(self.spatial_parameters['H'].gradient_cf[0] * self.spatial_parameters['H'].gradient_cf[0] + 
                                          self.spatial_parameters['H'].gradient_cf[1] * self.spatial_parameters['H'].gradient_cf[1])
        else:
            raise ValueError("Invalid value for 'based_on'. Please choose from the following options: 'bathygrad'.")
            
        for _ in range(numits):

            num_refined = refine_mesh_by_elemental_integration(self.mesh, bathy_gradnorm, threshold)

           
            for name, param in self.spatial_parameters.items(): # SpatialParameter-objects need to be redefined on the new mesh
                bfc = generate_bfc(self.mesh, self.numerical_information['order'], 'diffusion')
                self.spatial_parameters[name] = SpatialParameter(param.fh, bfc)

            bathy_gradnorm = ngsolve.sqrt(self.spatial_parameters['H'].gradient_cf[0] * self.spatial_parameters['H'].gradient_cf[0] + 
                                          self.spatial_parameters['H'].gradient_cf[1] * self.spatial_parameters['H'].gradient_cf[1])
                
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

    
    hydro = Hydrodynamics(model_options, geometric_information, numerical_information, constant_physical_parameters, spatial_parameters)
    hydro.solution_gf = ngsolve.GridFunction(hydro.femspace)
    load_basevector(hydro.solution_gf.vec, f"{name}/solution", format=solution_format)
    hydro.restructure_solution()

    return hydro


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

        H = self.hydro.spatial_parameters['H'].cf
        Hy = self.hydro.spatial_parameters['H'].gradient_cf[1]
        R = self.hydro.spatial_parameters['R'].cf
        Ry = self.hydro.spatial_parameters['H'].gradient_cf[1]

        if manual:
            self.discharge_cf = discharge
        else:
            self.discharge = discharge

            # integrate (H+R)^2 over width
            y = np.linspace(-0.5, 0.5, 1001)
            dy = y[1] - y[0]

            eval_integrand = evaluate_CF_range((H+R)*(H+R), hydro.mesh, np.ones_like(y), y) # currently only works for unit square domains!
            integral = dy * eval_integrand.sum() # numerical integration with leftpoint rule

            self.discharge_cf = (H+R) * (H+R) / (hydro.geometric_information['y_scaling'] * integral)

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
                self.normal_alpha_y.append(-projection.coefficients[m] * 1.5 * (Hy + Ry) / (hydro.geometric_information['y_scaling'] * integral))

        elif self.hydro.model_options['bed_bc'] == 'partial_slip':
            for m in range(hydro.numerical_information['M']):
                self.normal_alpha.append(-projection.coefficients[m] * (3 * sf * self.discharge_cf) / ((2*sf + 6*Av)))
                self.normal_alpha_y.append(-projection.coefficients[m] * (3 * sf * ((Hy + Ry)/(hydro.geometric_information['y_scaling']*integral)) / (2*sf + 6*Av)))


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

        self.cfdict = {0: self.amplitudes[0]}
        self.boundaryCFdict = {0: hydro.mesh.BoundaryCF({BOUNDARY_DICT[SEA]: self.cfdict[0]}, default=0)}
        for i in range(1, hydro.numerical_information['imax'] + 1):
            self.cfdict[i] = self.amplitudes[i] * ngsolve.cos(self.phases[i-1]) # phase for residual flow is not defined, therefore list index i - 1
            self.cfdict[-i] = self.amplitudes[i] * ngsolve.sin(self.phases[i-1])

            self.boundaryCFdict[i] = hydro.mesh.BoundaryCF({BOUNDARY_DICT[SEA]: self.cfdict[i]}, default=0)
            self.boundaryCFdict[-i] = hydro.mesh.BoundaryCF({BOUNDARY_DICT[SEA]: self.cfdict[-i]}, default=0)
