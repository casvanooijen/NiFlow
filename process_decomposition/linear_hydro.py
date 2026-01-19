import numpy as np
import ngsolve
from copy import deepcopy

from NiFlow.utils import *
from NiFlow.hydrodynamics import Hydrodynamics
from NiFlow.truncationbasis.truncationbasis import *
from NiFlow.geometry.create_geometry import BOUNDARY_DICT, SEA, RIVER, WALLUP, WALLDOWN


def construct_mu_nu(parent_hydro: Hydrodynamics):
    M = parent_hydro.numerical_information['M']
    imax = parent_hydro.numerical_information['imax']
    G1 = parent_hydro.vertical_basis.inner_product
    G3 = parent_hydro.vertical_basis.tensor_dict['G3']
    G4 = parent_hydro.vertical_basis.tensor_dict['G4']

    g = parent_hydro.constant_physical_parameters['g']
    sigma = parent_hydro.constant_physical_parameters['sigma']
    Av = parent_hydro.constant_physical_parameters['Av']
    H = parent_hydro.spatial_parameters['H'].cf

    mu = {k: [G4(m) * g * H / (G3(m, m) * Av + (G1(m,m)*2*np.pi*sigma*k*H)**2/(G3(m,m)*Av)) for m in range(M)] for k in range(imax + 1)}
    nu = {k: [G1(m,m)*2*np.pi*sigma*k*H/(G3(m,m)*Av) for m in range(M)] for k in range(imax+1)}

    # compile mu- and nu-CoefficientFunctions for faster evaluation
    for k in range(imax + 1):
        for m in range(M):
            mu[k][m].Compile()
            nu[k][m].Compile()

    return mu, nu


class Forcing(object):

    """

        Class stores data of a specific forcing mechanism, to be used in the decomposition. Contains body forcing, horizontal BC forcing.
        Body forcing part is a list of `lambda sig: ngsolve.CF(...)*f(sig)` functions per tidal component (starting from M0, M2, M4, ...).
        Must be provided when constructing Forcing object. For all forcings we consider, analytical expressions (given the non-linear solution)
        can be computed.
        Boundary (both sea and river) is a list of ngsolve CoefficientFunctions per tidal component.

    """

    def __init__(self, name, forcing_type, forcing_func, depth_integrated_forcing_func=0):
        self.name = name
        self.forcing_type = forcing_type #body, sea_boundary, or river_boundary
        self.forcing_func = forcing_func
        self.depth_integrated_forcing_func = depth_integrated_forcing_func


class LinearForcedHydrodynamics(object):

    def __init__(self, forcing: Forcing, parent_hydro: Hydrodynamics, order):
        self.forcing = forcing
        self.parent_hydro = parent_hydro
        self.order = order

        self._setup_fespace()
        self.mu, self.nu = construct_mu_nu(self.parent_hydro)
        self._setup_forms()
        self._assemble_forms()


    # def _Dn(self, n: int):

    #     sf = self.parent_hydro.constant_physical_parameters['sf']
    #     Av = self.parent_hydro.constant_physical_parameters['Av']
    #     H = self.parent_hydro.spatial_parameters['H'].cf
    #     sigma = self.parent_hydro.constant_physical_parameters['sigma']
    #     g = self.parent_hydro.constant_physical_parameters['g']

    #     if not isinstance(n, int) or n < 0:
    #         raise ValueError(f'Invalid value of n ({n}). Please provide positive integer')
    #     elif n == 0:
    #         return H * g * (-1 / (3*Av) + 1/sf)
    #     else:
    #         mu = ngsolve.sqrt(H / Av * 2 * np.pi * 1j * n * sigma)
    #         return -1j / (2*np.pi * n * sigma) / (2j*Av*mu*ngsolve.sin(mu) + 2*sf*ngsolve.cos(mu)) * ngsolve.sinh(mu) / mu * g  


    # def _dn(self, n: int):

    #     sf = self.parent_hydro.constant_physical_parameters['sf']
    #     Av = self.parent_hydro.constant_physical_parameters['Av']
    #     H = self.parent_hydro.spatial_parameters['H'].cf
    #     sigma = self.parent_hydro.constant_physical_parameters['sigma']
    #     g = self.parent_hydro.constant_physical_parameters['g']

    #     if not isinstance(n, int) or n < 0:
    #         raise ValueError(f'Invalid value of n ({n}). Please provide positive integer')
    #     elif n == 0:
    #         return lambda sig: g * (sig**2 * (1/(2*Av)) + 1/sf - 1/(2*Av))
    #     else:
    #         mu = ngsolve.sqrt(H / Av * 2 * np.pi * 1j * n * sigma)
    #         return lambda sig: CF_times_arr(-1j / (2*np.pi * n * sigma) / (2j*Av*mu*ngsolve.sin(mu) + 2*sf*ngsolve.cos(mu)) * g, cosh_of_arr(CF_times_arr(mu, sig)))       


    def _setup_fespace(self):

        M = self.parent_hydro.numerical_information['M']
        imax = self.parent_hydro.numerical_information['imax']

        mesh = self.parent_hydro.display_mesh # use mesh without ramping zone and use actual solutions as boundary conditions
        self.Zspace = ngsolve.H1(mesh, order = self.order, dirichlet=f'{BOUNDARY_DICT[SEA]}')
        self.Zspace_nonstationary = ngsolve.FESpace([self.Zspace, self.Zspace])
        self.Zspace_nodirichlet = ngsolve.H1(mesh, order=self.order)

        self.Z = {k: self.Zspace_nonstationary.TrialFunction() for k in range(imax + 1)}
        self.test_func = {k: self.Zspace_nonstationary.TestFunction() for k in range(imax + 1)}

        # replace stationary component with a single trial/test function per vertical component
        self.Z[0] = self.Zspace.TrialFunction()
        self.test_func[0] = self.Zspace.TestFunction()

    
    def _setup_forms(self):
        
        M = self.parent_hydro.numerical_information["M"]
        imax = self.parent_hydro.numerical_information['imax']

        self.bilinear_forms = {k: ngsolve.BilinearForm(self.Zspace_nonstationary) for k in range(imax + 1)}
        self.linear_forms = {k: ngsolve.LinearForm(self.Zspace_nonstationary) for k in range(imax + 1)}

        self.bilinear_forms[0] = ngsolve.BilinearForm(self.Zspace)
        self.linear_forms[0] = ngsolve.LinearForm(self.Zspace)

        G4 = self.parent_hydro.vertical_basis.tensor_dict['G4']

        # subtidal component
        L = self.parent_hydro.geometric_information['x_scaling']
        B = self.parent_hydro.geometric_information['y_scaling']
        H = self.parent_hydro.spatial_parameters['H'].cf

        sigma = self.parent_hydro.constant_physical_parameters['sigma']
        # self.bilinear_forms[0] += -self._Dn(0) * (ngsolve.grad(self.Z[0])[0] * ngsolve.grad(self.test_func[0])[0] / (L**2) + ngsolve.grad(self.Z[0])[1] * ngsolve.grad(self.test_func[0])[1] / (B**2)) * ngsolve.dx
        
        self.bilinear_forms[0] += -H * sum(self.mu[0][m] * G4(m) for m in range(M)) * (ngsolve.grad(self.Z[0])[0] * ngsolve.grad(self.test_func[0])[0] / (L**2) +
                                                          ngsolve.grad(self.Z[0])[1] * ngsolve.grad(self.test_func[0])[0] / (B**2)) * ngsolve.dx

        if self.forcing.forcing_type == 'body':
            self.linear_forms[0] += H * (self.forcing.depth_integrated_forcing_func[0][0] * ngsolve.grad(self.test_func[0])[0] / L + self.forcing.depth_integrated_forcing_func[1][0] * ngsolve.grad(self.test_func[0])[1] / B) * ngsolve.dx
            self.linear_forms[0] += -H * (self.forcing.depth_integrated_forcing_func[0][0] * self.test_func[0] / L) * ngsolve.ds(BOUNDARY_DICT[RIVER])
            self.linear_forms[0] += -H * (self.forcing.depth_integrated_forcing_func[1][0] * self.test_func[0] / B) * ngsolve.ds(BOUNDARY_DICT[WALLUP])
            self.linear_forms[0] += H * (self.forcing.depth_integrated_forcing_func[1][0] * self.test_func[0] / B) * ngsolve.ds(BOUNDARY_DICT[WALLDOWN])

        if self.forcing.forcing_type == 'river_boundary':
            self.linear_forms[0] += -H * self.forcing.forcing_func[0] * self.test_func[0] / L * ngsolve.ds(BOUNDARY_DICT[RIVER])

        # tidal components
        for k in range(1, self.parent_hydro.numerical_information['imax'] + 1):
            if self.forcing.forcing_type == 'body':
                self.linear_forms[k] += H * (self.forcing.depth_integrated_forcing_func[0][-k] * ngsolve.grad(self.test_func[k][0])[0] / L + self.forcing.depth_integrated_forcing_func[1][-k] * ngsolve.grad(self.test_func[k][0])[1] / B) * ngsolve.dx
                self.linear_forms[k] += -H * (self.forcing.depth_integrated_forcing_func[0][-k] * self.test_func[k][0] / L) * ngsolve.ds(BOUNDARY_DICT[RIVER])
                self.linear_forms[k] += -H * (self.forcing.depth_integrated_forcing_func[1][-k] * self.test_func[k][0] / B) * ngsolve.ds(BOUNDARY_DICT[WALLUP])
                self.linear_forms[k] += H * (self.forcing.depth_integrated_forcing_func[1][-k] * self.test_func[k][0] / B) * ngsolve.ds(BOUNDARY_DICT[WALLDOWN])
                
                self.linear_forms[k] += H * (self.forcing.depth_integrated_forcing_func[0][k] * ngsolve.grad(self.test_func[k][1])[0] / L + self.forcing.depth_integrated_forcing_func[1][k] * ngsolve.grad(self.test_func[k][1])[1] / B) * ngsolve.dx
                self.linear_forms[k] += -H * (self.forcing.depth_integrated_forcing_func[0][k] * self.test_func[k][1] / L) * ngsolve.ds(BOUNDARY_DICT[RIVER])
                self.linear_forms[k] += -H * (self.forcing.depth_integrated_forcing_func[1][k] * self.test_func[k][1] / B) * ngsolve.ds(BOUNDARY_DICT[WALLUP])
                self.linear_forms[k] += H * (self.forcing.depth_integrated_forcing_func[1][k] * self.test_func[k][1] / B) * ngsolve.ds(BOUNDARY_DICT[WALLDOWN])
                # forcing_grad_u_sine, forcing_grad_v_sine, forcing_grad_u_cosine, forcing_grad_v_cosine = \
                #     ngsolve.GridFunction(self.Zspace_nodirichlet), ngsolve.GridFunction(self.Zspace_nodirichlet), ngsolve.GridFunction(self.Zspace_nodirichlet), ngsolve.GridFunction(self.Zspace_nodirichlet)
                # forcing_grad_u_sine.Set(H * self.forcing.depth_integrated_forcing_func[0][-k])
                # forcing_grad_v_sine.Set(H * self.forcing.depth_integrated_forcing_func[1][-k])
                # forcing_grad_u_cosine.Set(H * self.forcing.depth_integrated_forcing_func[0][k])
                # forcing_grad_v_cosine.Set(H * self.forcing.depth_integrated_forcing_func[1][k])


                # self.linear_forms[k] += -1 * (ngsolve.grad(forcing_grad_u_sine)[0] / L + ngsolve.grad(forcing_grad_v_sine)[1] / B) * self.test_func[k][0] * ngsolve.dx
                # self.linear_forms[k] += -1 * (ngsolve.grad(forcing_grad_u_cosine)[0] / L + ngsolve.grad(forcing_grad_v_cosine)[1] / B) * self.test_func[k][1] * ngsolve.dx

            # time derivative in equation sine component
            self.bilinear_forms[k] += 2 * np.pi * sigma * k * self.Z[k][1] * self.test_func[k][0] * ngsolve.dx
            # time derivative in equation cosine component
            self.bilinear_forms[k] += -2 * np.pi * sigma * k * self.Z[k][0] * self.test_func[k][1] * ngsolve.dx

            # transport divergence from sine component in equation sine component
            self.bilinear_forms[k] += -H * sum(self.mu[k][m] * G4(m) for m in range(M)) * (ngsolve.grad(self.Z[k][0])[0] * ngsolve.grad(self.test_func[k][0])[0] / (L**2) +
                                                               ngsolve.grad(self.Z[k][0])[1] * ngsolve.grad(self.test_func[k][0])[1] / (B**2)) * ngsolve.dx
            # transport divergence from cosine component in equation sine component
            self.bilinear_forms[k] += -H * sum(self.mu[k][m]*self.nu[k][m] * G4(m) for m in range(M)) * (ngsolve.grad(self.Z[k][1])[0] * ngsolve.grad(self.test_func[k][0])[0] / (L**2) + 
                                                                                                 ngsolve.grad(self.Z[k][1])[1] * ngsolve.grad(self.test_func[k][0])[1] / (B**2)) * ngsolve.dx
            # transport divergence from cosine component in equation cosine component
            self.bilinear_forms[k] += -H * sum(self.mu[k][m] * G4(m) for m in range(M)) * (ngsolve.grad(self.Z[k][1])[0] * ngsolve.grad(self.test_func[k][1])[0] / (L**2) + 
                                                              ngsolve.grad(self.Z[k][1])[1] * ngsolve.grad(self.test_func[k][1])[1] / (B**2)) * ngsolve.dx
            # transport divergence from sine component in equation cosine component
            self.bilinear_forms[k] += H * sum(self.mu[k][m] * self.nu[k][m] * G4(m) for m in range(M)) * (ngsolve.grad(self.Z[k][0])[0] * ngsolve.grad(self.test_func[k][1])[0] / (L**2) +
                                                                                                  ngsolve.grad(self.Z[k][0])[1] * ngsolve.grad(self.test_func[k][1])[1] / (B**2)) * ngsolve.dx
            
                                                               


    def _assemble_forms(self):
        for n in range(self.parent_hydro.numerical_information['imax'] + 1):
            self.bilinear_forms[n].Assemble()
            self.linear_forms[n].Assemble()
    

    def solve(self):
        self.solutions = {}

        sol = ngsolve.GridFunction(self.Zspace)
        if self.forcing.forcing_type == 'sea_boundary': # Dirichlet boundary condition
            sol.Set(self.forcing.forcing_func[0], ngsolve.BND)

        print(f"Solving for tidal component M0")

        res = self.linear_forms[0].vec.CreateVector()
        res.data = self.linear_forms[0].vec - self.bilinear_forms[0].mat * sol.vec
        sol.vec.data += self.bilinear_forms[0].mat.Inverse(freedofs=self.Zspace.FreeDofs()) * res
        self.solutions[0] = sol

        for n in range(1, self.parent_hydro.numerical_information['imax'] + 1):
            sol = ngsolve.GridFunction(self.Zspace_nonstationary)
            if self.forcing.forcing_type == 'sea_boundary': # Dirichlet boundary condition
                sol.components[0].Set(self.forcing.forcing_func[-n], ngsolve.BND)
                sol.components[1].Set(self.forcing.forcing_func[n], ngsolve.BND)
                
            print(f"Solving for tidal component M{2*n}")

            res = self.linear_forms[n].vec.CreateVector()
            res.data = self.linear_forms[n].vec - self.bilinear_forms[n].mat * sol.vec
            sol.vec.data += self.bilinear_forms[n].mat.Inverse(freedofs=self.Zspace_nonstationary.FreeDofs()) * res
            self.solutions[n] = sol  
            print(f"Solver finished")         
    

    def convert_to_hydrodynamics(self):
        """Construct a Hydrodynamics object from the parameters and solution."""

        # disable ramping zone for this Hydrodynamics object
        new_geometric_information = deepcopy(self.parent_hydro.geometric_information)
        new_geometric_information['L_BL_sea'] = 0
        new_geometric_information['L_R_sea'] = 0
        new_geometric_information['L_R_sea'] = 0
        new_geometric_information['L_BL_river'] = 0
        new_geometric_information['L_R_river'] = 0
        new_geometric_information['L_R_river'] = 0

        spatial_parameters_sympy = {}
        for name, param in self.parent_hydro.spatial_parameters.items():
            spatial_parameters_sympy[name] = param.fh

        new_hydro = Hydrodynamics(self.parent_hydro.model_options,new_geometric_information, self.parent_hydro.numerical_information, 
                                  self.parent_hydro.constant_physical_parameters, spatial_parameters_sympy, solve_suitable=False)
        
        L = new_hydro.geometric_information['x_scaling']
        B = new_hydro.geometric_information['y_scaling']
        # H = new_hydro.spatial_parameters['H'].cf
        
        # subtidal component
        new_hydro.gamma_solution[0] = self.solutions[0]

        # forced_part_u_coef = new_hydro.vertical_basis.project_Galerkin_orthogonal(new_hydro.numerical_information['M'], lambda sig: (self._dn(0)(sig) * ngsolve.grad(self.solutions[0])[0]).real / L)
        # forced_part_v_coef = new_hydro.vertical_basis.project_Galerkin_orthogonal(new_hydro.numerical_information['M'], lambda sig: (self._dn(0)(sig) * ngsolve.grad(self.solutions[0])[1]).real / B)

        for m in range(new_hydro.numerical_information['M']):
            forcing_part_u = self.forcing.forcing_func[0][0][m] if self.forcing.forcing_type == 'body' else ngsolve.CF(0)
            forcing_part_v = self.forcing.forcing_func[1][0][m] if self.forcing.forcing_type == 'body' else ngsolve.CF(0)

            forced_part_u = self.mu[0][m] * ngsolve.grad(self.solutions[0])[0] / L
            forced_part_v = self.mu[0][m] * ngsolve.grad(self.solutions[0])[1] / B

            new_hydro.alpha_solution[m][0] = forced_part_u + forcing_part_u
            new_hydro.beta_solution[m][0] = forced_part_v + forcing_part_v
            

        # tidal components    

        for k in range(1, new_hydro.numerical_information['imax'] + 1):
            new_hydro.gamma_solution[k] = self.solutions[k].components[1]
            new_hydro.gamma_solution[-k] = self.solutions[k].components[0]
            # forced_part_u_coef = new_hydro.vertical_basis.project_Galerkin_orthogonal(new_hydro.numerical_information['M'], lambda sig: self._dn(l)(sig) * ngsolve.grad(self.solutions[l])[0] / L)
            # forced_part_v_coef = new_hydro.vertical_basis.project_Galerkin_orthogonal(new_hydro.numerical_information['M'], lambda sig: self._dn(l)(sig) * ngsolve.grad(self.solutions[l])[1] / B)

            for m in range(new_hydro.numerical_information['M']):
                forcing_part_u_cosine = self.forcing.forcing_func[0][k][m] if self.forcing.forcing_type == 'body' else ngsolve.CF(0)
                forcing_part_u_sine = self.forcing.forcing_func[0][-k][m] if self.forcing.forcing_type == 'body' else ngsolve.CF(0)
                forcing_part_v_cosine = self.forcing.forcing_func[1][k][m] if self.forcing.forcing_type == 'body' else ngsolve.CF(0)
                forcing_part_v_sine = self.forcing.forcing_func[1][-k][m] if self.forcing.forcing_type == 'body' else ngsolve.CF(0)

                new_hydro.alpha_solution[m][k] = self.mu[k][m] * ngsolve.grad(self.solutions[k].components[1])[0] / L - \
                                                 self.mu[k][m] * self.nu[k][m] * ngsolve.grad(self.solutions[k].components[0])[0] / L - \
                                                 forcing_part_u_cosine
                new_hydro.alpha_solution[m][-k] = self.mu[k][m] * ngsolve.grad(self.solutions[k].components[0])[0] / L + \
                                                  self.mu[k][m] * self.nu[k][m] * ngsolve.grad(self.solutions[k].components[1])[0] / L - \
                                                  forcing_part_u_sine

                new_hydro.beta_solution[m][k] = self.mu[k][m] * ngsolve.grad(self.solutions[k].components[1])[1] / B - \
                                                self.mu[k][m] * self.nu[k][m] * ngsolve.grad(self.solutions[k].components[0])[1] / B - \
                                                forcing_part_v_cosine
                new_hydro.beta_solution[m][-k] = self.mu[k][m] * ngsolve.grad(self.solutions[k].components[0])[1] / B + \
                                                 self.mu[k][m] * self.nu[k][m] * ngsolve.grad(self.solutions[k].components[1])[1] / B - \
                                                 forcing_part_u_sine

        return new_hydro
    

class Decomposed_Hydrodynamics(object):

    """
    
    What this class does:
    - Stores data/solution for multiple individual forcings/tidal components.
    - Has a function that takes a forcing, and sets up the system of equations to solve.
    
    """

    def __init__(self, parent_hydro: Hydrodynamics, order: int, forcing_list: list):
        """
        
        Arguments:

        - parent_hydro (Hydrodynamics): hydrodynamics-object of the simulation to decompose into processes
        - order (int):                  order of FE basis that is used to solve the decomposed equations
        - forcing_list (list):          list of forcing names that we should decompose into.
        
        """

        # save parent hydrodynamics object so we can reuse all of its parameters/geometry
        self.parent = parent_hydro
        self.order = order
        M = self.parent.numerical_information['M']
        imax = self.parent.numerical_information['imax']

        self.mu, self.nu = construct_mu_nu(parent_hydro)

        self.forcings = {}
        self.solutions = {}

        L = self.parent.geometric_information['x_scaling']
        B = self.parent.geometric_information['y_scaling']

        # self.complex_z = [self.parent.gamma_solution[0]]
        self.z = {0: self.parent.gamma_solution[0]}

        # self.complex_u = [[self.parent.alpha_solution[m][0] for m in range(self.parent.numerical_information['M'])]]
        # self.complex_v = [[self.parent.beta_solution[m][0] for m in range(self.parent.numerical_information['M'])]]
        self.u = {0: [self.parent.alpha_solution[m][0] for m in range(M)]}
        self.v = {0: [self.parent.beta_solution[m][0] for m in range(M)]}

        # self.complex_z_x = [ngsolve.grad(self.parent.gamma_solution[0])[0]]
        # self.complex_u_x = [[ngsolve.grad(self.parent.alpha_solution[m][0])[0] for m in range(self.parent.numerical_information['M'])]]
        # self.complex_v_x = [[ngsolve.grad(self.parent.beta_solution[m][0])[0] for m in range(self.parent.numerical_information['M'])]]

        self.zx = {0: ngsolve.grad(self.parent.gamma_solution[0])[0] / L}
        self.ux = {0: [ngsolve.grad(self.parent.alpha_solution[m][0])[0] / L for m in range(M)]}
        self.vx = {0: [ngsolve.grad(self.parent.beta_solution[m][0])[0] / L for m in range(M)]}

        self.zy = {0: ngsolve.grad(self.parent.gamma_solution[0])[1] / B}
        self.uy = {0: [ngsolve.grad(self.parent.alpha_solution[m][0])[1] / B for m in range(M)]}
        self.vy = {0: [ngsolve.grad(self.parent.beta_solution[m][0])[1] / B for m in range(M)]}

        self.uxx = {0: [take_second_derivative(self.parent.alpha_solution[m][0], self.parent.U, 0) / (L**2) for m in range(M)]}
        self.uyy = {0: [take_second_derivative(self.parent.alpha_solution[m][0], self.parent.U, 1) / (B**2) for m in range(M)]}
        self.vxx = {0: [take_second_derivative(self.parent.beta_solution[m][0], self.parent.V, 0) / (L**2) for m in range(M)]}
        self.vyy = {0: [take_second_derivative(self.parent.beta_solution[m][0], self.parent.V, 1) / (B**2) for m in range(M)]}

        # self.complex_z_y = [ngsolve.grad(self.parent.gamma_solution[0])[1]]
        # self.complex_u_y = [[ngsolve.grad(self.parent.alpha_solution[m][0])[1] for m in range(self.parent.numerical_information['M'])]]
        # self.complex_v_y = [[ngsolve.grad(self.parent.beta_solution[m][0])[1] for m in range(self.parent.numerical_information['M'])]]

        # self.complex_u_xx = [[take_second_derivative(self.parent.alpha_solution[m][0], self.parent.U, 0) for m in range(self.parent.numerical_information['M'])]]
        # self.complex_u_yy = [[take_second_derivative(self.parent.alpha_solution[m][0], self.parent.U, 1) for m in range(self.parent.numerical_information['M'])]]
        # self.complex_v_xx = [[take_second_derivative(self.parent.beta_solution[m][0], self.parent.V, 0) for m in range(self.parent.numerical_information['M'])]]
        # self.complex_v_yy = [[take_second_derivative(self.parent.beta_solution[m][0], self.parent.V, 1) for m in range(self.parent.numerical_information['M'])]]

        for k in range(1, imax + 1):
            # self.complex_z.append(self.parent.gamma_solution[l] - 1j*self.parent.gamma_solution[-l])
            # self.complex_u.append([self.parent.alpha_solution[m][l]-1j*self.parent.alpha_solution[m][-l] for m in range(self.parent.numerical_information['M'])])
            # self.complex_v.append([self.parent.beta_solution[m][l]-1j*self.parent.beta_solution[m][-l] for m in range(self.parent.numerical_information['M'])])

            # self.complex_z_x.append(ngsolve.grad(self.parent.gamma_solution[l])[0] - 1j * ngsolve.grad(self.parent.gamma_solution[-l])[0])
            # self.complex_u_x.append([ngsolve.grad(self.parent.alpha_solution[m][l])[0] - 1j * ngsolve.grad(self.parent.alpha_solution[m][-l])[0] for m in range(self.parent.numerical_information['M'])])
            # self.complex_v_x.append([ngsolve.grad(self.parent.beta_solution[m][l])[0] - 1j * ngsolve.grad(self.parent.beta_solution[m][-l])[0] for m in range(self.parent.numerical_information['M'])])

            # self.complex_z_y.append(ngsolve.grad(self.parent.gamma_solution[l])[1] - 1j * ngsolve.grad(self.parent.gamma_solution[-l])[1])
            # self.complex_u_y.append([ngsolve.grad(self.parent.alpha_solution[m][l])[1] - 1j * ngsolve.grad(self.parent.alpha_solution[m][-l])[1] for m in range(self.parent.numerical_information['M'])])
            # self.complex_v_y.append([ngsolve.grad(self.parent.beta_solution[m][l])[1] - 1j * ngsolve.grad(self.parent.beta_solution[m][-l])[1] for m in range(self.parent.numerical_information['M'])])

            # self.complex_u_xx.append([take_second_derivative(self.parent.alpha_solution[m][l], self.parent.U, 0) - 1j * take_second_derivative(self.parent.alpha_solution[m][-l], self.parent.U, 0) for m in range(self.parent.numerical_information['M'])])
            # self.complex_u_yy.append([take_second_derivative(self.parent.alpha_solution[m][l], self.parent.U, 1) - 1j * take_second_derivative(self.parent.alpha_solution[m][-l], self.parent.U, 1) for m in range(self.parent.numerical_information['M'])])
            # self.complex_v_xx.append([take_second_derivative(self.parent.beta_solution[m][l], self.parent.V, 0) - 1j * take_second_derivative(self.parent.beta_solution[m][-l], self.parent.V, 0) for m in range(self.parent.numerical_information['M'])])
            # self.complex_v_yy.append([take_second_derivative(self.parent.beta_solution[m][l], self.parent.V, 1) - 1j * take_second_derivative(self.parent.beta_solution[m][-l], self.parent.V, 1) for m in range(self.parent.numerical_information['M'])])

            self.z[k] = self.parent.gamma_solution[k]
            self.u[k] = [self.parent.alpha_solution[m][k] for m in range(M)]
            self.v[k] = [self.parent.beta_solution[m][k] for m in range(M)]
            self.z[-k] = self.parent.gamma_solution[-k]
            self.u[-k] = [self.parent.alpha_solution[m][-k] for m in range(M)]
            self.v[-k] = [self.parent.beta_solution[m][-k] for m in range(M)]

            self.zx[k] = ngsolve.grad(self.parent.gamma_solution[k])[0] / L
            self.ux[k] = [ngsolve.grad(self.parent.alpha_solution[m][k])[0] / L for m in range(M)]
            self.vx[k] = [ngsolve.grad(self.parent.beta_solution[m][k])[0] / L for m in range(M)]
            self.zx[-k] = ngsolve.grad(self.parent.gamma_solution[-k])[0] / L
            self.ux[-k] = [ngsolve.grad(self.parent.alpha_solution[m][-k])[0] / L for m in range(M)]
            self.vx[-k] = [ngsolve.grad(self.parent.beta_solution[m][-k])[0] / L for m in range(M)]
            
            self.zy[k] = ngsolve.grad(self.parent.gamma_solution[k])[1] / B
            self.uy[k] = [ngsolve.grad(self.parent.alpha_solution[m][k])[1] / B for m in range(M)]
            self.vy[k] = [ngsolve.grad(self.parent.beta_solution[m][k])[1] / B for m in range(M)]
            self.zy[-k] = ngsolve.grad(self.parent.gamma_solution[-k])[1] / B
            self.uy[-k] = [ngsolve.grad(self.parent.alpha_solution[m][-k])[1] / B for m in range(M)]
            self.vy[-k] = [ngsolve.grad(self.parent.beta_solution[m][-k])[1] / B for m in range(M)]

            self.uxx[k] = [take_second_derivative(self.parent.alpha_solution[m][k], self.parent.U, 0) / (L**2) for m in range(M)]
            self.vxx[k] = [take_second_derivative(self.parent.beta_solution[m][k], self.parent.V, 0) / (L**2) for m in range(M)]
            self.uyy[k] = [take_second_derivative(self.parent.alpha_solution[m][k], self.parent.U, 1) / (B**2) for m in range(M)]
            self.vyy[k] = [take_second_derivative(self.parent.beta_solution[m][k], self.parent.V, 1) / (B**2) for m in range(M)]
            self.uxx[-k] = [take_second_derivative(self.parent.alpha_solution[m][-k], self.parent.U, 0) / (L**2) for m in range(M)]
            self.vxx[-k] = [take_second_derivative(self.parent.beta_solution[m][-k], self.parent.V, 0) / (L**2) for m in range(M)]
            self.uyy[-k] = [take_second_derivative(self.parent.alpha_solution[m][-k], self.parent.U, 1) / (B**2) for m in range(M)]
            self.vyy[-k] = [take_second_derivative(self.parent.beta_solution[m][-k], self.parent.V, 1) / (B**2) for m in range(M)]

        self._construct_forcings(forcing_list)


    # private methods

    def _construct_vertically_variable_body_forcing(self, name, term_construction):
        """Constructs body forcing based on a vertically variable term in the equation. The parameter term_construction is a lambda expression taking
        variables u and v and their gradients (no mixed derivatives) for sine and cosine components and also m and k, and outputs a list with the 
        first entry being that term in the u-momentum equation sine,
        the second entry being that term in the u-momentum equation cosine, the third entry being that term in
        the v-momentum equation sine, and the fourth entry being that term in the v-momentum equation cosine. 
        For example, for the Coriolis function, the argument term_construction would be
        
        term_construction = lambda u_sine, u_cosine, v_sine, v_cosine, ux_sine, ux_cosine, vx_sine, vx_cosine,
                                   uy_sine, uy_cosine, vy_sine, vy_cosine, uxx_sine, uxx_cosine, vxx_sine, vxx_cosine,
                                   uyy_sine, uyy_cosine, vyy_sine, vyy_cosine, m, k : [-f*v_sine, -f*v_cosine, f*u_sine, f*u_cosine].
        
        """

        M = self.parent.numerical_information['M']
        imax = self.parent.numerical_information['imax']

        H = self.parent.spatial_parameters['H'].cf
        Av = self.parent.constant_physical_parameters['Av']
        sigma = self.parent.constant_physical_parameters['sigma']

        # forcing_func = {'u': [], 'v': []}
        integrated_forcing_func = {'u': {}, 'v': {}}
        forcing_u = {k: [] for k in range(-imax, imax + 1)}
        forcing_v = {k: [] for k in range(-imax, imax + 1)}
        # subtidal component
        for m in range(self.parent.numerical_information['M']):
            # term = term_construction(self.complex_u[0][m], self.complex_v[0][m], self.complex_u_x[0][m], self.complex_u_y[0][m], self.complex_v_x[0][m], self.complex_v_y[0][m], self.complex_u_xx[0][m], self.complex_u_yy[0][m], self.complex_v_xx[0][m], self.complex_v_yy[0][m])
            term = term_construction(self.u[0][m], self.u[0][m], self.v[0][m], self.v[0][m], self.ux[0][m], self.ux[0][m],
                                     self.vx[0][m], self.vx[0][m], self.uy[0][m], self.uy[0][m], self.vy[0][m], self.vy[0][m],
                                     self.uxx[0][m], self.uxx[0][m], self.vxx[0][m], self.vxx[0][m], self.uyy[0][m], self.uyy[0][m],
                                     self.vyy[0][m], self.vyy[0][m], m, 0)
            
            u_forcing_m = -term[0]
            v_forcing_m = -term[2]

            forcing_u[0].append(u_forcing_m)
            forcing_v[0].append(v_forcing_m)

        # forcing_func['u'].append(lambda sig: sum(forcing_u[0][m] * self.parent.vertical_basis.evaluation_function(sig, m) for m in range(self.parent.numerical_information['M'])))
        # forcing_func['v'].append(lambda sig: sum(forcing_v[0][m] * self.parent.vertical_basis.evaluation_function(sig, m) for m in range(self.parent.numerical_information['M'])))
        integrated_forcing_func['u'][0] = sum(forcing_u[0][m] * self.parent.vertical_basis.integrated_evaluation_function(0, m) for m in range(M))
        integrated_forcing_func['v'][0] = sum(forcing_v[0][m] * self.parent.vertical_basis.integrated_evaluation_function(0, m) for m in range(M))

        # tidal components
        for k in range(1, imax + 1):
            for m in range(M):
                term = term_construction(self.u[-k][m], self.u[k][m], self.v[-k][m], self.v[k][m], self.ux[-k][m], self.ux[k][m],
                                         self.vx[-k][m], self.vx[k][m], self.uy[-k][m], self.uy[k][m], self.vy[-k][m], self.vy[k][m],
                                         self.uxx[-k][m], self.uxx[k][m], self.vxx[-k][m], self.vxx[k][m], self.uyy[-k][m], self.uyy[k][m],
                                         self.vyy[-k][m], self.vyy[k][m], m, k)

                u_forcing_m_sine = -term[0] - self.nu[k][m] * term[1]
                u_forcing_m_cosine = -term[1] + self.nu[k][m] * term[0]
                v_forcing_m_sine = -term[2] - self.nu[k][m] * term[3]
                v_forcing_m_cosine = -term[3] + self.nu[k][m] * term[2]

                forcing_u[-k].append(u_forcing_m_sine)
                forcing_u[k].append(u_forcing_m_cosine)
                forcing_v[-k].append(v_forcing_m_sine)
                forcing_v[k].append(v_forcing_m_cosine)

            # forcing_func['u'].append(lambda sig: sum(forcing_u[l][m] * self.parent.vertical_basis.evaluation_function(sig, m) for m in range(self.parent.numerical_information['M'])))
            # forcing_func['v'].append(lambda sig: sum(forcing_v[l][m] * self.parent.vertical_basis.evaluation_function(sig, m) for m in range(self.parent.numerical_information['M'])))
            integrated_forcing_func['u'][-k] = sum(forcing_u[-k][m] * self.parent.vertical_basis.integrated_evaluation_function(0, m) for m in range(M))
            integrated_forcing_func['u'][k] = sum(forcing_u[k][m] * self.parent.vertical_basis.integrated_evaluation_function(0, m) for m in range(M))
            integrated_forcing_func['v'][-k] = sum(forcing_v[-k][m] * self.parent.vertical_basis.integrated_evaluation_function(0, m) for m in range(M))
            integrated_forcing_func['v'][k] = sum(forcing_v[k][m] * self.parent.vertical_basis.integrated_evaluation_function(0, m) for m in range(M))
            # integrated_forcing_func['v'].append(sum(forcing_v[l][m] * self.parent.vertical_basis.integrated_evaluation_function(0, m) for m in range(self.parent.numerical_information['M'])))

        new_forcing = Forcing(name, 'body', [forcing_u, forcing_v], [integrated_forcing_func['u'], integrated_forcing_func['v']])
        self.forcings[name] = new_forcing        



    def _construct_vertically_constant_body_forcing(self, name, const):
        """Constructs body forcing that does not vary in depth. Const is a dict of the following structure:
        
        const = {'u': {0: M0, -1: M2_sine, 1: M2_cosine, -2: M4_sine, 2: M4_cosine, ...}, 'v': {...}}
        
        """
        # sigma = self.parent.constant_physical_parameters['sigma']
        # Av = self.parent.constant_physical_parameters['Av']
        # H = self.parent.spatial_parameters['H']
        # sf = self.parent.constant_physical_parameters['sf']
        
        # mu = [ngsolve.sqrt(H / Av * 2 * np.pi * 1j * l * sigma) for l in range(1, self.parent.numerical_information['imax']) + 1]
        # C_u = [-1j / (2*np.pi * l * sigma) * const['u'][l] / (2j*Av*mu[l]*ngsolve.sin(mu[l]) + 2*sf*ngsolve.cos(mu[l])) for l in range(1, self.parent.numerical_information['imax'] + 1)]
        # C_v = [-1j / (2*np.pi * l * sigma) * const['v'][l] / (2j*Av*mu[l]*ngsolve.sin(mu[l]) + 2*sf*ngsolve.cos(mu[l])) for l in range(1, self.parent.numerical_information['imax'] + 1)]

        # # project forcing part U onto the vertical basis for later; will be necessary to convert this object into a Hydrodynamics-object to which we can apply our post-processing tools

        # forcing_u = []
        # forcing_v = []

        # forcing_func = {'u': [lambda sig: H * const['u'] * (1/(2*Av) * sig**2 + 1/sf - 1/(2*Av))], 'v': [lambda sig: H * const['v'] * (1/(2*Av) * sig**2 + 1/sf - 1/(2*Av))]}
        # integrated_forcing_func = {'u': [H * const['u'] * (1/sf - 1/(3*Av))], 'v': [H * const['u'] * (1/sf - 1/(3*Av))]}

        # forcing_u.append(self.parent.vertical_basis.project_Galerkin_orthogonal(self.parent.numerical_information['M'], forcing_func['u'][-1]))
        # forcing_v.append(self.parent.vertical_basis.project_Galerkin_orthogonal(self.parent.numerical_information['M'], forcing_func['v'][-1]))

        # for l in range(1, self.parent.numerical_information['imax'] + 1):
        #     forcing_func['u'].append(lambda sig: C_u[l-1]*ngsolve.cosh(mu[l-1] * sig) + 1j/(2*np.pi*l*sigma)*const['u'][l])
        #     forcing_func['v'].append(lambda sig: C_v[l-1]*ngsolve.cosh(mu[l-1] * sig) + 1j/(2*np.pi*l*sigma)*const['v'][l])
        #     integrated_forcing_func['u'].append(C_u[l-1]*ngsolve.sinh(mu[l-1])/mu[l-1] + 1j/(2*np.pi*l*sigma)*const['u'][l])
        #     integrated_forcing_func['u'].append(C_v[l-1]*ngsolve.sinh(mu[l-1])/mu[l-1] + 1j/(2*np.pi*l*sigma)*const['v'][l])

        #     forcing_u.append(self.parent.vertical_basis.project_Galerkin_orthogonal(self.parent.numerical_information['M'], forcing_func['u'][-1]))
        #     forcing_v.append(self.parent.vertical_basis.project_Galerkin_orthogonal(self.parent.numerical_information['M'], forcing_func['v'][-1]))        

        M = self.parent.numerical_information['M']
        imax = self.parent.numerical_information['imax']
        G1 = self.parent.vertical_basis.inner_product
        G4 = self.parent.vertical_basis.tensor_dict['G4']


        forcing_u = {k: [] for k in range(-imax, imax + 1)}
        forcing_v = {k: [] for k in range(-imax, imax + 1)}
        integrated_forcing_func = {'u': {}, 'v': {}}

        for k in range(imax + 1):
            for m in range(M):
                forcing_u[k].append(-const['u'][k] * G4(m) / G1(m, m) + self.nu[k][m] * const['u'][-k] * G4(m) / G1(m, m))
                forcing_v[k].append(-const['v'][k] * G4(m) / G1(m, m) + self.nu[k][m] * const['v'][-k] * G4(m) / G1(m, m))
                forcing_u[-k].append(-const['u'][-k] * G4(m) / G1(m, m) - self.nu[k][m] * const['u'][k] * G4(m) / G1(m, m))
                forcing_v[-k].append(-const['v'][-k] * G4(m) / G1(m, m) - self.nu[k][m] * const['v'][k] * G4(m) / G1(m, m))

            integrated_forcing_func['u'][k] = sum(forcing_u[k][m] * self.parent.vertical_basis.integrated_evaluation_function(0, m) for m in range(M))
            integrated_forcing_func['v'][k] = sum(forcing_u[k][m] * self.parent.vertical_basis.integrated_evaluation_function(0, m) for m in range(M))

        new_forcing = Forcing(name, 'body', [forcing_u, forcing_v], [integrated_forcing_func['u'], integrated_forcing_func['v']])
        self.forcings[name] = new_forcing


    def _construct_tide_forcing(self):
        forcing_func = self.z
        tide_forcing = Forcing('tide', 'sea_boundary', forcing_func)
        self.forcings['tide'] = tide_forcing


    def _construct_river_forcing(self):
        forcing_func = [sum(self.u[k][m] * self.parent.vertical_basis.integrated_evaluation_function(0, m) for m in range(self.parent.numerical_information['M']))
                        for k in range(-self.parent.numerical_information['imax'], self.parent.numerical_information['imax'] + 1)]
        river_forcing = Forcing('river', 'river_boundary', forcing_func)
        self.forcings['river'] = river_forcing


    def _construct_coriolis_forcing(self):
        f = self.parent.constant_physical_parameters['f']
        H = self.parent.spatial_parameters['H'].cf
        term_construction = lambda u_sine, u_cosine, v_sine, v_cosine, ux_sine, ux_cosine, \
                                   uy_sine, uy_cosine, vx_sine, vx_cosine, vy_sine, vy_cosine, \
                                   uxx_sine, uxx_cosine, uyy_sine, uyy_cosine, vxx_sine, vxx_cosine, \
                                   vyy_sine, vyy_cosine, m ,k: [-f * v_sine, -f * v_cosine, f * u_sine, f * u_cosine]
        self._construct_vertically_variable_body_forcing('coriolis', term_construction)


    def _construct_horizontal_eddy_viscosity(self, include_surface_interactions=False):
        Ah = self.parent.constant_physical_parameters['Ah']
        H = self.parent.spatial_parameters['H'].cf

        term_construction = lambda u, v, ux, uy, vx, vy, uxx, uyy, vxx, vyy: [-Ah * (uxx  + uyy), -Ah * (vxx + vyy)]
        term_construction = lambda u_sine, u_cosine, v_sine, v_cosine, ux_sine, ux_cosine, \
                                   uy_sine, uy_cosine, vx_sine, vx_cosine, vy_sine, vy_cosine, \
                                   uxx_sine, uxx_cosine, uyy_sine, uyy_cosine, vxx_sine, vxx_cosine, \
                                   vyy_sine, vyy_cosine, m , k: [-Ah * H * (uxx_sine + uyy_sine), -Ah * H * (uxx_cosine + uyy_cosine),
                                                                 -Ah * H * (vxx_sine + vyy_sine), -Ah * H * (vxx_cosine + vyy_cosine)]
        
        self._construct_vertically_variable_body_forcing('horizontal_eddy_viscosity', term_construction)


    def _construct_forcings(self, forcing_list):

        for forcing_name in forcing_list:
            if forcing_name == 'tide':
                self._construct_tide_forcing()
            elif forcing_name == 'horizontal_eddy_viscosity':
                self._construct_horizontal_eddy_viscosity()
            elif forcing_name == 'Coriolis' or forcing_name == 'coriolis':
                self._construct_coriolis_forcing()
            else:
                raise ValueError(f"Forcing {forcing_name} not known.")


    def solve_decompositions(self) -> dict[Hydrodynamics]:
        
        for name, forcing in self.forcings.items():
            linear_hydro = LinearForcedHydrodynamics(forcing, self.parent, self.order)
            linear_hydro.solve()
            self.solutions[name] = linear_hydro.convert_to_hydrodynamics()

        

        



    





