import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import timeit
import ngsolve
import copy
import matplotlib.pyplot as plt
import pypardiso

from NiFlow.hydrodynamics import *
import NiFlow.define_weak_forms as weakforms
from NiFlow.utils import *


def solve(hydro: Hydrodynamics, max_iterations: int = 10, tolerance: float = 1e-12, linear_solver = 'pardiso', 
          continuation_parameters: dict = {'advection_epsilon': [1], 'Av': [1], 'Ah': [1]}, stopcriterion = 'scaled_2norm',
          plot_intermediate_results='none', parallel=True, matrix_analysis=False, print_log:bool=True):

    """
    
    Compute the solution of the model given by a Hydrodynamics-object and add it as an attribute to the Hydrodynamics-object. Solution is computed using a Newton-Raphson
    method, combined with a continuation (homology) method to guide the Newton method towards the correct solution.
    
    Arguments:
    
        - hydro (Hydrodynamics):            object containing the (weak) model equations and options;
        - max_iterations (int):             maximum number of Newton iterations per continuation step;
        - tolerance (float):                if the stopping criterion is less than this value, the Newton method terminates and the procedure moves to the next continuation step;
        - linear_solver:                    choice of linear solver; options: 'pardiso', 'pypardiso', 'scipy_direct', 'bicgstab'
        - continuation_parameters (dict):   dictionary with keys 'advection_epsilon' and 'Av' and 'Ah', with values indicating what the default value of these parameters should be multiplied by in each continuation step;
        - stopcriterion:                    choice of stopping criterion; options: 'matrix_norm', 'scaled_2norm', 'relative_newtonstepsize';
        - plot_intermediate_results:        indicates whether intermediate results should be plotted and saved; options: 'none' (default), 'all' and 'overview'.
        - parallel:                         flag indicating whether time-costly operations should be performed in parallel (see https://docu.ngsolve.org/latest/how_to/howto_parallel.html)
        - matrix_analysis:                  if True, plots the non-zero elements of the matrix, the size of all entries via a colormap, and the right-hand side vector using a colormap.
        - print_log:                        if True, prints runtimes of the steps of the solution process, as well as stopping criterion values.
    
    """

    M = hydro.numerical_information['M']
    imax = hydro.numerical_information['imax']

    # Quick argument check

    if len(continuation_parameters['advection_epsilon']) == len(continuation_parameters['Av']) and len(continuation_parameters['Ah']) == len(continuation_parameters['Av']):
        num_continuation_steps = len(continuation_parameters['advection_epsilon'])
    else:
        raise ValueError(f"Length of both continuation parameter lists must be equal; now the lenghts are {len(continuation_parameters['advection_epsilon'])} and {len(continuation_parameters['Av'])} and {len(continuation_parameters['Ah'])}")

    # Report that solution procedure is about to start.
    if print_log:
        print(f"Initiating solution procedure for hydrodynamics-model with {hydro.numerical_information['M']} vertical components and {hydro.numerical_information['imax'] + 1} tidal constituents (including residual).\nThe total number of free degrees of freedom is {hydro.nfreedofs}.")

    # If improve_seaward_initial_guess, do a linear simulation first

    if hydro.model_options['sea_boundary_treatment'] == 'linear_guess':
        if print_log:
            print("Doing a linear simulation to get a good guess for the correct computational seaward boundary condition.\n")
        linear_geometric_information = copy.deepcopy(hydro.geometric_information)
        # linear_geometric_information['L_BL_sea'] += linear_geometric_information['L_R_sea']
        # linear_geometric_information['L_R_sea'] = 0
        
        linear_model_options = copy.deepcopy(hydro.model_options)
        linear_model_options['advection_influence_matrix'] = np.full((imax+1, imax+1), False)
        linear_model_options['sea_boundary_treatment'] = 'simple'

        linear_hydro = Hydrodynamics(linear_model_options, linear_geometric_information, hydro.numerical_information, hydro.constant_physical_parameters, hydro.sympy_spatial_parameters)
        solve(linear_hydro, max_iterations, tolerance, linear_solver, continuation_parameters={'advection_epsilon': [1], 'Av': [1], 'Ah': [1]}, stopcriterion=stopcriterion, plot_intermediate_results=False, 
              parallel=parallel, matrix_analysis=False, print_log=False)

        linear_gamma_complex = np.array([evaluate_CF_point(linear_hydro.gamma_solution[l], linear_hydro.mesh, 0, 0) - 1j*evaluate_CF_point(linear_hydro.gamma_solution[-l], linear_hydro.mesh, 0, 0) for l in range(1, imax+1)]).astype(complex)
        standard_initial_guess_complex = np.array([hydro.seaward_forcing.amplitudes[l]*np.cos(hydro.seaward_forcing.phases[l-1]) - 1j*hydro.seaward_forcing.amplitudes[l]*np.sin(hydro.seaward_forcing.phases[l-1]) for l in range(1, imax+1)]).astype(complex)
        improved_initial_guess_complex = standard_initial_guess_complex / linear_gamma_complex

        improved_initial_guess_cos = improved_initial_guess_complex.real
        improved_initial_guess_sin = improved_initial_guess_complex.imag


    # Set initial guess
    sol = ngsolve.GridFunction(hydro.femspace)
        # tidal waterlevel

    if hydro.model_options['sea_boundary_treatment'] == 'simple': # if we handle the seaward boundary condition, then setting the boundary condition will be done later
        sol.components[2*(M)*(2*imax+1)].Set(hydro.seaward_forcing.cfdict[0], ngsolve.BND)

        for q in range(1, imax + 1):
            sol.components[2*(M)*(2*imax+1) + q].Set(hydro.seaward_forcing.cfdict[-q], ngsolve.BND)
            sol.components[2*(M)*(2*imax+1) + imax + q].Set(hydro.seaward_forcing.cfdict[q], ngsolve.BND)
    elif hydro.model_options['sea_boundary_treatment'] == 'exact':
        sol.components[2*(M)*(2*imax+1) + (2*imax + 1)].Set(hydro.seaward_forcing.amplitudes[0])
        for q in range(1, imax + 1):
            sol.components[2*(M)*(2*imax + 1) + (2*imax + 1) + q].Set(hydro.seaward_forcing.amplitudes[q]*np.sin(hydro.seaward_forcing.phases[q-1])) # phase_list starts at semidiurnal component instead of residual; therefore index - 1
            sol.components[2*(M)*(2*imax + 1) + (2*imax + 1) + imax + q].Set(hydro.seaward_forcing.amplitudes[q]*np.cos(hydro.seaward_forcing.phases[q-1]))   
            # sol.components[2*(M)*(2*imax + 1) + (2*imax + 1) + q].Set(improved_initial_guess_sin[q-1]) # phase_list starts at semidiurnal component instead of residual; therefore index - 1
            # sol.components[2*(M)*(2*imax + 1) + (2*imax + 1) + imax + q].Set(improved_initial_guess_cos[q-1])   
    elif hydro.model_options['sea_boundary_treatment'] == 'linear_guess':
        sol.components[2*(M)*(2*imax+1)].Set(hydro.seaward_forcing.cfdict[0], ngsolve.BND)
        for q in range(1, imax + 1):
            sol.components[2*(M)*(2*imax+1) + q].Set(improved_initial_guess_sin[q-1], ngsolve.BND)
            sol.components[2*(M)*(2*imax+1) + imax + q].Set(improved_initial_guess_cos[q-1], ngsolve.BND)

        # river discharge
    if hydro.model_options['river_boundary_treatment'] != 'exact':
        for m in range(M):
            sol.components[m * (2*imax + 1)].Set(hydro.constant_physical_parameters['discharge'] * hydro.riverine_forcing.normal_alpha[m], ngsolve.BND)
            
    hydro.solution_gf = sol

    # Save true values of advection_epsilon and Av before modifying them in the continuation (homology) method

    true_epsilon = copy.copy(hydro.constant_physical_parameters['advection_epsilon'])
    true_Av = copy.copy(hydro.constant_physical_parameters['Av'])
    true_Ah = copy.copy(hydro.constant_physical_parameters['Ah'])

    for continuation_counter in range(num_continuation_steps):
        hydro.constant_physical_parameters['advection_epsilon'] = true_epsilon * continuation_parameters['advection_epsilon'][continuation_counter]
        hydro.constant_physical_parameters['Av'] = true_Av * continuation_parameters['Av'][continuation_counter]
        hydro.constant_physical_parameters['Ah'] = true_Ah * continuation_parameters['Ah'][continuation_counter]

        if print_log:
            if num_continuation_steps > 1:
                print(f"\nCONTINUATION STEP {continuation_counter}: Epsilon = {hydro.constant_physical_parameters['advection_epsilon']}, Av = {hydro.constant_physical_parameters['Av']}, Ah = {hydro.constant_physical_parameters['Ah']}.\n")
            print("Setting up full weak form\n")

        if parallel:
            with ngsolve.TaskManager():
                hydro.setup_weak_form()
        else:
            hydro.setup_weak_form()


        # Start the Newton method

        previous_iterate = copy.copy(hydro.solution_gf)

        for newton_counter in range(max_iterations):
            if print_log:
                print(f"Newton-Raphson iteration {newton_counter}")
            hydro.restructure_solution() # restructure solution so that hydro.alpha_solution, hydro.beta_solution, and hydro.gamma_solution are specified.
            # Set-up weak form of the linearisation

            forms_start = timeit.default_timer()
            if parallel:
                with ngsolve.TaskManager():
                    a = ngsolve.BilinearForm(hydro.femspace)
                    if hydro.model_options['sea_boundary_treatment'] == 'exact':
                        if hydro.model_options['river_boundary_treatment'] == 'exact':
                            weakforms.add_weak_form(a, hydro.model_options, hydro.numerical_information, hydro.geometric_information, hydro.constant_physical_parameters, hydro.spatial_parameters,
                                                    hydro.alpha_trialfunctions, hydro.beta_trialfunctions, hydro.gamma_trialfunctions,
                                                    hydro.umom_testfunctions, hydro.vmom_testfunctions, hydro.DIC_testfunctions,
                                                    hydro.vertical_basis, hydro.time_basis, hydro.riverine_forcing.normal_alpha, hydro.riverine_forcing.normal_alpha_y, only_linear=True,
                                                    A_trialfunctions=hydro.A_trialfunctions, sea_boundary_testfunctions=hydro.sea_boundary_testfunctions,
                                                    Q_trialfunctions=hydro.Q_trialfunctions, river_boundary_testfunctions=hydro.river_boundary_testfunctions)
                        else:
                            weakforms.add_weak_form(a, hydro.model_options, hydro.numerical_information, hydro.geometric_information, hydro.constant_physical_parameters, hydro.spatial_parameters,
                                                    hydro.alpha_trialfunctions, hydro.beta_trialfunctions, hydro.gamma_trialfunctions,
                                                    hydro.umom_testfunctions, hydro.vmom_testfunctions, hydro.DIC_testfunctions,
                                                    hydro.vertical_basis, hydro.time_basis, hydro.riverine_forcing.normal_alpha, hydro.riverine_forcing.normal_alpha_y, only_linear=True,
                                                    A_trialfunctions=hydro.A_trialfunctions, sea_boundary_testfunctions=hydro.sea_boundary_testfunctions)
                    elif hydro.model_options['river_boundary_treatment'] == 'exact':
                        weakforms.add_weak_form(a, hydro.model_options, hydro.numerical_information, hydro.geometric_information, hydro.constant_physical_parameters, hydro.spatial_parameters,
                                                hydro.alpha_trialfunctions, hydro.beta_trialfunctions, hydro.gamma_trialfunctions,
                                                hydro.umom_testfunctions, hydro.vmom_testfunctions, hydro.DIC_testfunctions,
                                                hydro.vertical_basis, hydro.time_basis, hydro.riverine_forcing.normal_alpha, hydro.riverine_forcing.normal_alpha_y, only_linear=True,
                                                Q_trialfunctions=hydro.Q_trialfunctions, river_boundary_testfunctions=hydro.river_boundary_testfunctions)
                    else:
                        weakforms.add_weak_form(a, hydro.model_options, hydro.numerical_information, hydro.geometric_information, hydro.constant_physical_parameters, hydro.spatial_parameters,
                                                hydro.alpha_trialfunctions, hydro.beta_trialfunctions, hydro.gamma_trialfunctions,
                                                hydro.umom_testfunctions, hydro.vmom_testfunctions, hydro.DIC_testfunctions,
                                                hydro.vertical_basis, hydro.time_basis, hydro.riverine_forcing.normal_alpha, only_linear=True)
                        
                    if hydro.constant_physical_parameters['advection_epsilon'] != 0:
                        if hydro.model_options['sea_boundary_treatment'] == 'exact':
                            if hydro.model_options['river_boundary_treatment'] == 'exact':
                                weakforms.add_linearised_nonlinear_terms(a, hydro.model_options, hydro.numerical_information, hydro.geometric_information, hydro.constant_physical_parameters, hydro.spatial_parameters,
                                                                         hydro.alpha_trialfunctions, hydro.alpha_solution, hydro.beta_trialfunctions, hydro.beta_solution,
                                                                         hydro.gamma_trialfunctions, hydro.gamma_solution,
                                                                         hydro.umom_testfunctions, hydro.vmom_testfunctions, hydro.DIC_testfunctions,
                                                                         hydro.vertical_basis, hydro.time_basis, hydro.riverine_forcing.normal_alpha, hydro.riverine_forcing.normal_alpha_y,
                                                                         A_trialfunctions=hydro.A_trialfunctions, A0=hydro.A_solution, sea_boundary_testfunctions=hydro.sea_boundary_testfunctions,
                                                                         Q_trialfunctions=hydro.Q_trialfunctions, Q0=hydro.Q_solution)
                            else:
                                weakforms.add_linearised_nonlinear_terms(a, hydro.model_options, hydro.numerical_information, hydro.geometric_information, hydro.constant_physical_parameters, hydro.spatial_parameters,
                                                                         hydro.alpha_trialfunctions, hydro.alpha_solution, hydro.beta_trialfunctions, hydro.beta_solution,
                                                                         hydro.gamma_trialfunctions, hydro.gamma_solution,
                                                                         hydro.umom_testfunctions, hydro.vmom_testfunctions, hydro.DIC_testfunctions,
                                                                         hydro.vertical_basis, hydro.time_basis, hydro.riverine_forcing.normal_alpha, hydro.riverine_forcing.normal_alpha_y,
                                                                         A_trialfunctions=hydro.A_trialfunctions, A0=hydro.A_solution, sea_boundary_testfunctions=hydro.sea_boundary_testfunctions)
                        elif hydro.model_options['river_boundary_treatment'] == 'exact':
                            weakforms.add_linearised_nonlinear_terms(a, hydro.model_options, hydro.numerical_information, hydro.geometric_information, hydro.constant_physical_parameters, hydro.spatial_parameters,
                                                                     hydro.alpha_trialfunctions, hydro.alpha_solution, hydro.beta_trialfunctions, hydro.beta_solution,
                                                                     hydro.gamma_trialfunctions, hydro.gamma_solution,
                                                                     hydro.umom_testfunctions, hydro.vmom_testfunctions, hydro.DIC_testfunctions,
                                                                     hydro.vertical_basis, hydro.time_basis, hydro.riverine_forcing.normal_alpha, hydro.riverine_forcing.normal_alpha_y,
                                                                     Q_trialfunctions=hydro.Q_trialfunctions, Q0=hydro.Q_solution)
                        else:
                            weakforms.add_linearised_nonlinear_terms(a, hydro.model_options, hydro.numerical_information, hydro.geometric_information, hydro.constant_physical_parameters, hydro.spatial_parameters,
                                                                     hydro.alpha_trialfunctions, hydro.alpha_solution, hydro.beta_trialfunctions, hydro.beta_solution,
                                                                     hydro.gamma_trialfunctions, hydro.gamma_solution,
                                                                     hydro.umom_testfunctions, hydro.vmom_testfunctions, hydro.DIC_testfunctions,
                                                                     hydro.vertical_basis, hydro.time_basis, hydro.riverine_forcing.normal_alpha)

            else:
                a = ngsolve.BilinearForm(hydro.femspace)
                if hydro.model_options['sea_boundary_treatment'] == 'exact':
                    if hydro.model_options['river_boundary_treatment'] == 'exact':
                        weakforms.add_weak_form(a, hydro.model_options, hydro.numerical_information, hydro.geometric_information, hydro.constant_physical_parameters, hydro.spatial_parameters,
                                                hydro.alpha_trialfunctions, hydro.beta_trialfunctions, hydro.gamma_trialfunctions,
                                                hydro.umom_testfunctions, hydro.vmom_testfunctions, hydro.DIC_testfunctions,
                                                hydro.vertical_basis, hydro.time_basis, hydro.riverine_forcing.normal_alpha, hydro.riverine_forcing.normal_alpha_y, only_linear=True,
                                                A_trialfunctions=hydro.A_trialfunctions, sea_boundary_testfunctions=hydro.sea_boundary_testfunctions,
                                                Q_trialfunctions=hydro.Q_trialfunctions, river_boundary_testfunctions=hydro.river_boundary_testfunctions)
                    else:
                        weakforms.add_weak_form(a, hydro.model_options, hydro.numerical_information, hydro.geometric_information, hydro.constant_physical_parameters, hydro.spatial_parameters,
                                                hydro.alpha_trialfunctions, hydro.beta_trialfunctions, hydro.gamma_trialfunctions,
                                                hydro.umom_testfunctions, hydro.vmom_testfunctions, hydro.DIC_testfunctions,
                                                hydro.vertical_basis, hydro.time_basis, hydro.riverine_forcing.normal_alpha, hydro.riverine_forcing.normal_alpha_y, only_linear=True,
                                                A_trialfunctions=hydro.A_trialfunctions, sea_boundary_testfunctions=hydro.sea_boundary_testfunctions)
                elif hydro.model_options['river_boundary_treatment'] == 'exact':
                    weakforms.add_weak_form(a, hydro.model_options, hydro.numerical_information, hydro.geometric_information, hydro.constant_physical_parameters, hydro.spatial_parameters,
                                            hydro.alpha_trialfunctions, hydro.beta_trialfunctions, hydro.gamma_trialfunctions,
                                            hydro.umom_testfunctions, hydro.vmom_testfunctions, hydro.DIC_testfunctions,
                                            hydro.vertical_basis, hydro.time_basis, hydro.riverine_forcing.normal_alpha, hydro.riverine_forcing.normal_alpha_y, only_linear=True,
                                            Q_trialfunctions=hydro.Q_trialfunctions, river_boundary_testfunctions=hydro.river_boundary_testfunctions)
                else:
                    weakforms.add_weak_form(a, hydro.model_options, hydro.numerical_information, hydro.geometric_information, hydro.constant_physical_parameters, hydro.spatial_parameters,
                                            hydro.alpha_trialfunctions, hydro.beta_trialfunctions, hydro.gamma_trialfunctions,
                                            hydro.umom_testfunctions, hydro.vmom_testfunctions, hydro.DIC_testfunctions,
                                            hydro.vertical_basis, hydro.time_basis, hydro.riverine_forcing.normal_alpha, only_linear=True)
                    
                if hydro.constant_physical_parameters['advection_epsilon'] != 0:
                    if hydro.model_options['sea_boundary_treatment'] == 'exact':
                        if hydro.model_options['river_boundary_treatment'] == 'exact':
                            weakforms.add_linearised_nonlinear_terms(a, hydro.model_options, hydro.numerical_information, hydro.geometric_information, hydro.constant_physical_parameters, hydro.spatial_parameters,
                                                                        hydro.alpha_trialfunctions, hydro.alpha_solution, hydro.beta_trialfunctions, hydro.beta_solution,
                                                                        hydro.gamma_trialfunctions, hydro.gamma_solution,
                                                                        hydro.umom_testfunctions, hydro.vmom_testfunctions, hydro.DIC_testfunctions,
                                                                        hydro.vertical_basis, hydro.time_basis, hydro.riverine_forcing.normal_alpha, hydro.riverine_forcing.normal_alpha_y,
                                                                        A_trialfunctions=hydro.A_trialfunctions, A0=hydro.A_solution, sea_boundary_testfunctions=hydro.sea_boundary_testfunctions,
                                                                        Q_trialfunctions=hydro.Q_trialfunctions, Q0=hydro.Q_solution)
                        else:
                            weakforms.add_linearised_nonlinear_terms(a, hydro.model_options, hydro.numerical_information, hydro.geometric_information, hydro.constant_physical_parameters, hydro.spatial_parameters,
                                                                        hydro.alpha_trialfunctions, hydro.alpha_solution, hydro.beta_trialfunctions, hydro.beta_solution,
                                                                        hydro.gamma_trialfunctions, hydro.gamma_solution,
                                                                        hydro.umom_testfunctions, hydro.vmom_testfunctions, hydro.DIC_testfunctions,
                                                                        hydro.vertical_basis, hydro.time_basis, hydro.riverine_forcing.normal_alpha, hydro.riverine_forcing.normal_alpha_y,
                                                                        A_trialfunctions=hydro.A_trialfunctions, A0=hydro.A_solution, sea_boundary_testfunctions=hydro.sea_boundary_testfunctions)
                    elif hydro.model_options['river_boundary_treatment'] == 'exact':
                        weakforms.add_linearised_nonlinear_terms(a, hydro.model_options, hydro.numerical_information, hydro.geometric_information, hydro.constant_physical_parameters, hydro.spatial_parameters,
                                                                    hydro.alpha_trialfunctions, hydro.alpha_solution, hydro.beta_trialfunctions, hydro.beta_solution,
                                                                    hydro.gamma_trialfunctions, hydro.gamma_solution,
                                                                    hydro.umom_testfunctions, hydro.vmom_testfunctions, hydro.DIC_testfunctions,
                                                                    hydro.vertical_basis, hydro.time_basis, hydro.riverine_forcing.normal_alpha, hydro.riverine_forcing.normal_alpha_y,
                                                                    Q_trialfunctions=hydro.Q_trialfunctions, Q0=hydro.Q_solution)
                    else:
                        weakforms.add_linearised_nonlinear_terms(a, hydro.model_options, hydro.numerical_information, hydro.geometric_information, hydro.constant_physical_parameters, hydro.spatial_parameters,
                                                                    hydro.alpha_trialfunctions, hydro.alpha_solution, hydro.beta_trialfunctions, hydro.beta_solution,
                                                                    hydro.gamma_trialfunctions, hydro.gamma_solution,
                                                                    hydro.umom_testfunctions, hydro.vmom_testfunctions, hydro.DIC_testfunctions,
                                                                    hydro.vertical_basis, hydro.time_basis, hydro.riverine_forcing.normal_alpha)
            forms_time = timeit.default_timer() - forms_start
            if print_log:
                print(f"    Weak form construction took {np.round(forms_time, 3)} seconds")

            # Assemble system matrix
            assembly_start = timeit.default_timer()
            if parallel:
                with ngsolve.TaskManager():
                    a.Assemble()
            else:
                a.Assemble()
            assembly_time = timeit.default_timer() - assembly_start
            if print_log:
                print(f"    Assembly took {np.round(assembly_time, 3)} seconds")

            if matrix_analysis:
                fig, ax = plt.subplots(1, 3)
                freedofs = get_freedof_list(hydro.femspace.FreeDofs())
                mat = remove_fixeddofs_from_csr(basematrix_to_csr_matrix(a.mat), freedofs).todense()
                ax[0].spy(mat)
                im = ax[1].imshow(mat, cmap='RdBu', vmin = -np.amax(np.absolute(mat)), vmax = np.amax(np.absolute(mat)))
                print(f"Largest matrix element has magnitude {np.amax(np.absolute(mat))}")

            # Solve linearisation
            rhs = hydro.solution_gf.vec.CreateVector()
            hydro.total_bilinearform.Apply(hydro.solution_gf.vec, rhs)

            if matrix_analysis:
                ax[2].imshow(np.tile(rhs.FV().NumPy()[freedofs], (len(freedofs),1)).T, cmap='RdBu', vmin=-np.amax(rhs.FV().NumPy()), vmax=np.amax(rhs.FV().NumPy()))
                plt.show()

 
            du = ngsolve.GridFunction(hydro.femspace)
            for i in range(hydro.femspace.dim):
                du.components[i].Set(0, ngsolve.BND) # homogeneous boundary conditions

            inversion_start = timeit.default_timer()
            if linear_solver == 'pardiso':
                if parallel:
                    with ngsolve.TaskManager():
                        du.vec.data = a.mat.Inverse(freedofs=hydro.femspace.FreeDofs(), inverse='pardiso') * rhs
                else:
                    du.vec.data = a.mat.Inverse(freedofs=hydro.femspace.FreeDofs(), inverse='pardiso') * rhs
            elif linear_solver == 'scipy_direct':

                freedof_list = get_freedof_list(hydro.femspace.FreeDofs())
                mat = remove_fixeddofs_from_csr(basematrix_to_csr_matrix(a.mat), freedof_list)
                rhs_arr = rhs.FV().NumPy()[freedof_list]

                sol = spsolve(mat, rhs_arr)
                du.vec.FV().NumPy()[freedof_list] = sol
            elif linear_solver == 'pypardiso':

                freedof_list = get_freedof_list(hydro.femspace.FreeDofs())
                mat = remove_fixeddofs_from_csr(basematrix_to_csr_matrix(a.mat), freedof_list)
                rhs_arr = rhs.FV().NumPy()[freedof_list]

                sol = pypardiso.spsolve(mat, rhs_arr) 
                du.vec.FV().NumPy()[freedof_list] = sol

            elif linear_solver == 'bicgstab':
                freedof_list = get_freedof_list(hydro.femspace.FreeDofs())
                mat = remove_fixeddofs_from_csr(basematrix_to_csr_matrix(a.mat), freedof_list)
                initial_guess = hydro.solution_gf.vec.FV().NumPy()[freedof_list]
                sol, exitcode = bicgstab(mat, rhs_arr, initial_guess)

                if exitcode == 0:
                    print(f"    Bi-CGSTAB did not converge in 500 iterations")
                else:
                    print(f"    Bi-CGSTAB converged in {exitcode} iterations")

                du.vec.FV().NumPy()[freedof_list] = sol
            else:
                raise ValueError(f"Linear solver '{linear_solver}' not known to the system.")
            
            inversion_time = timeit.default_timer() - inversion_start
            if print_log:
                print(f"    Inversion took {np.round(inversion_time, 3)} seconds")

            # UPDATE ITERATE

            hydro.solution_gf.vec.data = hydro.solution_gf.vec.data - du.vec.data

            # Compute stopping criterion
            residual = hydro.solution_gf.vec.CreateVector()
            apply_start = timeit.default_timer()
            if parallel:
                with ngsolve.TaskManager():
                    hydro.total_bilinearform.Apply(hydro.solution_gf.vec, residual)
            else:
                hydro.total_bilinearform.Apply(hydro.solution_gf.vec, residual)
            
            apply_time = timeit.default_timer() - apply_start
            if print_log:
                print(f"    Evaluating weak form at current Newton iterate took {np.round(apply_time, 3)} seconds.")

            homogenise_essential_Dofs(residual, hydro.femspace.FreeDofs()) # remove the Dirichlet DOFs from the stopping criterion results, as this skews the results, and these DOFs are fixed anyway

            if stopcriterion == 'matrix_norm':
                stopcriterion_value = abs(ngsolve.InnerProduct(hydro.solution_gf.vec - previous_iterate.vec, residual))
            elif stopcriterion == 'scaled_2norm':
                stopcriterion_value = ngsolve.sqrt(ngsolve.InnerProduct(residual, residual) / hydro.nfreedofs)
            elif stopcriterion == 'relative_newtonstepsize':
                stopcriterion_value = ngsolve.sqrt(ngsolve.InnerProduct(hydro.solution_gf.vec - previous_iterate.vec, hydro.solution_gf.vec - previous_iterate.vec)) / ngsolve.sqrt(ngsolve.InnerProduct(hydro.solution_gf.vec, hydro.solution_gf.vec))
            else:
                raise ValueError(f"Stopping criterion '{stopcriterion}' not known to the system.")
            
            if print_log:
                print(f"    Stopping criterion value is {stopcriterion_value}\n")


            # PLOT INTERMEDIATE RESULTS

            if plot_intermediate_results == 'all':
                if hydro.model_options['sea_boundary_treatment'] == 'exact':
                    x_scaling = hydro.geometric_information['x_scaling']
                    sea_interpolant = ((hydro.geometric_information['riverine_boundary_x'] / x_scaling) + (hydro.geometric_information['L_BL_river']/x_scaling) + (hydro.geometric_information['L_R_river']/x_scaling) + \
                                    (hydro.geometric_information['L_RA_river']/x_scaling) - ngsolve.x) / \
                                    ((hydro.geometric_information['riverine_boundary_x']+hydro.geometric_information['L_BL_river']+hydro.geometric_information['L_R_river']+hydro.geometric_information['L_RA_river'] +
                                        hydro.geometric_information['L_BL_sea'] + hydro.geometric_information['L_R_sea'] + hydro.geometric_information['L_RA_sea']) / x_scaling)
                    sea_interpolant_x = -1 / ((hydro.geometric_information['riverine_boundary_x']+hydro.geometric_information['L_BL_river']+hydro.geometric_information['L_R_river']+hydro.geometric_information['L_RA_river'] +
                                        hydro.geometric_information['L_BL_sea'] + hydro.geometric_information['L_R_sea'] + hydro.geometric_information['L_RA_sea']) / x_scaling)
                if hydro.model_options['river_boundary_treatment'] == 'exact':
                    river_interpolant = (-(hydro.geometric_information['L_BL_sea']/x_scaling) - (hydro.geometric_information['L_R_sea']/x_scaling) - \
                                        (hydro.geometric_information['L_RA_sea']/x_scaling) + ngsolve.x) / \
                                        ((hydro.geometric_information['riverine_boundary_x']+hydro.geometric_information['L_BL_river']+hydro.geometric_information['L_R_river']+hydro.geometric_information['L_RA_river'] +
                                        hydro.geometric_information['L_BL_sea'] + hydro.geometric_information['L_R_sea'] + hydro.geometric_information['L_RA_sea']) / x_scaling)
                    river_interpolant_x = 1 / ((hydro.geometric_information['riverine_boundary_x']+hydro.geometric_information['L_BL_river']+hydro.geometric_information['L_R_river']+hydro.geometric_information['L_RA_river'] +
                                        hydro.geometric_information['L_BL_sea'] + hydro.geometric_information['L_R_sea'] + hydro.geometric_information['L_RA_sea']) / x_scaling)
                # plotting to test where convergence goes wrong
                for m in range(M):
                    for i in range(-imax, imax+1):
                        plot_CF_colormap(hydro.alpha_solution[m][i], hydro.mesh, refinement_level=3, show_mesh=True, title=f'alpha_({m},{i})', save = f"iteration{newton_counter}_alpha({m},{i})")
                        plot_CF_colormap(ngsolve.grad(hydro.alpha_solution[m][i])[0], hydro.mesh, refinement_level=3, show_mesh=True, title=f'alphax_({m},{i})', save = f"iteration{newton_counter}_alphax({m},{i})")
                        plot_CF_colormap(ngsolve.grad(hydro.alpha_solution[m][i])[1], hydro.mesh, refinement_level=3, show_mesh=True, title=f'alphay_({m},{i})', save = f"iteration{newton_counter}_alphay({m},{i})")
                        plot_CF_colormap(hydro.beta_solution[m][i], hydro.mesh, refinement_level=3, show_mesh=True, title=f'beta_({m},{i})', save = f"iteration{newton_counter}_beta({m},{i})")
                        plot_CF_colormap(ngsolve.grad(hydro.beta_solution[m][i])[0], hydro.mesh, refinement_level=3, show_mesh=True, title=f'betax_({m},{i})', save = f"iteration{newton_counter}_betax({m},{i})")
                        plot_CF_colormap(ngsolve.grad(hydro.beta_solution[m][i])[1], hydro.mesh, refinement_level=3, show_mesh=True, title=f'betay_({m},{i})', save = f"iteration{newton_counter}_betay({m},{i})")

                for i in range(-imax, imax+1):
                    if hydro.model_options['sea_boundary_treatment'] == 'exact':
                        plot_CF_colormap(hydro.gamma_solution[i]+hydro.A_solution[i]*sea_interpolant, hydro.mesh, refinement_level=3, show_mesh=True, title=f'gamma_({i})', save = f"iteration{newton_counter}_gamma({i})")
                    else:
                        plot_CF_colormap(hydro.gamma_solution[i], hydro.mesh, refinement_level=3, show_mesh=True, title=f'gamma_({i})', save = f"iteration{newton_counter}_gamma({i})")
                    u_DA = sum([hydro.vertical_basis.tensor_dict['G4'](m) * hydro.alpha_solution[m][i] for m in range(M)])
                    plot_CF_colormap(u_DA, hydro.mesh, refinement_level=3, show_mesh=True, title=f'u_DA_({i})', save = f"iteration{newton_counter}_uDA({i})")
                    v_DA = sum([hydro.vertical_basis.tensor_dict['G4'](m) * hydro.beta_solution[m][i] for m in range(M)])
                    plot_CF_colormap(v_DA, hydro.mesh, refinement_level=3, show_mesh=True, title=f'v_DA_({i})', save = f"iteration{newton_counter}_vDA({i})")


                plt.close(fig = 'all')
            elif plot_intermediate_results == 'overview':
                if hydro.model_options['sea_boundary_treatment'] == 'exact':
                    x_scaling = hydro.geometric_information['x_scaling']
                    sea_interpolant = ((hydro.geometric_information['riverine_boundary_x'] / x_scaling) + (hydro.geometric_information['L_BL_river']/x_scaling) + (hydro.geometric_information['L_R_river']/x_scaling) + \
                                    (hydro.geometric_information['L_RA_river']/x_scaling) - ngsolve.x) / \
                                    ((hydro.geometric_information['riverine_boundary_x']+hydro.geometric_information['L_BL_river']+hydro.geometric_information['L_R_river']+hydro.geometric_information['L_RA_river'] +
                                        hydro.geometric_information['L_BL_sea'] + hydro.geometric_information['L_R_sea'] + hydro.geometric_information['L_RA_sea']) / x_scaling)
                    sea_interpolant_x = -1 / ((hydro.geometric_information['riverine_boundary_x']+hydro.geometric_information['L_BL_river']+hydro.geometric_information['L_R_river']+hydro.geometric_information['L_RA_river'] +
                                        hydro.geometric_information['L_BL_sea'] + hydro.geometric_information['L_R_sea'] + hydro.geometric_information['L_RA_sea']) / x_scaling)
                if hydro.model_options['river_boundary_treatment'] == 'exact':
                    river_interpolant = (-(hydro.geometric_information['L_BL_sea']/x_scaling) - (hydro.geometric_information['L_R_sea']/x_scaling) - \
                                        (hydro.geometric_information['L_RA_sea']/x_scaling) + ngsolve.x) / \
                                        ((hydro.geometric_information['riverine_boundary_x']+hydro.geometric_information['L_BL_river']+hydro.geometric_information['L_R_river']+hydro.geometric_information['L_RA_river'] +
                                        hydro.geometric_information['L_BL_sea'] + hydro.geometric_information['L_R_sea'] + hydro.geometric_information['L_RA_sea']) / x_scaling)
                    river_interpolant_x = 1 / ((hydro.geometric_information['riverine_boundary_x']+hydro.geometric_information['L_BL_river']+hydro.geometric_information['L_R_river']+hydro.geometric_information['L_RA_river'] +
                                        hydro.geometric_information['L_BL_sea'] + hydro.geometric_information['L_R_sea'] + hydro.geometric_information['L_RA_sea']) / x_scaling)
                for i in range(-imax, imax+1):
                    if hydro.model_options['sea_boundary_treatment'] == 'exact':
                        plot_CF_colormap(hydro.gamma_solution[i]+hydro.A_solution[i]*sea_interpolant, hydro.mesh, refinement_level=3, show_mesh=True, title=f'gamma_({i})', save = f"iteration{newton_counter}_gamma({i})")
                    else:
                        plot_CF_colormap(hydro.gamma_solution[i], hydro.mesh, refinement_level=3, show_mesh=True, title=f'gamma_({i})', save = f"iteration{newton_counter}_gamma({i})")
                    u_DA = sum([hydro.vertical_basis.tensor_dict['G4'](m) * hydro.alpha_solution[m][i] for m in range(M)])
                    u_DAx = sum([hydro.vertical_basis.tensor_dict['G4'](m) * ngsolve.grad(hydro.alpha_solution[m][i])[0] for m in range(M)])
                    u_DAy = sum([hydro.vertical_basis.tensor_dict['G4'](m) * ngsolve.grad(hydro.alpha_solution[m][i])[1] for m in range(M)])
                    plot_CF_colormap(u_DA, hydro.mesh, refinement_level=3, show_mesh=True, title=f'u_DA_({i})', save = f"iteration{newton_counter}_uDA({i})")
                    plot_CF_colormap(u_DAx, hydro.mesh, refinement_level=3, show_mesh=True, title=f'u_DAx_({i})', save = f"iteration{newton_counter}_uDAx({i})")
                    plot_CF_colormap(u_DAy, hydro.mesh, refinement_level=3, show_mesh=True, title=f'u_DAy_({i})', save = f"iteration{newton_counter}_uDAy({i})")

                    v_DA = sum([hydro.vertical_basis.tensor_dict['G4'](m) * hydro.beta_solution[m][i] for m in range(M)])
                    v_DAx = sum([hydro.vertical_basis.tensor_dict['G4'](m) * ngsolve.grad(hydro.beta_solution[m][i])[0] for m in range(M)])
                    v_DAy = sum([hydro.vertical_basis.tensor_dict['G4'](m) * ngsolve.grad(hydro.beta_solution[m][i])[1] for m in range(M)])
                    plot_CF_colormap(v_DA, hydro.mesh, refinement_level=3, show_mesh=True, title=f'v_DA_({i})', save = f"iteration{newton_counter}_vDA({i})")
                    plot_CF_colormap(v_DAx, hydro.mesh, refinement_level=3, show_mesh=True, title=f'v_DAx_({i})', save = f"iteration{newton_counter}_vDAx({i})")
                    plot_CF_colormap(v_DAy, hydro.mesh, refinement_level=3, show_mesh=True, title=f'v_DAy_({i})', save = f"iteration{newton_counter}_vDAy({i})")

            if stopcriterion_value < tolerance:
                if print_log:
                    print('Newton-Raphson method converged')
                break
            else:
                previous_iterate = copy.copy(hydro.solution_gf)
    if print_log:
        print('\nSolution process complete.')

# tools for matrix analysis

def is_symmetric(mat: sp.csr_matrix, tol=1e-12):
    """Returns True if a sparse matrix (CSR) is symmetric within a certain (absolute) tolerance.
    
    Arguments:

    - mat (sp.csr_matrix):      sparse matrix to be checked;
    - tol (float):              if elements are further apart than this number, the function returns False.
    
    """
    diff = mat - mat.transpose()
    return not np.any(np.absolute(diff.data) >= tol * np.ones_like(diff.data))


def is_antisymmetric(mat: sp.csr_matrix, tol=1e-12):
    """Returns True if a sparse matrix (CSR) is antisymmetric within a certain (absolute) tolerance.
    
    Arguments:

    - mat (sp.csr_matrix):      sparse matrix to be checked;
    - tol (float):              if elements are further apart than this number, the function returns False.
    
    """
    diff = mat + mat.transpose()
    return not np.any(np.absolute(diff.data) >= tol * np.ones_like(diff.data))


def get_eigenvalue(mat, shift_inverse=False, maxits = 100, tol=1e-9):
    """Computes the largest eigenvalue of a matrix (sparse or otherwise) using the power method. If shift_inverse is True, then the method computes the smallest eigenvalue using
    the shift-inverse version of the power method.
    
    Arguments:

        - mat:                  matrix for which the eigenvalue is computed;
        - shift_inverse:        if True, uses shift inverse version of power method to compute the smallest eigenvalue.
        - maxits:               maximum number of iterations
    
    """

    previous_vec = np.random.randn(mat.shape[0]) # starting vector
    previous_eig = 0

    if not shift_inverse:
        for i in range(maxits):
            new_vec = mat @ previous_vec
            new_eig = np.inner(np.conj(previous_vec), new_vec)
            previous_vec = new_vec / np.linalg.norm(new_vec, ord=2)

            stopvalue = abs(new_eig - previous_eig) / abs(new_eig)
            if stopvalue < tol:
                break

            previous_eig = new_eig

            if i == maxits - 1:
                print('Method did not converge')
    else:
        for i in range(maxits):
            new_vec = spsolve(mat, previous_vec)
            new_eig = np.inner(np.conj(previous_vec), new_vec)

            previous_vec = new_vec / np.linalg.norm(new_vec, ord=2)

            stopvalue = abs(new_eig - previous_eig) / abs(new_eig)
            if stopvalue < tol:
                break

            previous_eig = new_eig

            if i == maxits - 1:
                print('Method did not converge')

    if shift_inverse:
        return 1 / new_eig
    else:
        return new_eig


def get_condition_number(mat, maxits = 100, tol=1e-9):
    """Computes 2-condition number of a sparse matrix by approximating the largest and smallest (in modulus) eigenvalues.
    
    Arguments:

    - mat:              sparse matrix;
    - maxits:           maximum number of iterations used in the power method;
    - tol:              tolerance used in the power method.

    """
    large_eig = abs(get_eigenvalue(mat, shift_inverse=False, maxits=maxits, tol=tol))
    small_eig = abs(get_eigenvalue(mat, shift_inverse=True, maxits=maxits, tol=tol))
    return large_eig / small_eig



# Linear solvers


def bicgstab(A, f, u0, tol=1e-12, maxits = 500):
    """Carries out a Bi-CGSTAB solver based on the pseudocode in Van der Vorst (1992). This function has the option for a
    reduced basis preconditioner if reduced_A and transition_mass_matrix are specified. Returns the solution and an exitcode
    indicating how many iterations it took for convergence. If exitcode=0 is returned, then the method did not converge.
    This implementation is heavily based on the scipy-implementation that can be found on https://github.com/scipy/scipy/blob/main/scipy/sparse/linalg/_isolve/iterative.py.
    The function is also included in this file, so that the reduced-basis preconditioner can be used.
    
    Arguments:
    
    - A:                        system matrix;
    - f:                        right-hand side;
    - u0:                       initial guess;
    - tol:                      tolerance for stopping criterion;    
    """
    # initialising parameters
    r = f - A @ u0
    shadow_r0 = np.copy(r)

    previous_rho = 1
    alpha = 1
    omega = 1

    v = np.zeros_like(u0)
    p = np.zeros_like(u0)

    solution = u0[:]
    f_norm = np.linalg.norm(f, 2) # precompute this so this need not happen every iteration

    for i in range(maxits):
        rho = np.inner(shadow_r0, r)

        beta = (rho / previous_rho) * (alpha / omega)

        p -= omega * v
        p *= beta
        p += r

        preconditioned_p = np.copy(p)

        v = A @ preconditioned_p
        alpha = rho / np.inner(shadow_r0, v)
        s = r - alpha * v

        
        z = np.copy(s)

        t = A @ z
        omega = np.inner(t, s) / np.inner(t, t)

        solution += alpha * p + omega * z
        r = s - omega * t
        
        if np.linalg.norm(r, 2) / f_norm < tol:
            return solution, i+1 # return the solution and how many iterations it took for convergence

        previous_rho = np.copy(rho)

    return solution, 0 # return the solution after the final iteration, but with a 0 indicating non-convergence


