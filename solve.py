import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from scipy.sparse.csgraph import reverse_cuthill_mckee
import timeit
import ngsolve
import copy
import matplotlib.pyplot as plt
from contextlib import nullcontext

from NiFlow.hydrodynamics import *
import NiFlow.define_weak_forms as weakforms
from NiFlow.utils import *
from NiFlow.linear_solver import *


def solve(hydro: Hydrodynamics, max_iterations: int = 10, tolerance: float = 1e-12, linear_solver = 'pypardiso', preconditioner = 'identity',
          continuation_parameters: dict = {'advection_epsilon': [1], 'surface_epsilon': [1], 'Av': [1], 'Ah': [1]}, stopcriterion = 'scaled_2norm',
          plot_intermediate_results='none', num_threads=1, static_condensation=False, matrix_analysis=False, print_log:bool=True, reduce_bandwidth:bool=False, oseen_linearisation=False):

    """
    
    Compute the solution of the model given by a Hydrodynamics-object and add it as an attribute to the Hydrodynamics-object. Solution is computed using a Newton-Raphson
    method, combined with a continuation (homology) method to guide the Newton method towards the correct solution.
    
    Arguments:
    
        - hydro (Hydrodynamics):            object containing the (weak) model equations and options;
        - max_iterations (int):             maximum number of Newton iterations per continuation step;
        - tolerance (float):                if the stopping criterion is less than this value, the Newton method terminates and the procedure moves to the next continuation step;
        - linear_solver:                    choice of linear solver; options: 'pardiso', 'pypardiso', 'scipy_direct', 'bicgstab', 'gmres', 'uzawa'
        - preconditioner:                   choice of preconditioner; options: 'identity', 'block_diagonal_diagonal', 'block_diagonal_utriangular', 'block_diagonal_ltriangular'
        - continuation_parameters (dict):   dictionary with keys 'advection_epsilon' and 'Av' and 'Ah', with values indicating what the default value of these parameters should be multiplied by in each continuation step;
        - stopcriterion:                    choice of stopping criterion; options: 'relative_residual_norm', 'matrix_norm', 'scaled_2norm', 'relative_newtonstepsize';
        - plot_intermediate_results:        indicates whether intermediate results should be plotted and saved; options: 'none' (default), 'all' and 'overview'.
        - num_threads:                      flag indicating whether time-costly operations should be performed in parallel (see https://docu.ngsolve.org/latest/how_to/howto_parallel.html) with num_threads threads. If num_threads==1, then no parallel computation performed.
        - static_condensation:              flag indicating whether static condensation should be used. 
        - matrix_analysis:                  if True, plots the non-zero elements of the matrix, the size of all entries via a colormap, and the right-hand side vector using a colormap.
        - print_log:                        if True, prints runtimes of the steps of the solution process, as well as stopping criterion values.
        - reduce_bandwidth:                 if True, applies the reverse Cuthill-McKee algorithm to the matrix and right-hand side vector.
    
    """

    M = hydro.numerical_information['M']
    imax = hydro.numerical_information['imax']

    # Quick argument check

    if len(continuation_parameters['advection_epsilon']) == len(continuation_parameters['Av']) and len(continuation_parameters['Ah']) == len(continuation_parameters['Av']):
        num_continuation_steps = len(continuation_parameters['advection_epsilon'])
    else:
        raise ValueError(f"Length of both continuation parameter lists must be equal; now the lenghts are {len(continuation_parameters['advection_epsilon'])} and {len(continuation_parameters['Av'])} and {len(continuation_parameters['Ah'])}")

    # set options for parallel computation
    if num_threads > 1:
        ngsolve.SetHeapSize(200_000_000)
        ngsolve.SetNumThreads(num_threads)

    context = ngsolve.TaskManager() if num_threads > 1 else nullcontext()

    # Report that solution procedure is about to start.
    if print_log:
        print(f"Initiating solution procedure for hydrodynamics-model with {hydro.numerical_information['M']} vertical components and {hydro.numerical_information['imax'] + 1} tidal constituents (including residual).\nThe total number of free degrees of freedom is {hydro.nfreedofs}.")

    # Set initial guess
    sol = ngsolve.GridFunction(hydro.femspace)

    # surface
    sol.components[2*(M)*(2*imax+1)].Set(hydro.seaward_forcing.cfdict[0])

    for q in range(1, imax + 1):
        sol.components[2*(M)*(2*imax+1) + q].Set(hydro.seaward_forcing.cfdict[-q])
        sol.components[2*(M)*(2*imax+1) + imax + q].Set(hydro.seaward_forcing.cfdict[q])

    # surface coefficients A_l
    if hydro.model_options['sea_boundary_treatment'] == 'exact':
        sol.components[2*(M)*(2*imax+1) + (2*imax + 1)].Set(hydro.seaward_forcing.amplitudes[0] * np.sqrt(2))
        for q in range(1, imax + 1):
            sol.components[2*(M)*(2*imax + 1) + (2*imax + 1) + q].Set(-hydro.seaward_forcing.amplitudes[q]*np.sin(hydro.seaward_forcing.phases[q-1])) # phase_list starts at semidiurnal component instead of residual; therefore index - 1
            sol.components[2*(M)*(2*imax + 1) + (2*imax + 1) + imax + q].Set(hydro.seaward_forcing.amplitudes[q]*np.cos(hydro.seaward_forcing.phases[q-1])) 

        # river discharge
    # if hydro.model_options['river_boundary_treatment'] != 'exact':
    for m in range(M): # do this always.
        sol.components[m * (2*imax + 1)].Set(hydro.constant_physical_parameters['discharge'] * hydro.riverine_forcing.normal_alpha[m], ngsolve.BND)
            
    hydro.solution_gf = sol 

    # Save true values of advection_epsilon and Av before modifying them in the continuation (homology) method

    true_epsilon = copy.copy(hydro.constant_physical_parameters['advection_epsilon'])
    true_surface_epsilon = copy.copy(hydro.constant_physical_parameters['surface_epsilon'])
    true_Av = copy.copy(hydro.constant_physical_parameters['Av'])
    true_Ah = copy.copy(hydro.constant_physical_parameters['Ah'])

    for continuation_counter in range(num_continuation_steps):
        hydro.constant_physical_parameters['advection_epsilon'] = true_epsilon * continuation_parameters['advection_epsilon'][continuation_counter]
        hydro.constant_physical_parameters['surface_epsilon'] = true_surface_epsilon * continuation_parameters['surface_epsilon'][continuation_counter]
        hydro.constant_physical_parameters['Av'] = true_Av * continuation_parameters['Av'][continuation_counter]
        hydro.constant_physical_parameters['Ah'] = true_Ah * continuation_parameters['Ah'][continuation_counter]

        if print_log:
            if num_continuation_steps > 1:
                print(f"\nCONTINUATION STEP {continuation_counter}: Advection Epsilon = {hydro.constant_physical_parameters['advection_epsilon']}, Surface epsilon = {hydro.constant_physical_parameters['surface_epsilon']}, Av = {hydro.constant_physical_parameters['Av']}, Ah = {hydro.constant_physical_parameters['Ah']}.\n")
            print("Setting up full weak form\n")

        with context:
            hydro.setup_weak_form(static_condensation=static_condensation)

        # Start the Newton method

        previous_iterate = copy.copy(hydro.solution_gf)

        for newton_counter in range(max_iterations):
            if print_log:
                print(f"Newton-Raphson iteration {newton_counter}")
            hydro.restructure_solution() # restructure solution so that hydro.alpha_solution, hydro.beta_solution, and hydro.gamma_solution are specified.
            hydro.get_gradients(compiled=True)
            # Set-up weak form of the linearisation

            forms_start = timeit.default_timer()
            
            with context:
                a = ngsolve.BilinearForm(hydro.femspace, condense=static_condensation)

                if hydro.model_options['sea_boundary_treatment'] == 'exact':
                    A_trial_functions, sea_bc_test_functions = hydro.A_trialfunctions, hydro.sea_boundary_testfunctions
                else:
                    A_trial_functions, sea_bc_test_functions = None, None

                if hydro.model_options['river_boundary_treatment'] == 'exact':
                    Q_trial_functions, river_bc_test_functions = hydro.Q_trialfunctions, hydro.river_boundary_testfunctions
                    Q0 = hydro.Q_solution
                    normal_alpha, normal_alpha_y = hydro.riverine_forcing.normal_alpha, hydro.riverine_forcing.normal_alpha_y
                else:
                    Q_trial_functions, Q0, river_bc_test_functions, normal_alpha, normal_alpha_y = None, None, None, None, None

                # compile_previous_newton_iterate(hydro)
                weakforms.construct_linearised_weak_form(a, hydro.model_options, hydro.geometric_information, hydro.numerical_information,
                                                            hydro.constant_physical_parameters, hydro.spatial_parameters, hydro.spatial_parameters_grad,
                                                            hydro.time_basis, hydro.vertical_basis,
                                                            hydro.alpha_trialfunctions, hydro.alpha_solution, hydro.beta_trialfunctions, hydro.beta_solution,
                                                            hydro.gamma_trialfunctions, hydro.gamma_solution,
                                                            hydro.umom_testfunctions, hydro.vmom_testfunctions, hydro.DIC_testfunctions,
                                                            A_trial_functions=A_trial_functions, Q_trial_functions=Q_trial_functions, Q0 = Q0,
                                                            sea_bc_test_functions=sea_bc_test_functions, river_bc_test_functions=river_bc_test_functions,
                                                            normal_alpha=normal_alpha, normal_alpha_y=normal_alpha_y, operator='full', oseen_linearisation=oseen_linearisation)


            forms_time = timeit.default_timer() - forms_start
            if print_log:
                print(f"    Weak form construction took {np.round(forms_time, 3)} seconds")

            # Assemble system matrix
            assembly_start = timeit.default_timer()
            with context:
                a.Assemble()
            assembly_time = timeit.default_timer() - assembly_start
            if print_log:
                print(f"    Assembly took {np.round(assembly_time, 3)} seconds")

            if matrix_analysis:
                fig, ax = plt.subplots()
                freedofs = get_freedof_list(hydro.femspace.FreeDofs())
                mat = remove_fixeddofs_from_csr(basematrix_to_csr_matrix(a.mat), freedofs)
                if reduce_bandwidth:
                    cmind = reverse_cuthill_mckee(mat)
                    mat = mat[cmind[:, None], cmind]
                # ax.spy(mat.todense())
                ax.set_xticklabels(["" for _ in ax.get_xticks()])
                ax.set_yticklabels(["" for _ in ax.get_yticks()])
                for i in range(mat.shape[0]):
                    print(f'{i}: maximum is {np.amax(np.absolute(mat[i, :]))}')
                # im = ax[1].imshow(mat, cmap='RdBu', vmin = -np.amax(np.absolute(mat)), vmax = np.amax(np.absolute(mat)))
                print(f"Largest matrix element has magnitude {np.amax(np.absolute(mat))}")

            # Solve linearisation
            if print_log:
                rhs_creation_start = timeit.default_timer()
            rhs = hydro.solution_gf.vec.CreateVector()
            with context:
                hydro.total_bilinearform.Apply(hydro.solution_gf.vec, rhs)

            if print_log:
                print(f"    Construction of right-hand side vector took {np.round(timeit.default_timer() - rhs_creation_start, 3)} seconds")
 
            du = ngsolve.GridFunction(hydro.femspace)
            for i in range(hydro.femspace.dim):
                du.components[i].Set(0, ngsolve.BND) # homogeneous boundary conditions

            if linear_solver != 'pardiso':
                conversion_start = timeit.default_timer()
                # Extract matrix and vector from ngsolve
                freedof_list = get_freedof_list(hydro.femspace.FreeDofs())
                mat = remove_fixeddofs_from_csr(basematrix_to_csr_matrix(a.mat), freedof_list)
                rhs_arr = rhs.FV().NumPy()[freedof_list]
                conversion_time = timeit.default_timer() - conversion_start
                if print_log:
                    print(f"    Conversion to scipy-sparse matrix took {np.round(conversion_time, 3)} seconds")

            inversion_start = timeit.default_timer()
            if linear_solver == 'pardiso':
                with context:
                    if static_condensation: # see https://docu.ngsolve.org/latest/i-tutorials/unit-1.4-staticcond/staticcond.html for explanation of why we do the solution like this
                        invS = a.mat.Inverse(freedofs=hydro.femspace.FreeDofs(coupling=True), inverse='pardiso')
                        ext = ngsolve.IdentityMatrix() + a.harmonic_extension
                        extT = ngsolve.IdentityMatrix() + a.harmonic_extension_trans
                        invA = ext @ invS @ extT + a.inner_solve
                        du.vec.data += invA * rhs
                        # for testing static condensation
                        if print_log: 
                            dof_types = dof_division(hydro.femspace)
                            for ctype in dof_types.keys():
                                if str(ctype) == "COUPLING_TYPE.LOCAL_DOF":
                                    num_local_dofs = dof_types[ctype]
                            print(f"    Number of effective degrees of freedom (non-local dofs) equal to {hydro.nfreedofs - num_local_dofs}.")
                    else:
                        du.vec.data += a.mat.Inverse(freedofs=hydro.femspace.FreeDofs(), inverse='pardiso') * rhs

            elif linear_solver == 'scipy_direct':
                solver = scipyLU_solver
                sol = solver.solve(mat, rhs_arr, rcm=reduce_bandwidth)
                du.vec.FV().NumPy()[freedof_list] = sol
            elif linear_solver == 'pypardiso':
                solver = pypardiso_spsolve
                sol = solver.solve(mat, rhs_arr, rcm=reduce_bandwidth) 
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
            with context:
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
            elif stopcriterion == 'relative_residual_norm':
                stopcriterion_value = ngsolve.sqrt(ngsolve.InnerProduct(residual, residual) / ngsolve.InnerProduct(rhs, rhs))
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
        hydro.restructure_solution() #
        hydro.get_gradients(compiled=True) 
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


def compile_previous_newton_iterate(hydro: Hydrodynamics):
    for i in range(-hydro.numerical_information['imax'], hydro.numerical_information['imax'] + 1):
        hydro.gamma_solution[i] = hydro.gamma_solution[i].Compile()
        for m in range(hydro.numerical_information['M']):
            hydro.alpha_solution[m][i].Compile()
            hydro.beta_solution[m][i].Compile()
