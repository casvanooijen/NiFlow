import numpy as np
import pypardiso
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as splin
import timeit
from scipy.sparse.csgraph import reverse_cuthill_mckee

from NiFlow.utils import *


class Preconditioner(object):

    def __init__(self, precondition_step_func):
        """precondition_step_func takes mat, vec and returns a vector"""
        self.step = precondition_step_func

def identity_prec_step(vec):
    return vec

identity_preconditioner = Preconditioner(identity_prec_step)

class Solver(object):

    def __init__(self, solve_func, preconditioner: Preconditioner = identity_preconditioner, is_iterative: bool=False):
        """solve_func takes mat, vec, and optionally initial guess and/or preconditioner"""
        self.solve = solve_func    
        self.prec = preconditioner
        self.is_iterative = is_iterative


    def solve_matrix(self, system_matrix, rhs_matrix):
        if not isinstance(rhs_matrix, np.ndarray):
            rhs_matrix = rhs_matrix.todense()

        solution_matrix = np.zeros_like(rhs_matrix)
        for i in range(solution_matrix.shape[1]):
            solution_matrix[:, i] = np.reshape(self.solve(system_matrix, rhs_matrix[:, i]), (solution_matrix.shape[0], 1))

        return solution_matrix

        


# SOLVERS ==========================================================================


def pypardiso_func(mat, vec, rcm=False):
    if rcm:
        cmind = reverse_cuthill_mckee(mat)
        mat = mat[cmind[:, None], cmind]
        vec = vec[cmind]
        undo_cm = np.argsort(cmind)

    solution = pypardiso.spsolve(mat, vec)
    if rcm:
        solution = solution[undo_cm]

    return solution

pypardiso_spsolve = Solver(pypardiso_func)


def scipyLU_func(mat, vec, rcm=False):
    if rcm:
        cmind = reverse_cuthill_mckee(mat)
        mat = mat[cmind[:, None], cmind]
        vec = vec[cmind]
        undo_cm = np.argsort(cmind)

    solution = splin.spsolve(mat, vec)
    if rcm:
        solution = solution[undo_cm]

    return solution

scipyLU_solver = Solver(scipyLU_func)

def bicgstab_func(mat, vec, init_guess, prec: Preconditioner, tol=1e-12, maxits=500):
    """Carries out a Bi-CGSTAB solver based on the pseudocode in Van der Vorst (1992). This function has the option for a
    reduced basis preconditioner if reduced_A and transition_mass_matrix are specified. Returns the solution and an exitcode
    indicating how many iterations it took for convergence. If exitcode=0 is returned, then the method did not converge.
    This implementation is heavily based on the scipy-implementation that can be found on https://github.com/scipy/scipy/blob/main/scipy/sparse/linalg/_isolve/iterative.py.
    The function is also included in this file, so that the reduced-basis preconditioner can be used.
    
    Arguments:
    
    - mat:                      system matrix;
    - vec:                      right-hand side;
    - init_guess:               initial guess;
    - tol:                      tolerance for stopping criterion; 
    - maxits:                   maximum number of iterations.   
    """
    # initialising parameters
    r = vec - mat @ init_guess
    shadow_r0 = np.copy(r)

    previous_rho = 1
    alpha = 1
    omega = 1

    v = np.zeros_like(init_guess)
    p = np.zeros_like(init_guess)

    solution = init_guess[:]
    f_norm = np.linalg.norm(vec, 2) # precompute this so this need not happen every iteration

    for i in range(maxits):
        rho = np.inner(shadow_r0, r)

        beta = (rho / previous_rho) * (alpha / omega)

        p -= omega * v
        p *= beta
        p += r

        preconditioned_p = prec.step(p)

        v = mat @ preconditioned_p
        alpha = rho / np.inner(shadow_r0, v)
        s = r - alpha * v

        
        z = prec.step(s)

        t = mat @ z
        omega = np.inner(t, s) / np.inner(t, t)

        solution += alpha * p + omega * z
        r = s - omega * t
        
        if np.linalg.norm(r, 2) / f_norm < tol:
            return solution, i+1 # return the solution and how many iterations it took for convergence

        print(np.linalg.norm(r, 2) / f_norm)

        previous_rho = np.copy(rho)

    return solution, 0 # return the solution after the final iteration, but with a 0 indicating non-convergence


bicgstab = Solver(bicgstab_func)


def make_uzawa(omega, momentum_block_size, mass_block_size, inner_solver = 'LU', boundary_handling=True):

    def uzawa_func(mat, vec, init_guess, tol=1e-12, maxits=500): # stopcriterion: ||r||/||f|| < tol

        A = slice_csr_matrix(mat, range(momentum_block_size), range(momentum_block_size))
        B1 = slice_csr_matrix(mat, range(momentum_block_size), range(momentum_block_size, momentum_block_size + mass_block_size))
        B2 = slice_csr_matrix(mat, range(momentum_block_size, momentum_block_size + mass_block_size), range(momentum_block_size))
        C = slice_csr_matrix(mat, range(momentum_block_size, momentum_block_size + mass_block_size), range(momentum_block_size, momentum_block_size + mass_block_size))

        fig, ax = plt.subplots(2, 2)
        ax[0, 0].spy(A.todense())
        ax[0, 1].spy(B1.todense())
        ax[1, 0].spy(B2.todense())
        ax[1, 1].spy(C.todense())
        plt.show()

        print(np.linalg.cond(A.todense(), 2))

        sol = init_guess[:]
        # prev_x = init_guess[:momentum_block_size]
        prev_y = init_guess[momentum_block_size:(momentum_block_size+mass_block_size)]

        # if mat.shape[0] > momentum_block_size + mass_block_size: # in which case there is an inner boundary condition, add the inner boundary condition part to 
        for i in range(maxits):
            if inner_solver == 'LU':
                new_x = splin.spsolve(A, vec[:momentum_block_size] - B1 @ prev_y)
            elif inner_solver == 'bicgstab':
                new_x = splin.bicgstab(A, vec[:momentum_block_size] - B1 @ prev_y)
            elif inner_solver == 'gmres':
                new_x = splin.gmres(A, vec[:momentum_block_size] - B1 @ prev_y)

            new_y = prev_y + omega * (B2 @ new_x + C @ prev_y - vec[momentum_block_size:(momentum_block_size + mass_block_size)])

            sol = np.concatenate((new_x, new_y))

            print(np.linalg.norm(mat @ sol - vec, 2))
            if np.linalg.norm(mat @ sol - vec, 2) < 1e-12:
                return sol, i+1 # return solution and number of iterations to convergence
            
            
            # prev_x = sol[:momentum_block_size]
            prev_y = sol[momentum_block_size:(momentum_block_size+mass_block_size)]

        return sol, -1 # return current solution and exit code -1 to indicate non-convergence

    return Solver(uzawa_func, is_iterative=True)


def make_gmres(preconditioner: Preconditioner):

    def gmres_func(mat, vec, init_guess=None):

        class gmres_counter(object): # class copied from stackoverflow comment by user ali_m, Nov 5 2011
            def __init__(self, disp=True):
                self._disp = disp
                self.niter = 0
                self.resid_list = []
            def __call__(self, x=None):
                self.niter += 1
                # self.resid_list.append(np.linalg.norm(mat @ x - vec, 2) / np.linalg.norm(vec, 2))
                if self._disp:
                    # print('iter %3i\trk = %s' % (self.niter, str(self.resid_list[-1])))
                    print(f"        Iteration {self.niter}")

        if init_guess is None:
            init_guess = np.zeros_like(vec)

        M_x = lambda x : preconditioner.step(x)
        prec_operator = splin.LinearOperator(mat.shape, M_x)
        counter = gmres_counter(disp=False)
        # return splin.gmres(mat, vec, init_guess, M=prec_operator, callback=counter, callback_type='x', rtol=1e-9, maxiter=2000), counter.resid_list
        return splin.gmres(mat, vec, init_guess, M=prec_operator, callback=counter, rtol=1e-9, maxiter=3000), counter.niter
    
    return Solver(gmres_func, preconditioner, is_iterative=True)


def make_bicgstab(preconditioner: Preconditioner):

    def bicgstab_func(mat, vec, init_guess=None):

        vec_norm = np.linalg.norm(vec, 2)

        class bicgstab_counter(object): # class copied from stackoverflow comment by user ali_m, Nov 5 2011
            def __init__(self, disp=True):
                self._disp = disp
                self.niter = 0
                self.resid_list = []
            def __call__(self, xk=None):
                self.niter += 1
                # self.resid_list.append(np.linalg.norm(mat @ xk - vec, 2) / np.linalg.norm(vec, 2))
                if self._disp:
                    # print('iter %3i\trk = %s' % (self.niter, str(self.resid_list[-1])))
                    print(f"        Iteration {self.niter}")

        if init_guess is None:
            init_guess = np.zeros_like(vec)
        M_x = lambda x : preconditioner.step(x)
        prec_operator = splin.LinearOperator(mat.shape, M_x)
        counter = bicgstab_counter(disp=False)
        # return splin.bicgstab(mat, vec, init_guess, M=prec_operator, callback=counter, rtol=1e-9, maxiter=2000), counter.resid_list
        return splin.bicgstab(mat, vec, init_guess, M=prec_operator, callback=counter, rtol=1e-9, maxiter=3000), counter.niter
    
    return Solver(bicgstab_func, preconditioner, is_iterative=True)


# def make_arrow_hurwicz(alpha, omega, momentum_block_size, mass_block_size, inner_solver, boundary_handling=True):

#     def arrow_hurwicz_func(mat, vec, tol=1e-12, maxits=500):
#         pass

#     return Solver(arrow_hurwicz_func)



# PRECONDITIONERS =========================================================


def make_block_diagonal_prec(system_matrix, momentum_block_size, mass_block_size, schur_complement_sign=1,
                             inverse_approximation_type='none', schur_complement_approximation_type='none',
                             inner_solver='spsolve'):
    """Makes a block-diagonal preconditioner (see page 61 of Benzi et al. (2005)) designed for saddle point problems.
    Because the full block-diagonal preconditioner requires full inversion of the momentum block, we use approximations
    of the inverse of the momentum block based on common matrix splittings (diagonal, upper/lower triangular). These
    approximations are used for the Schur complement as well. This preconditioner leaves the bottom rows for the internal
    boundary condition unaffected. 

    Arguments:

    - system_matrix (csr_matrix): system matrix of the linear system you want to solve,
    - momentum_block_size (int): dimension of the top-left momentum block (called A in Benzi et al. (2005))
    - mass_block_size (int): dimension of the bottom-right mass block
    - schur_complement_sign (int): either 1 or -1. The TU Delft CFD course uses positive Schur complement; Benzi et al. (2005) use negative.
    - inverse_approximation_type (str): either 'diagonal', 'upper_triangular', or 'lower_triangular'; used for upper left block of preconditioner
    - schur_complement_approximation_type (str): either 'diagonal', 'upper_triangular', or 'lower_triangular'; used for bottom right block of preconditioner.  

    """

    # check if an internal boundary condition is applied

    internal_BC = (system_matrix.shape[0] > momentum_block_size + mass_block_size)

    if internal_BC:
        A = slice_csr_matrix(system_matrix, list(range(momentum_block_size)) + list(range(momentum_block_size + mass_block_size, system_matrix.shape[0])),
                             list(range(momentum_block_size)) + list(range(momentum_block_size + mass_block_size, system_matrix.shape[0])))
        B1 = slice_csr_matrix(system_matrix, list(range(momentum_block_size)) + list(range(momentum_block_size + mass_block_size, system_matrix.shape[0])), 
                              range(momentum_block_size, momentum_block_size + mass_block_size))
        B2 = slice_csr_matrix(system_matrix, range(momentum_block_size, momentum_block_size + mass_block_size), 
                              list(range(momentum_block_size)) + list(range(momentum_block_size + mass_block_size, system_matrix.shape[0])))
        C = slice_csr_matrix(system_matrix, range(momentum_block_size, momentum_block_size + mass_block_size), range(momentum_block_size, momentum_block_size + mass_block_size))
    else:
        A = slice_csr_matrix(system_matrix, range(momentum_block_size), range(momentum_block_size))
        B1 = slice_csr_matrix(system_matrix, range(momentum_block_size), range(momentum_block_size, momentum_block_size + mass_block_size))
        B2 = slice_csr_matrix(system_matrix, range(momentum_block_size, momentum_block_size + mass_block_size), range(momentum_block_size))
        C = slice_csr_matrix(system_matrix, range(momentum_block_size, momentum_block_size + mass_block_size), range(momentum_block_size, momentum_block_size + mass_block_size))


    if schur_complement_approximation_type == 'diagonal':
        Ainv = sp.diags(np.power(A.diagonal(), -1))
        schur_complement = schur_complement_sign * (C - B2 @ Ainv @ B1)
    elif schur_complement_approximation_type == 'none':
        schur_complement = schur_complement_sign * (C - B2 @ splin.spsolve(A.tocsc(), B1.tocsc()))
    elif schur_complement_approximation_type == 'lower_triangular':
        schur_complement = schur_complement_sign * (C - B2 @ splin.spsolve(sp.tril(A, format='csc'), B1.tocsc()))
    elif schur_complement_approximation_type == 'upper_triangular':
        schur_complement = schur_complement_sign * (C - B2 @ splin.spsolve(sp.triu(A, format='csc'), B1.tocsc()))
    else:
        raise ValueError(f"{schur_complement_approximation_type} not implemented yet. Sorry!")

    def prec_step(vec):
        if internal_BC:
            reordered_return_vec = np.zeros_like(vec)

            reordered_rhs = np.zeros_like(vec)
            reordered_rhs[:momentum_block_size] = vec[:momentum_block_size]
            reordered_rhs[momentum_block_size:(system_matrix.shape[0] - mass_block_size)] = vec[(momentum_block_size+mass_block_size):]
            reordered_rhs[(system_matrix.shape[0] - mass_block_size):] = vec[momentum_block_size:(momentum_block_size+mass_block_size)]

            if inverse_approximation_type == 'diagonal':
                reordered_return_vec[:(system_matrix.shape[0] - mass_block_size)] = reordered_rhs[:(system_matrix.shape[0] - mass_block_size)] / A.diagonal()
            elif inverse_approximation_type == 'lower_triangular':
                reordered_return_vec[:(system_matrix.shape[0] - mass_block_size)] = splin.spsolve_triangular(sp.tril(A, format='csc'), reordered_rhs[:(system_matrix.shape[0] - mass_block_size)])
            elif inverse_approximation_type == 'upper_triangular':
                reordered_return_vec[:(system_matrix.shape[0] - mass_block_size)] = splin.spsolve_triangular(sp.triu(A, format='csc'), reordered_rhs[:(system_matrix.shape[0] - mass_block_size)], lower=False)
            elif inverse_approximation_type == 'none':
                reordered_return_vec[:(system_matrix.shape[0] - mass_block_size)] = splin.spsolve(A, reordered_rhs[:(system_matrix.shape[0] - mass_block_size)])

            reordered_return_vec[(system_matrix.shape[0] - mass_block_size):] = splin.spsolve(schur_complement, reordered_rhs[(system_matrix.shape[0] - mass_block_size):])

            return_vec = np.zeros_like(reordered_return_vec)
            return_vec[:momentum_block_size] = reordered_return_vec[:momentum_block_size]
            return_vec[momentum_block_size:(momentum_block_size+mass_block_size)] = reordered_return_vec[(system_matrix.shape[0] - mass_block_size):]
            return_vec[(momentum_block_size+mass_block_size):] = reordered_return_vec[momentum_block_size:(system_matrix.shape[0] - mass_block_size)]

        else:

            return_vec = np.zeros_like(vec)

            if inverse_approximation_type == 'diagonal':
                return_vec[:momentum_block_size] = vec[:momentum_block_size] / A.diagonal()
            elif inverse_approximation_type == 'lower_triangular':
                return_vec[:momentum_block_size] = splin.spsolve_triangular(sp.tril(A, format='csc'), vec[:momentum_block_size])
            elif inverse_approximation_type == 'upper_triangular':
                return_vec[:momentum_block_size] = splin.spsolve_triangular(sp.triu(A, format='csc'), vec[:momentum_block_size], lower=False)
            elif inverse_approximation_type == 'none':
                return_vec[:momentum_block_size] = splin.spsolve(A, vec[:momentum_block_size])
            
            return_vec[momentum_block_size:(momentum_block_size+mass_block_size)] = splin.spsolve(schur_complement,
                                                                                    vec[momentum_block_size:(momentum_block_size+mass_block_size)])
            
            return_vec[(momentum_block_size+mass_block_size):] = vec[(momentum_block_size+mass_block_size):]

        return return_vec

    block_diag_prec = Preconditioner(prec_step)
    return block_diag_prec      


def make_block_ltriangular_prec(system_matrix, momentum_block_size, mass_block_size, schur_complement_sign=1,
                                inverse_approximation_type='none', schur_complement_approximation_type='none',
                                inner_solver='spsolve'):
    """Makes a block-diagonal preconditioner (see page 61 of Benzi et al. (2005)) designed for saddle point problems.
    Because the full block-diagonal preconditioner requires full inversion of the momentum block, we use approximations
    of the inverse of the momentum block based on common matrix splittings (diagonal, upper/lower triangular). These
    approximations are used for the Schur complement as well. This preconditioner leaves the bottom rows for the internal
    boundary condition unaffected. 

    Arguments:

    - system_matrix (csr_matrix): system matrix of the linear system you want to solve,
    - momentum_block_size (int): dimension of the top-left momentum block (called A in Benzi et al. (2005))
    - mass_block_size (int): dimension of the bottom-right mass block
    - schur_complement_sign (int): either 1 or -1. The TU Delft CFD course uses positive Schur complement; Benzi et al. (2005) use negative.
    - inverse_approximation_type (str): either 'diagonal', 'upper_triangular', or 'lower_triangular'; used for upper left block of preconditioner
    - schur_complement_approximation_type (str): either 'diagonal', 'upper_triangular', or 'lower_triangular'; used for bottom right block of preconditioner.  

    """

    # check if an internal boundary condition is applied

    internal_BC = (system_matrix.shape[0] > momentum_block_size + mass_block_size)

    if internal_BC:
        A = slice_csr_matrix(system_matrix, list(range(momentum_block_size)) + list(range(momentum_block_size + mass_block_size, system_matrix.shape[0])),
                             list(range(momentum_block_size)) + list(range(momentum_block_size + mass_block_size, system_matrix.shape[0])))
        B1 = slice_csr_matrix(system_matrix, list(range(momentum_block_size)) + list(range(momentum_block_size + mass_block_size, system_matrix.shape[0])), 
                              range(momentum_block_size, momentum_block_size + mass_block_size))
        B2 = slice_csr_matrix(system_matrix, range(momentum_block_size, momentum_block_size + mass_block_size), 
                              list(range(momentum_block_size)) + list(range(momentum_block_size + mass_block_size, system_matrix.shape[0])))
        C = slice_csr_matrix(system_matrix, range(momentum_block_size, momentum_block_size + mass_block_size), range(momentum_block_size, momentum_block_size + mass_block_size))
    else:
        A = slice_csr_matrix(system_matrix, range(momentum_block_size), range(momentum_block_size))
        B1 = slice_csr_matrix(system_matrix, range(momentum_block_size), range(momentum_block_size, momentum_block_size + mass_block_size))
        B2 = slice_csr_matrix(system_matrix, range(momentum_block_size, momentum_block_size + mass_block_size), range(momentum_block_size))
        C = slice_csr_matrix(system_matrix, range(momentum_block_size, momentum_block_size + mass_block_size), range(momentum_block_size, momentum_block_size + mass_block_size))

    if schur_complement_approximation_type == 'diagonal':
        Ainv = sp.diags(np.power(A.diagonal(), -1))
        schur_complement = schur_complement_sign * (C - B2 @ Ainv @ B1)
    elif schur_complement_approximation_type == 'none':
        schur_complement = schur_complement_sign * (C - B2 @ splin.spsolve(A.tocsc(), B1.tocsc()))
    elif schur_complement_approximation_type == 'lower_triangular':
        schur_complement = schur_complement_sign * (C - B2 @ splin.spsolve(sp.tril(A, format='csc'), B1.tocsc()))
    elif schur_complement_approximation_type == 'upper_triangular':
        schur_complement = schur_complement_sign * (C - B2 @ splin.spsolve(sp.triu(A, format='csc'), B1.tocsc()))
    else:
        raise ValueError(f"{schur_complement_approximation_type} not implemented yet. Sorry!")

    def prec_step(vec):
        
        if internal_BC:
            reordered_return_vec = np.zeros_like(vec)
            reordered_rhs = np.zeros_like(vec)
            reordered_rhs[:momentum_block_size] = vec[:momentum_block_size]
            reordered_rhs[momentum_block_size:(system_matrix.shape[0] - mass_block_size)] = vec[(momentum_block_size+mass_block_size):]
            reordered_rhs[(system_matrix.shape[0] - mass_block_size):] = vec[momentum_block_size:(momentum_block_size+mass_block_size)]

            if inverse_approximation_type == 'diagonal':
                Ax = reordered_rhs[:(system_matrix.shape[0] - mass_block_size)] / A.diagonal()
            elif inverse_approximation_type == 'lower_triangular':
                Ax = splin.spsolve_triangular(sp.tril(A, format='csr'), reordered_rhs[:(system_matrix.shape[0] - mass_block_size)])
            elif inverse_approximation_type == 'upper_triangular':
                Ax = splin.spsolve_triangular(sp.triu(A, format='csr'), reordered_rhs[:(system_matrix.shape[0] - mass_block_size)], lower=False)
            elif inverse_approximation_type == 'none':
                Ax = splin.spsolve(A, reordered_rhs[:(system_matrix.shape[0] - mass_block_size)])

            reordered_return_vec[(system_matrix.shape[0] - mass_block_size):] = splin.spsolve(schur_complement, reordered_rhs[(system_matrix.shape[0] - mass_block_size):] - B2 @ Ax)

            return_vec = np.zeros_like(reordered_return_vec)
            return_vec[:momentum_block_size] = reordered_return_vec[:momentum_block_size]
            return_vec[momentum_block_size:(momentum_block_size+mass_block_size)] = reordered_return_vec[(system_matrix.shape[0] - mass_block_size):]
            return_vec[(momentum_block_size+mass_block_size):] = reordered_return_vec[momentum_block_size:(system_matrix.shape[0] - mass_block_size)]

        else:
            return_vec = np.zeros_like(vec)
            if inverse_approximation_type == 'diagonal':
                Ax = vec[:momentum_block_size] / A.diagonal()
            elif inverse_approximation_type == 'lower_triangular':
                Ax = splin.spsolve_triangular(sp.tril(A, format='csr'), vec[:momentum_block_size])
            elif inverse_approximation_type == 'upper_triangular':
                Ax = splin.spsolve_triangular(sp.triu(A, format='csr'), vec[:momentum_block_size], lower=False)
            elif inverse_approximation_type == 'none':
                Ax = splin.spsolve(A, vec[:momentum_block_size])

            return_vec[:momentum_block_size] = Ax[:]
            
            return_vec[momentum_block_size:(momentum_block_size+mass_block_size)] = splin.spsolve(schur_complement,
                                                                                    vec[momentum_block_size:(momentum_block_size+mass_block_size)] - B2 @ Ax)
            
            return_vec[(momentum_block_size+mass_block_size):] = vec[(momentum_block_size+mass_block_size):]

        return return_vec

    block_diag_prec = Preconditioner(prec_step)
    return block_diag_prec     


def make_block_utriangular_prec(system_matrix, momentum_block_size, mass_block_size, schur_complement_sign=1,
                                inverse_approximation_type='none', schur_complement_approximation_type='none',
                                inner_solver='spsolve'):
    """Makes a block-diagonal preconditioner (see page 61 of Benzi et al. (2005)) designed for saddle point problems.
    Because the full block-diagonal preconditioner requires full inversion of the momentum block, we use approximations
    of the inverse of the momentum block based on common matrix splittings (diagonal, upper/lower triangular). These
    approximations are used for the Schur complement as well. This preconditioner leaves the bottom rows for the internal
    boundary condition unaffected. 

    Arguments:

    - system_matrix (csr_matrix): system matrix of the linear system you want to solve,
    - momentum_block_size (int): dimension of the top-left momentum block (called A in Benzi et al. (2005))
    - mass_block_size (int): dimension of the bottom-right mass block
    - schur_complement_sign (int): either 1 or -1. The TU Delft CFD course uses positive Schur complement; Benzi et al. (2005) use negative.
    - inverse_approximation_type (str): either 'diagonal', 'upper_triangular', or 'lower_triangular'; used for upper left block of preconditioner
    - schur_complement_approximation_type (str): either 'diagonal', 'upper_triangular', or 'lower_triangular'; used for bottom right block of preconditioner.  

    """

    # check if an internal boundary condition is applied

    internal_BC = (system_matrix.shape[0] > momentum_block_size + mass_block_size)

    if internal_BC:
        A = slice_csr_matrix(system_matrix, list(range(momentum_block_size)) + list(range(momentum_block_size + mass_block_size, system_matrix.shape[0])),
                             list(range(momentum_block_size)) + list(range(momentum_block_size + mass_block_size, system_matrix.shape[0])))
        B1 = slice_csr_matrix(system_matrix, list(range(momentum_block_size)) + list(range(momentum_block_size + mass_block_size, system_matrix.shape[0])), 
                              range(momentum_block_size, momentum_block_size + mass_block_size))
        B2 = slice_csr_matrix(system_matrix, range(momentum_block_size, momentum_block_size + mass_block_size), 
                              list(range(momentum_block_size)) + list(range(momentum_block_size + mass_block_size, system_matrix.shape[0])))
        C = slice_csr_matrix(system_matrix, range(momentum_block_size, momentum_block_size + mass_block_size), range(momentum_block_size, momentum_block_size + mass_block_size))
    else:
        A = slice_csr_matrix(system_matrix, range(momentum_block_size), range(momentum_block_size))
        B1 = slice_csr_matrix(system_matrix, range(momentum_block_size), range(momentum_block_size, momentum_block_size + mass_block_size))
        B2 = slice_csr_matrix(system_matrix, range(momentum_block_size, momentum_block_size + mass_block_size), range(momentum_block_size))
        C = slice_csr_matrix(system_matrix, range(momentum_block_size, momentum_block_size + mass_block_size), range(momentum_block_size, momentum_block_size + mass_block_size))

    if schur_complement_approximation_type == 'diagonal':
        Ainv = sp.diags(np.power(A.diagonal(), -1))
        schur_complement = schur_complement_sign * (C - B2 @ Ainv @ B1)
    elif schur_complement_approximation_type == 'none':
        schur_complement = schur_complement_sign * (C - B2 @ splin.spsolve(A.tocsc(), B1.tocsc()))
    elif schur_complement_approximation_type == 'lower_triangular':
        schur_complement = schur_complement_sign * (C - B2 @ splin.spsolve(sp.tril(A, format='csc'), B1.tocsc()))
    elif schur_complement_approximation_type == 'upper_triangular':
        schur_complement = schur_complement_sign * (C - B2 @ splin.spsolve(sp.triu(A, format='csc'), B1.tocsc()))
    else:
        raise ValueError(f"{schur_complement_approximation_type} not implemented yet. Sorry!")

    def prec_step(vec):
        reordered_return_vec = np.zeros_like(vec)
        if internal_BC:
            reordered_rhs = np.zeros_like(vec)
            reordered_rhs[:momentum_block_size] = vec[:momentum_block_size]
            reordered_rhs[momentum_block_size:(system_matrix.shape[0] - mass_block_size)] = vec[(momentum_block_size+mass_block_size):]
            reordered_rhs[(system_matrix.shape[0] - mass_block_size):] = vec[momentum_block_size:(momentum_block_size+mass_block_size)]

            Sy = splin.spsolve(schur_complement, reordered_rhs[(system_matrix.shape[1] - mass_block_size):])
            reordered_return_vec[(system_matrix.shape[1] - mass_block_size):] = Sy[:]

            if inverse_approximation_type == 'diagonal':
                reordered_return_vec[:(system_matrix.shape[1] - mass_block_size)] = (reordered_rhs[:(system_matrix.shape[1] - mass_block_size)] - B1 @ Sy) / A.diagonal()
            elif inverse_approximation_type == 'lower_triangular':
                reordered_return_vec[:(system_matrix.shape[1] - mass_block_size)] = splin.spsolve_triangular(sp.tril(A, format='csr'), reordered_rhs[:(system_matrix.shape[1] - mass_block_size)] - B1 @ Sy)
            elif inverse_approximation_type == 'upper_triangular':
                reordered_return_vec[:(system_matrix.shape[1] - mass_block_size)] = splin.spsolve_triangular(sp.triu(A, format='csr'), reordered_rhs[:(system_matrix.shape[1] - mass_block_size)] - B1 @ Sy, lower=False)
            elif inverse_approximation_type == 'none':
                reordered_return_vec[:(system_matrix.shape[1] - mass_block_size)] = splin.spsolve(A, reordered_rhs[:(system_matrix.shape[1] - mass_block_size)] - B1 @ Sy)

            return_vec = np.zeros_like(reordered_return_vec)
            return_vec[:momentum_block_size] = reordered_return_vec[:momentum_block_size]
            return_vec[momentum_block_size:(momentum_block_size+mass_block_size)] = reordered_return_vec[(system_matrix.shape[0] - mass_block_size):]
            return_vec[(momentum_block_size+mass_block_size):] = reordered_return_vec[momentum_block_size:(system_matrix.shape[0] - mass_block_size)]
        else:
            return_vec = np.zeros_like(vec)
            Sy = splin.spsolve(schur_complement, vec[momentum_block_size:(momentum_block_size+mass_block_size)])
            return_vec[momentum_block_size:(momentum_block_size+mass_block_size)] = Sy[:]

            if inverse_approximation_type == 'diagonal':
                return_vec[:momentum_block_size] = (vec[:momentum_block_size] - B1 @ Sy) / A.diagonal()
            elif inverse_approximation_type == 'lower_triangular':
                return_vec[:momentum_block_size] = splin.spsolve_triangular(sp.tril(A, format='csr'), vec[:momentum_block_size] - B1 @ Sy)
            elif inverse_approximation_type == 'upper_triangular':
                return_vec[:momentum_block_size] = splin.spsolve_triangular(sp.triu(A, format='csr'), vec[:momentum_block_size] - B1 @ Sy, lower=False)
            elif inverse_approximation_type == 'none':
                return_vec[:momentum_block_size] = splin.spsolve(A, vec[:momentum_block_size] - B1 @ Sy)

            return_vec[(momentum_block_size+mass_block_size):] = vec[(momentum_block_size+mass_block_size):]
            
        return return_vec

    block_diag_prec = Preconditioner(prec_step)
    return block_diag_prec  

