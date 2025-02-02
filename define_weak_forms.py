import ngsolve
import ngsolve.solvers
import numpy as np
import matplotlib.pyplot as plt

from NiFlow.truncationbasis.truncationbasis import TruncationBasis
from NiFlow.utils import *


def ngsolve_tanh(argument):
    return ngsolve.sinh(argument) / ngsolve.cosh(argument)
        

def add_weak_form(a: ngsolve.BilinearForm, model_options: dict, numerical_information: dict, geometric_information: dict, constant_parameters: dict, spatial_parameters: dict, 
                  alpha_trialfunctions: dict, beta_trialfunctions: dict, gamma_trialfunctions: dict,
                  umom_testfunctions: dict, vmom_testfunctions: dict, DIC_testfunctions: dict,
                  vertical_basis: TruncationBasis, time_basis: TruncationBasis, normalalpha: list, normalalpha_y = None, only_linear = False, A_trialfunctions=None,
                  sea_boundary_testfunctions=None, Q_trialfunctions=None, river_boundary_testfunctions=None):
    """
    Constructs a weak form of the model equations (as an ngsolve.BilinearForm-object) by adding it to an empty one.

    Arguments:
    
        - a (ngsolve.BilinearForm):                     bilinear form object on which the weak form is constructed;
        - model_options (dict):                         model options of the hydrodynamics object;
        - numerical_information (dict):                 numerical information dictionary of the hydrodynamics object;
        - geometric_information (dict):                 geometric information dictionary of the hydrodynamics object;
        - constant_parameters (dict):                   dictionary of constant physical parameters associated with the model;
        - spatial_parameters (dict):                    dictionary of spatial physical parameters (as SpatialParameter objects) associated with the model;
        - alpha_trialfunctions (dict):                  dictionary of trial functions representing the alpha-Fourier/eigenfunction coefficients (obtainable via hydrodynamics._setup_TnT)
        - beta_trialfunctions (dict):                   dictionary of trial functions representing the beta-Fourier/eigenfunction coefficients;
        - gamma_trialfunctions (dict):                  dictionary of trial functions representing the gamma-Fourier/eigenfunction coefficients;
        - umom_testfunctions (dict):                    dictionary of test functions for the along-channel momentum equations;
        - vmom_testfunctions (dict):                    dictionary of test functions for the lateral momentum equations;
        - DIC_testfunctions (dict):                     dictionary of test functions for the depth-integrated continuity equation;
        - vertical_basis (TruncationBasis):             vertical eigenfunction basis;
        - time_basis (TruncationBasis):                 temporal Fourier basis;
        - normalalpha (list):                           list of coefficient functions containing the lateral structure of the riverine along-channel flow for that boundary condition;
        - normalalpha_y (list):                         analytical y-derivatives (eta) for the normalalpha-functions;
        - only_linear (bool):                           flag indicating whether only the linear part of the weak form should be constructed (used in the Newton scheme);
        - A_trialfunctions (dict):                      dictionary of trial functions representing the computational seaward boundary condition of gamma (free surface);
        - sea_boundary_testfunctions (dict):            dictionary of test functions for the interpretable seaward boundary condition;
        - Q_trialfunctions (dict):                      dictionary of trial functions representing the computational riverine boundary condition for alpha (along-channel velocity);
        - river_boundary_testfunctions (dict):          dictionary of test functions for the interpretable riverine boundary condition.

    """

    # Defining shorthands of variables

    H = spatial_parameters['H'].cf
    Hx = spatial_parameters["H"].gradient_cf[0]
    Hy = spatial_parameters['H'].gradient_cf[1]
    rho = spatial_parameters['rho'].cf
    rhox = spatial_parameters['rho'].gradient_cf[0]
    rhoy = spatial_parameters['rho'].gradient_cf[1]
    R = spatial_parameters['R'].cf
    Rx = spatial_parameters['R'].gradient_cf[0]
    Ry = spatial_parameters['R'].gradient_cf[1]

    f = constant_parameters['f']
    g = constant_parameters['g']
    Av = constant_parameters['Av']
    Ah = constant_parameters['Ah']
    sigma = constant_parameters['sigma']
    Q = constant_parameters['discharge']

    G1 = vertical_basis.tensor_dict['G1']
    G2 = vertical_basis.tensor_dict['G2']
    G3 = vertical_basis.tensor_dict['G3']
    G4 = vertical_basis.tensor_dict['G4']
    G6 = vertical_basis.tensor_dict['G5']

    vertical_innerproduct = vertical_basis.inner_product

    H3 = time_basis.tensor_dict['H3']
    H3_iszero = time_basis.tensor_dict['H3_iszero']

    x_scaling = geometric_information['x_scaling']
    y_scaling = geometric_information['y_scaling']

    advection_matrix = model_options['advection_influence_matrix']
    advection_epsilon = constant_parameters['advection_epsilon']

    M = numerical_information['M']
    imax = numerical_information['imax']
    
    seaward_amplitudes = constant_parameters['seaward_amplitudes']
    seaward_phases = constant_parameters['seaward_phases']

    # Construct the non-linear ramp
    if geometric_information['L_R_sea'] > 1e-16:
        ramp_sea = ngsolve.IfPos(
            -geometric_information['L_R_sea']/x_scaling - ngsolve.x,
            ngsolve.IfPos(
                -geometric_information['L_R_sea']/x_scaling - geometric_information['L_RA_sea']/x_scaling - ngsolve.x,
                0,
                0.5 * (1 + ngsolve_tanh((4 * (ngsolve.x + geometric_information['L_RA_sea']/x_scaling + 0.5 * geometric_information['L_R_sea']/x_scaling) / (geometric_information['L_R_sea']/x_scaling)) / 
                                        (1 - (2*(ngsolve.x + geometric_information['L_RA_sea']/x_scaling + 0.5*geometric_information['L_R_sea']/x_scaling) / (geometric_information['L_R_sea']/x_scaling))**2)))
            ),
            1
        )
    else:
        ramp_sea = ngsolve.CF(1)

    if geometric_information['L_R_river'] > 1e-16:
        ramp_river = ngsolve.IfPos(
                    -(geometric_information['riverine_boundary_x'] + geometric_information['L_RA_river']) / x_scaling + ngsolve.x,
                    ngsolve.IfPos(
                        -(geometric_information['riverine_boundary_x'] + geometric_information['L_R_river'] + geometric_information['L_RA_river']) / x_scaling + ngsolve.x,
                        0,
                        0.5 * (1 + ngsolve_tanh((-4 * (ngsolve.x - geometric_information['L_RA_river'] / x_scaling - 0.5 * geometric_information["L_R_river"] / x_scaling - geometric_information['riverine_boundary_x'] / x_scaling) / 
                                                (geometric_information['L_R_river'] / x_scaling)) / (1 - (2*(ngsolve.x- geometric_information['L_RA_river']/x_scaling - 0.5 * geometric_information["L_R_river"]/x_scaling - geometric_information['riverine_boundary_x']/x_scaling)/(geometric_information["L_R_river"]/x_scaling))**2)))
                    ),
                    1
                )
    else:
        ramp_river = ngsolve.CF(1)

    ramp = ramp_sea * ramp_river    

    # make the linear interpolant functions to treat computational boundary conditions
    if model_options['sea_boundary_treatment'] == 'exact':
        sea_interpolant = ((geometric_information['riverine_boundary_x'] / x_scaling) + (geometric_information['L_BL_river']/x_scaling) + (geometric_information['L_R_river']/x_scaling) + \
                           (geometric_information['L_RA_river']/x_scaling) - ngsolve.x) / \
                           ((geometric_information['riverine_boundary_x']+geometric_information['L_BL_river']+geometric_information['L_R_river']+geometric_information['L_RA_river'] +
                            geometric_information['L_BL_sea'] + geometric_information['L_R_sea'] + geometric_information['L_RA_sea']) / x_scaling)
        sea_interpolant_x = -1 / ((geometric_information['riverine_boundary_x']+geometric_information['L_BL_river']+geometric_information['L_R_river']+geometric_information['L_RA_river'] +
                            geometric_information['L_BL_sea'] + geometric_information['L_R_sea'] + geometric_information['L_RA_sea']) / x_scaling)
    if model_options['river_boundary_treatment'] == 'exact':
        river_interpolant = (-(geometric_information['L_BL_sea']/x_scaling) - (geometric_information['L_R_sea']/x_scaling) - \
                             (geometric_information['L_RA_sea']/x_scaling) + ngsolve.x) / \
                             ((geometric_information['riverine_boundary_x']+geometric_information['L_BL_river']+geometric_information['L_R_river']+geometric_information['L_RA_river'] +
                             geometric_information['L_BL_sea'] + geometric_information['L_R_sea'] + geometric_information['L_RA_sea']) / x_scaling)
        river_interpolant_x = 1 / ((geometric_information['riverine_boundary_x']+geometric_information['L_BL_river']+geometric_information['L_R_river']+geometric_information['L_RA_river'] +
                             geometric_information['L_BL_sea'] + geometric_information['L_R_sea'] + geometric_information['L_RA_sea']) / x_scaling)


    # 1: Depth-integrated continuity equation ====================================================================================================================
    # See project notes for an analytical expression of these weak forms
    a += sum([0.5 * G4(m) * (Hx + Rx) * alpha_trialfunctions[m][0] * DIC_testfunctions[0] / x_scaling * ngsolve.dx for m in range(M)])
    a += sum([0.5 * G4(m) * (Hy + Ry) * beta_trialfunctions[m][0] * DIC_testfunctions[0] / y_scaling * ngsolve.dx for m in range(M)])
    a += sum([0.5 * G4(m) * (H + R) * DIC_testfunctions[0] * (ngsolve.grad(alpha_trialfunctions[m][0])[0] / x_scaling + ngsolve.grad(beta_trialfunctions[m][0])[1] / y_scaling) * ngsolve.dx for m in range(M)])
    if model_options['river_boundary_treatment'] == 'exact': # if we handle the riverine boundary condition, add correction terms
        a += sum([0.5 * G4(m) * Q_trialfunctions[0] * normalalpha[m] * river_interpolant * (Hx + Rx) * DIC_testfunctions[0] / x_scaling * ngsolve.dx for m in range(M)])
        a += sum([0.5 * G4(m) * Q_trialfunctions[0] * normalalpha[m] * river_interpolant_x * (H + R) * DIC_testfunctions[0] / x_scaling * ngsolve.dx for m in range(M)])

    # terms l != 0
    for l in range(1, imax + 1):
        a += sigma * np.pi * l * DIC_testfunctions[-l] * gamma_trialfunctions[l] * ngsolve.dx
        a += sigma * np.pi * -l * DIC_testfunctions[l] * gamma_trialfunctions[-l] * ngsolve.dx
        if model_options['sea_boundary_treatment'] == 'exact':
            a += sigma * np.pi * l * A_trialfunctions[l] * sea_interpolant * DIC_testfunctions[-l] * ngsolve.dx
            a += sigma * np.pi * -l * A_trialfunctions[-l] * sea_interpolant * DIC_testfunctions[l] * ngsolve.dx

        a += sum([0.5 * G4(m) * (Hx + Rx) * alpha_trialfunctions[m][-l] * DIC_testfunctions[-l] / x_scaling * ngsolve.dx for m in range(M)])
        a += sum([0.5 * G4(m) * (Hy + Ry) * beta_trialfunctions[m][-l] * DIC_testfunctions[-l] / y_scaling * ngsolve.dx for m in range(M)])
        a += sum([0.5 * G4(m) * (H + R) * DIC_testfunctions[-l] * (ngsolve.grad(alpha_trialfunctions[m][-l])[0] / x_scaling + ngsolve.grad(beta_trialfunctions[m][-l])[1] / y_scaling) * ngsolve.dx for m in range(M)])
        
        a += sum([0.5 * G4(m) * (Hx + Rx) * alpha_trialfunctions[m][l] * DIC_testfunctions[l] / x_scaling * ngsolve.dx for m in range(M)])
        a += sum([0.5 * G4(m) * (Hy + Ry) * beta_trialfunctions[m][l] * DIC_testfunctions[l] / y_scaling * ngsolve.dx for m in range(M)])
        a += sum([0.5 * G4(m) * (H + R) * DIC_testfunctions[l] * (ngsolve.grad(alpha_trialfunctions[m][l])[0] / x_scaling + ngsolve.grad(beta_trialfunctions[m][l])[1] / y_scaling) * ngsolve.dx for m in range(M)])
        if model_options['river_boundary_treatment'] == 'exact':
            a += sum([0.5 * G4(m) * Q_trialfunctions[-l] * normalalpha[m] * river_interpolant * (Hx + Rx) * DIC_testfunctions[-l] / x_scaling * ngsolve.dx for m in range(M)])
            a += sum([0.5 * G4(m) * Q_trialfunctions[-l] * normalalpha[m] * river_interpolant_x * (H+R) * DIC_testfunctions[-l] / x_scaling * ngsolve.dx for m in range(M)])

            a += sum([0.5 * G4(m) * Q_trialfunctions[l] * normalalpha[m] * river_interpolant * (Hx + Rx) * DIC_testfunctions[l] / x_scaling * ngsolve.dx for m in range(M)])
            a += sum([0.5 * G4(m) * Q_trialfunctions[l] * normalalpha[m] * river_interpolant_x * (H+R) * DIC_testfunctions[l] / x_scaling * ngsolve.dx for m in range(M)])
            
    # INTERPRETABLE SEAWARD BOUNDARY CONDITION ======================================================================================================

    dirac_delta_width = 0.05

    if model_options['sea_boundary_treatment'] == 'exact': # linear equation for residual waterlevel
        dirac_delta_sea = ngsolve.IfPos((ngsolve.x - dirac_delta_width/2) * (-dirac_delta_width/2 - ngsolve.x),
                                        (4/(dirac_delta_width**2)) * (dirac_delta_width / 2 - ngsolve.sqrt((ngsolve.x)**2)),
                                        0) # Hat function Dirac Delta
        a += (gamma_trialfunctions[0]+A_trialfunctions[0]*sea_interpolant) * sea_boundary_testfunctions[0] * dirac_delta_sea * ngsolve.dx - seaward_amplitudes[0] * dirac_delta_sea * sea_boundary_testfunctions[0] * ngsolve.dx # don't use absolute value here, but the actual water level

    if not only_linear:        
        if model_options['sea_boundary_treatment'] == 'exact':
            # terms l != 0
            for l in range(1, imax + 1):
                a += ngsolve.sqrt((gamma_trialfunctions[-l]+A_trialfunctions[-l]*sea_interpolant)**2 + (gamma_trialfunctions[l]+A_trialfunctions[l]*sea_interpolant)**2) * sea_boundary_testfunctions[-l] * dirac_delta_sea * ngsolve.dx  - seaward_amplitudes[l] * dirac_delta_sea * sea_boundary_testfunctions[-l] * ngsolve.dx# amplitude condition
                a += ngsolve.atan2(-(gamma_trialfunctions[-l]+A_trialfunctions[-l]*sea_interpolant), (gamma_trialfunctions[l]+A_trialfunctions[l]*sea_interpolant)) * sea_boundary_testfunctions[l] * dirac_delta_sea * ngsolve.dx - seaward_phases[l - 1] * dirac_delta_sea * sea_boundary_testfunctions[l] * ngsolve.dx # phase condition, phases at l - 1 because there is no element corresponding to l = 0 in that list

    # INTERPRETABLE RIVERINE BOUNDARY CONDITION =====================================================================================================

    # dirac_delta_river = ngsolve.IfPos((ngsolve.x - dirac_delta_width/2 - (1 - geometric_information['L_R_river'])) * ((1 - geometric_information['L_R_river']) - dirac_delta_width/2 - ngsolve.x),
    #                                   1 / dirac_delta_width,
    #                                   0) # Step function Dirac Delta
    
    dirac_delta_river = ngsolve.IfPos((ngsolve.x - dirac_delta_width/2 - (geometric_information['riverine_boundary_x']/x_scaling)) * ((geometric_information['riverine_boundary_x']/x_scaling) - dirac_delta_width/2 - ngsolve.x),
                                        (4/(dirac_delta_width**2)) * (dirac_delta_width / 2 - ngsolve.sqrt((ngsolve.x-(geometric_information['riverine_boundary_x']/x_scaling))**2)),
                                        0) # Hat function Dirac Delta
    
    if model_options['river_boundary_treatment'] == 'exact':
        # term l = 0
        a += sum([y_scaling * (H+R) * dirac_delta_river * G4(m) * (alpha_trialfunctions[m][0] + Q_trialfunctions[0] * normalalpha[m] * river_interpolant) * river_boundary_testfunctions[0] * ngsolve.dx for m in range(M)]) + Q * river_boundary_testfunctions[0] * dirac_delta_river * ngsolve.dx
        # terms l != 0
        for l in range(1, imax + 1):
            a += sum([(H+R) * dirac_delta_river * G4(m) * (alpha_trialfunctions[m][-l] + Q_trialfunctions[-l] * normalalpha[m] * river_interpolant) * river_boundary_testfunctions[-l] * ngsolve.dx for m in range(M)])
            a += sum([(H+R) * dirac_delta_river * G4(m) * (alpha_trialfunctions[m][l] + Q_trialfunctions[l] * normalalpha[m] * river_interpolant) * river_boundary_testfunctions[l] * ngsolve.dx for m in range(M)])

        
    # 2: Momentum equations =========================================================================================================================
    # For analytical forms of these weak forms, see Project Notes
    for p in range(M): # loop through all vertical components
        # term l = 0
        if not only_linear:
            for i in range(-imax, imax + 1):
                for j in range(-imax, imax + 1):
                    if H3_iszero(i,j,0):
                        continue
                    else:
                        if advection_matrix[0, abs(i)] and advection_matrix[0, abs(j)]:
                            # Along-channel advection
                            a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,0) * G1(m,n,p) * umom_testfunctions[p][0] * alpha_trialfunctions[m][i] * ngsolve.grad(alpha_trialfunctions[n][j])[0] / x_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                            a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,0) * G1(m,n,p) * vmom_testfunctions[p][0] * alpha_trialfunctions[m][i] * ngsolve.grad(beta_trialfunctions[n][j])[0] / x_scaling *ngsolve.dx for n in range(M)]) for m in range(M)])
                            # Lateral advection
                            a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,0) * G1(m,n,p) * umom_testfunctions[p][0] * beta_trialfunctions[m][i] * ngsolve.grad(alpha_trialfunctions[n][j])[1] / y_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                            a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,0) * G1(m,n,p) * vmom_testfunctions[p][0] * beta_trialfunctions[m][i] * ngsolve.grad(beta_trialfunctions[n][j])[1] / y_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                            # Vertical advection
                            a += sum([sum([advection_epsilon * ramp * H3(i,j,0) * G2(m,n,p) * umom_testfunctions[p][0] * alpha_trialfunctions[m][i] * (
                                (Hx + Rx) * alpha_trialfunctions[n][j] / x_scaling + (H+R) * ngsolve.grad(alpha_trialfunctions[n][j])[0] / x_scaling + \
                                (Hy + Ry) * beta_trialfunctions[n][j] / y_scaling + (H+R) * ngsolve.grad(beta_trialfunctions[n][j])[1] / y_scaling
                            ) * ngsolve.dx for n in range(M)]) for m in range(M)])
                            a += sum([sum([advection_epsilon * ramp * H3(i,j,0) * G2(m,n,p) * vmom_testfunctions[p][0] * beta_trialfunctions[m][i] * (
                                (Hx + Rx) * alpha_trialfunctions[n][j] / x_scaling + (H+R) * ngsolve.grad(alpha_trialfunctions[n][j])[0] / x_scaling + \
                                (Hy + Ry) * beta_trialfunctions[n][j] / y_scaling + (H+R) * ngsolve.grad(beta_trialfunctions[n][j])[1] / y_scaling
                            ) * ngsolve.dx for n in range(M)]) for m in range(M)])

                            if model_options['river_boundary_treatment'] == 'exact':
                                # Along-channel advection
                                a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,0) * G1(m,n,p) * umom_testfunctions[p][0] * Q_trialfunctions[i] * normalalpha[m] * river_interpolant * ngsolve.grad(alpha_trialfunctions[n][j])[0] / x_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                                a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,0) * G1(m,n,p) * umom_testfunctions[p][0] * alpha_trialfunctions[m][i] *  Q_trialfunctions[j] * normalalpha[n] * river_interpolant_x / x_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                                a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,0) * G1(m,n,p) * umom_testfunctions[p][0] * Q_trialfunctions[i] * normalalpha[m] * river_interpolant * Q_trialfunctions[j] * normalalpha[n] / x_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])

                                a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,0) * G1(m,n,p) * vmom_testfunctions[p][0] * Q_trialfunctions[i] * normalalpha[m] * river_interpolant * ngsolve.grad(beta_trialfunctions[n][j])[0] / x_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                                # Lateral advection
                                a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,0) * G1(m,n,p) * umom_testfunctions[p][0] * beta_trialfunctions[m][i] * Q_trialfunctions[j] * normalalpha_y[n] * river_interpolant / y_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                                # Vertical advection
                                a += sum([sum([advection_epsilon * ramp * H3(i,j,0) * G2(m,n,p) * umom_testfunctions[p][0] * Q_trialfunctions[i] * normalalpha[m] * river_interpolant * (H+R) * ngsolve.grad(alpha_trialfunctions[n][j])[0] / x_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                                a += sum([sum([advection_epsilon * ramp * H3(i,j,0) * G2(m,n,p) * umom_testfunctions[p][0] * Q_trialfunctions[i] * normalalpha[m] * river_interpolant * (Hx+Rx) * alpha_trialfunctions[n][j] / x_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                                a += sum([sum([advection_epsilon * ramp * H3(i,j,0) * G2(m,n,p) * umom_testfunctions[p][0] * Q_trialfunctions[i] * normalalpha[m] * river_interpolant * (H+R) * Q_trialfunctions[j] * normalalpha[n] / x_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                                a += sum([sum([advection_epsilon * ramp * H3(i,j,0) * G2(m,n,p) * umom_testfunctions[p][0] * Q_trialfunctions[i] * normalalpha[m] * river_interpolant * (Hx+Rx) * Q_trialfunctions[j] * normalalpha[n] * ngsolve.x / x_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                                a += sum([sum([advection_epsilon * ramp * H3(i,j,0) * G2(m,n,p) * umom_testfunctions[p][0] * alpha_trialfunctions[m][i] * (H+R) * Q_trialfunctions[j] * normalalpha[n] * river_interpolant_x / x_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                                a += sum([sum([advection_epsilon * ramp * H3(i,j,0) * G2(m,n,p) * umom_testfunctions[p][0] * alpha_trialfunctions[m][i] * (Hx+Rx) * Q_trialfunctions[j] * normalalpha[n] * river_interpolant / x_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])

                                a += sum([sum([advection_epsilon * ramp * H3(i,j,0) * G2(m,n,p) * vmom_testfunctions[p][0] * beta_trialfunctions[m][i] * (H+R) * Q_trialfunctions[j] * normalalpha[n] * river_interpolant_x / x_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                                a += sum([sum([advection_epsilon * ramp * H3(i,j,0) * G2(m,n,p) * vmom_testfunctions[p][0] * beta_trialfunctions[m][i] * (Hx+Rx) * Q_trialfunctions[j] * normalalpha[n] * river_interpolant / x_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])

        
        # Coriolis
        a += -0.5 * vertical_innerproduct(p, p) * f * (H+R) * beta_trialfunctions[p][0] * umom_testfunctions[p][0] * ngsolve.dx
        a += 0.5 * vertical_innerproduct(p, p) * f * (H+R) * alpha_trialfunctions[p][0] * vmom_testfunctions[p][0] * ngsolve.dx
        if model_options['river_boundary_treatment'] == 'exact':
            a += 0.5 * vertical_innerproduct(p, p) * f * (H+R) * Q_trialfunctions[0] * normalalpha[p] * river_interpolant * vmom_testfunctions[p][0] * ngsolve.dx
        # Barotropic pressure gradient
        a += 0.5 * (H+R) * g * G4(p) * ngsolve.grad(gamma_trialfunctions[0])[0] * umom_testfunctions[p][0] / x_scaling * ngsolve.dx
        a += 0.5 * (H+R) * g * G4(p) * ngsolve.grad(gamma_trialfunctions[0])[1] * vmom_testfunctions[p][0] / y_scaling * ngsolve.dx
        if model_options['sea_boundary_treatment'] == 'exact':
            a += 0.5 * (H+R) * g * G4(p) * A_trialfunctions[0] * sea_interpolant_x * umom_testfunctions[p][0] / x_scaling * ngsolve.dx
        # Baroclinic pressure gradient
        a += (1/x_scaling) * 0.5 * np.sqrt(2) * G6(p) * (H+R) * (H+R) * umom_testfunctions[p][0] * rhox / rho * ngsolve.dx # assumes density is depth-independent
        a += (1/y_scaling) * 0.5 * np.sqrt(2) * G6(p) * (H+R) * (H+R) * vmom_testfunctions[p][0] * rhoy / rho * ngsolve.dx
        # Vertical eddy viscosity
        if model_options['veddy_viscosity_assumption'] == 'constant':
            a += -0.5 * Av * G3(p, p) * alpha_trialfunctions[p][0] * umom_testfunctions[p][0] / (H+R) * ngsolve.dx# assumes that vertical basis consists of eigenfunctions of the vertical mixing operator
            a += -0.5 * Av * G3(p, p) * beta_trialfunctions[p][0] * vmom_testfunctions[p][0] / (H+R) * ngsolve.dx
            if model_options['river_boundary_treatment'] == 'exact':
                a += -0.5 * Av * G3(p, p) * Q_trialfunctions[0] * normalalpha[p] * river_interpolant * umom_testfunctions[p][0] / (H+R) * ngsolve.dx
        elif model_options['veddy_viscosity_assumption'] == 'depth-scaled&constantprofile': 
            a += -0.5 * Av * G3(p, p) * alpha_trialfunctions[p][0] * umom_testfunctions[p][0] *ngsolve.dx# assumes that vertical basis consists of eigenfunctions of the vertical mixing operator
            a += -0.5 * Av * G3(p, p) * beta_trialfunctions[p][0] * vmom_testfunctions[p][0] * ngsolve.dx
            if model_options['river_boundary_treatment'] == 'exact':
                a += -0.5 * Av * G3(p, p) * Q_trialfunctions[0] * normalalpha[p] * river_interpolant * umom_testfunctions[p][0] * ngsolve.dx
        # Horizontal eddy viscosity
        a += 0.5 * vertical_innerproduct(p, p) * Ah * (
            ngsolve.grad(alpha_trialfunctions[p][0])[0] * ((H+R)*ngsolve.grad(umom_testfunctions[p][0])[0] / (x_scaling**2) + umom_testfunctions[p][0]*(Hx+Rx) / (x_scaling**2)) + \
            ngsolve.grad(alpha_trialfunctions[p][0])[1] * ((H+R)*ngsolve.grad(umom_testfunctions[p][0])[1] / (y_scaling**2) + umom_testfunctions[p][0]*(Hy+Ry) / (y_scaling**2))
        ) * ngsolve.dx
        a += 0.5 * vertical_innerproduct(p, p) * Ah * (
            ngsolve.grad(beta_trialfunctions[p][0])[0] * ((H+R)*ngsolve.grad(vmom_testfunctions[p][0])[0] / (x_scaling**2) + vmom_testfunctions[p][0]*(Hx+Rx) / (x_scaling**2)) + \
            ngsolve.grad(beta_trialfunctions[p][0])[1] * ((H+R)*ngsolve.grad(vmom_testfunctions[p][0])[1] / (y_scaling**2) + vmom_testfunctions[p][0]*(Hy+Ry) / (y_scaling**2))
        ) * ngsolve.dx
        if model_options['river_boundary_treatment'] == 'exact':
            a += 0.5 * vertical_innerproduct(p, p) * Ah * (
                Q_trialfunctions[0] * normalalpha[p] * river_interpolant_x * ((H+R)*ngsolve.grad(umom_testfunctions[p][0])[0] / (x_scaling**2) + umom_testfunctions[p][0]*(Hx+Rx) / (x_scaling**2)) + \
                Q_trialfunctions[0] * normalalpha_y[p] * river_interpolant * ((H+R)*ngsolve.grad(umom_testfunctions[p][0])[1] / (y_scaling**2) + umom_testfunctions[p][0]*(Hy+Ry) / (y_scaling**2))
            ) * ngsolve.dx

        # Terms l != 0
        for l in range(1, imax + 1):
            # Local acceleration
            a += vertical_innerproduct(p, p) * np.pi * l * (H+R) * sigma * umom_testfunctions[p][-l] * alpha_trialfunctions[p][l] * ngsolve.dx# factor 0.5 from vertical projection coefficient
            a += vertical_innerproduct(p, p) * np.pi * -l * (H+R) * sigma * umom_testfunctions[p][l] * alpha_trialfunctions[p][-l] * ngsolve.dx
            a += vertical_innerproduct(p, p) * np.pi * l *  (H+R) * sigma * vmom_testfunctions[p][-l] * beta_trialfunctions[p][l] * ngsolve.dx # factor 0.5 from vertical projection coefficient
            a += vertical_innerproduct(p, p) * np.pi * -l *  (H+R) * sigma * vmom_testfunctions[p][l] * beta_trialfunctions[p][-l] * ngsolve.dx
            if model_options['river_boundary_treatment'] == 'exact':
                a += vertical_innerproduct(p, p) * np.pi * l * (H+R) * sigma * umom_testfunctions[p][-l] * Q_trialfunctions[l] * normalalpha[p] * river_interpolant * ngsolve.dx
                a += vertical_innerproduct(p, p) * np.pi * -l * (H+R) * sigma * umom_testfunctions[p][l] * Q_trialfunctions[-l] * normalalpha[p] * river_interpolant * ngsolve.dx

            if not only_linear:
                for i in range(-imax, imax + 1):
                    for j in range(-imax, imax + 1):
                        if H3_iszero(i,j,-l) and H3_iszero(i,j,l):
                            continue 
                        else:
                            if advection_matrix[l, abs(i)] and advection_matrix[l, abs(j)]:
                                # Along-channel advection
                                a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,-l) * umom_testfunctions[p][-l] * G1(m,n,p) * alpha_trialfunctions[m][i] * ngsolve.grad(alpha_trialfunctions[n][j])[0] / x_scaling * ngsolve.dx for n in range(0, M)]) for m in range(0, M)])
                                a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,-l) * vmom_testfunctions[p][-l] * G1(m,n,p) * alpha_trialfunctions[m][i] * ngsolve.grad(beta_trialfunctions[n][j])[0] / x_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                                # Lateral advection
                                a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,-l) * umom_testfunctions[p][-l] * G1(m,n,p) * beta_trialfunctions[m][i] * ngsolve.grad(alpha_trialfunctions[n][j])[1] / y_scaling * ngsolve.dx for n in range(0, M)]) for m in range(0, M)])
                                a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,-l) * vmom_testfunctions[p][-l] * G1(m,n,p) * beta_trialfunctions[m][i] * ngsolve.grad(beta_trialfunctions[n][j])[1] / y_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                                # Vertical advection
                                a += sum([sum([advection_epsilon * ramp * H3(i,j,-l) * G2(m,n,p) * umom_testfunctions[p][-l] * alpha_trialfunctions[m][i] * (
                                    (H+R) * ngsolve.grad(alpha_trialfunctions[n][j])[0] / x_scaling + (Hx+Rx) * alpha_trialfunctions[n][j] / x_scaling + \
                                    (H+R) * ngsolve.grad(beta_trialfunctions[n][j])[1] / y_scaling + (Hy+Ry) * beta_trialfunctions[n][j] / y_scaling
                                ) * ngsolve.dx for n in range(M)]) for m in range(M)])
                                a += sum([sum([advection_epsilon * ramp * H3(i,j,-l) * G2(m,n,p) * vmom_testfunctions[p][-l] * beta_trialfunctions[m][i] * (
                                    (H+R) * ngsolve.grad(alpha_trialfunctions[n][j])[0] / x_scaling + (Hx+Rx) * alpha_trialfunctions[n][j] / x_scaling + \
                                    (H+R) * ngsolve.grad(beta_trialfunctions[n][j])[1] / y_scaling + (Hy+Ry) * beta_trialfunctions[n][j] / y_scaling
                                ) * ngsolve.dx for n in range(M)]) for m in range(M)])

                                # Along-channel advection
                                a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,l) * umom_testfunctions[p][l] * G1(m,n,p) * alpha_trialfunctions[m][i] * ngsolve.grad(alpha_trialfunctions[n][j])[0] / x_scaling * ngsolve.dx for n in range(0, M)]) for m in range(0, M)])
                                a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,l) * vmom_testfunctions[p][l] * G1(m,n,p) * alpha_trialfunctions[m][i] * ngsolve.grad(beta_trialfunctions[n][j])[0] / x_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                                # Lateral advection
                                a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,l) * umom_testfunctions[p][l] * G1(m,n,p) * beta_trialfunctions[m][i] * ngsolve.grad(alpha_trialfunctions[n][j])[1] / y_scaling * ngsolve.dx for n in range(0, M)]) for m in range(0, M)])
                                a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,l) * vmom_testfunctions[p][l] * G1(m,n,p) * beta_trialfunctions[m][i] * ngsolve.grad(beta_trialfunctions[n][j])[1] / y_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                                # Vertical advection
                                a += sum([sum([advection_epsilon * ramp * H3(i,j,l) * G2(m,n,p) * umom_testfunctions[p][l] * alpha_trialfunctions[m][i] * (
                                    (H+R) * ngsolve.grad(alpha_trialfunctions[n][j])[0] / x_scaling + (Hx+Rx) * alpha_trialfunctions[n][j] / x_scaling + \
                                    (H+R) * ngsolve.grad(beta_trialfunctions[n][j])[1] / y_scaling + (Hy+Ry) * beta_trialfunctions[n][j] / y_scaling
                                ) * ngsolve.dx for n in range(M)]) for m in range(M)])
                                a += sum([sum([advection_epsilon * ramp * H3(i,j,-l) * G2(m,n,p) * vmom_testfunctions[p][l] * beta_trialfunctions[m][i] * (
                                    (H+R) * ngsolve.grad(alpha_trialfunctions[n][j])[0] / x_scaling + (Hx+Rx) * alpha_trialfunctions[n][j] / x_scaling + \
                                    (H+R) * ngsolve.grad(beta_trialfunctions[n][j])[1] / y_scaling + (Hy+Ry) * beta_trialfunctions[n][j] / y_scaling
                                ) * ngsolve.dx for n in range(M)]) for m in range(M)])

                                if model_options['river_boundary_treatment'] == 'exact':
                                    # along-channel advection
                                    a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,-l) * umom_testfunctions[p][-l] * G1(m,n,p) * Q_trialfunctions[i] * normalalpha[m] * river_interpolant * ngsolve.grad(alpha_trialfunctions[n][j])[0] / x_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                                    a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,-l) * umom_testfunctions[p][-l] * G1(m,n,p) * Q_trialfunctions[i] * normalalpha[m] * river_interpolant * Q_trialfunctions[j] * normalalpha[n] * river_interpolant_x / x_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                                    a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,-l) * umom_testfunctions[p][-l] * G1(m,n,p) * alpha_trialfunctions[m][i] * Q_trialfunctions[j] * normalalpha[n] * river_interpolant_x / x_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                                    
                                    a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,-l) * vmom_testfunctions[p][-l] * G1(m,n,p) * Q_trialfunctions[i] * normalalpha[m] * river_interpolant * ngsolve.grad(beta_trialfunctions[n][j])[0] / x_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                                    # lateral advection
                                    a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,-l) * umom_testfunctions[p][-l] * G1(m,n,p) * beta_trialfunctions[m][i] * Q_trialfunctions[j] * normalalpha_y[n] * river_interpolant / y_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                                    # vertical advection
                                    a += sum([sum([advection_epsilon * ramp * H3(i,j,-l) * G2(m,n,p) * umom_testfunctions[p][-l] * Q_trialfunctions[i] * normalalpha[m] * river_interpolant * (H+R) * ngsolve.grad(alpha_trialfunctions[n][j])[0] / x_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                                    a += sum([sum([advection_epsilon * ramp * H3(i,j,-l) * G2(m,n,p) * umom_testfunctions[p][-l] * Q_trialfunctions[i] * normalalpha[m] * river_interpolant * (Hx+Rx) * alpha_trialfunctions[n][j][0] / x_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                                    a += sum([sum([advection_epsilon * ramp * H3(i,j,-l) * G2(m,n,p) * umom_testfunctions[p][-l] * alpha_trialfunctions[m][i] * (H+R) * Q_trialfunctions[j] * normalalpha[n] * river_interpolant_x / x_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                                    a += sum([sum([advection_epsilon * ramp * H3(i,j,-l) * G2(m,n,p) * umom_testfunctions[p][-l] * alpha_trialfunctions[m][i] * (Hx+Rx) * Q_trialfunctions[j] * normalalpha[n] * river_interpolant / x_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                                    a += sum([sum([advection_epsilon * ramp * H3(i,j,-l) * G2(m,n,p) * umom_testfunctions[p][-l] * Q_trialfunctions[i] * normalalpha[m] * river_interpolant * (H+R) * Q_trialfunctions[j] * normalalpha[n] * river_interpolant_x / x_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                                    a += sum([sum([advection_epsilon * ramp * H3(i,j,-l) * G2(m,n,p) * umom_testfunctions[p][-l] * Q_trialfunctions[i] * normalalpha[m] * river_interpolant * (Hx+Rx) * Q_trialfunctions[j] * normalalpha[n] * river_interpolant / x_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                                    a += sum([sum([advection_epsilon * ramp * H3(i,j,-l) * G2(m,n,p) * umom_testfunctions[p][-l] * Q_trialfunctions[i] * normalalpha[m] * river_interpolant * (H+R) * ngsolve.grad(beta_trialfunctions[n][j])[1] / y_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                                    a += sum([sum([advection_epsilon * ramp * H3(i,j,-l) * G2(m,n,p) * umom_testfunctions[p][-l] * Q_trialfunctions[i] * normalalpha[m] * river_interpolant * (Hy+Ry) * beta_trialfunctions[n][j]/ y_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                                    a += sum([sum([advection_epsilon * ramp * H3(i,j,-l) * G2(m,n,p) * vmom_testfunctions[p][-l] * beta_trialfunctions[m][i] * (H+R) * Q_trialfunctions[j] * normalalpha[n] * river_interpolant_x / x_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                                    a += sum([sum([advection_epsilon * ramp * H3(i,j,-l) * G2(m,n,p) * vmom_testfunctions[p][-l] * beta_trialfunctions[m][i] * (Hx+Rx) * Q_trialfunctions[j] * normalalpha[n] * river_interpolant / x_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])

                                    # along-channel advection
                                    a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,l) * umom_testfunctions[p][l] * G1(m,n,p) * Q_trialfunctions[i] * normalalpha[m] * river_interpolant * ngsolve.grad(alpha_trialfunctions[n][j])[0] / x_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                                    a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,l) * umom_testfunctions[p][l] * G1(m,n,p) * Q_trialfunctions[i] * normalalpha[m] * river_interpolant * Q_trialfunctions[j] * normalalpha[n] * river_interpolant_x / x_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                                    a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,l) * umom_testfunctions[p][l] * G1(m,n,p) * alpha_trialfunctions[m][i] * Q_trialfunctions[j] * normalalpha[n] * river_interpolant_x / x_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                                    
                                    a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,l) * vmom_testfunctions[p][l] * G1(m,n,p) * Q_trialfunctions[i] * normalalpha[m] * river_interpolant * ngsolve.grad(beta_trialfunctions[n][j])[0] / x_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                                    # lateral advection
                                    a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,l) * umom_testfunctions[p][l] * G1(m,n,p) * beta_trialfunctions[m][i] * Q_trialfunctions[j] * normalalpha_y[n] * river_interpolant / y_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                                    # vertical advection
                                    a += sum([sum([advection_epsilon * ramp * H3(i,j,l) * G2(m,n,p) * umom_testfunctions[p][l] * Q_trialfunctions[i] * normalalpha[m] * river_interpolant * (H+R) * ngsolve.grad(alpha_trialfunctions[n][j])[0] / x_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                                    a += sum([sum([advection_epsilon * ramp * H3(i,j,l) * G2(m,n,p) * umom_testfunctions[p][l] * Q_trialfunctions[i] * normalalpha[m] * river_interpolant * (Hx+Rx) * alpha_trialfunctions[n][j][0] / x_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                                    a += sum([sum([advection_epsilon * ramp * H3(i,j,l) * G2(m,n,p) * umom_testfunctions[p][l] * alpha_trialfunctions[m][i] * (H+R) * Q_trialfunctions[j] * normalalpha[n] * river_interpolant_x / x_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                                    a += sum([sum([advection_epsilon * ramp * H3(i,j,l) * G2(m,n,p) * umom_testfunctions[p][l] * alpha_trialfunctions[m][i] * (Hx+Rx) * Q_trialfunctions[j] * normalalpha[n] * river_interpolant / x_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                                    a += sum([sum([advection_epsilon * ramp * H3(i,j,l) * G2(m,n,p) * umom_testfunctions[p][l] * Q_trialfunctions[i] * normalalpha[m] * river_interpolant * (H+R) * Q_trialfunctions[j] * normalalpha[n] * river_interpolant_x / x_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                                    a += sum([sum([advection_epsilon * ramp * H3(i,j,l) * G2(m,n,p) * umom_testfunctions[p][l] * Q_trialfunctions[i] * normalalpha[m] * river_interpolant * (Hx+Rx) * Q_trialfunctions[j] * normalalpha[n] * river_interpolant / x_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                                    a += sum([sum([advection_epsilon * ramp * H3(i,j,l) * G2(m,n,p) * umom_testfunctions[p][l] * Q_trialfunctions[i] * normalalpha[m] * river_interpolant * (H+R) * ngsolve.grad(beta_trialfunctions[n][j])[1] / y_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                                    a += sum([sum([advection_epsilon * ramp * H3(i,j,l) * G2(m,n,p) * umom_testfunctions[p][l] * Q_trialfunctions[i] * normalalpha[m] * river_interpolant * (Hy+Ry) * beta_trialfunctions[n][j]/ y_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                                    a += sum([sum([advection_epsilon * ramp * H3(i,j,l) * G2(m,n,p) * vmom_testfunctions[p][l] * beta_trialfunctions[m][i] * (H+R) * Q_trialfunctions[j] * normalalpha[n] * river_interpolant_x / x_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                                    a += sum([sum([advection_epsilon * ramp * H3(i,j,l) * G2(m,n,p) * vmom_testfunctions[p][l] * beta_trialfunctions[m][i] * (Hx+Rx) * Q_trialfunctions[j] * normalalpha[n] * river_interpolant / x_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                                

            # Coriolis
            a += -0.5 * vertical_innerproduct(p, p) * f * (H+R) * beta_trialfunctions[p][-l] * umom_testfunctions[p][-l] * ngsolve.dx
            a += 0.5 * vertical_innerproduct(p, p) * f * (H+R) * alpha_trialfunctions[p][-l] * vmom_testfunctions[p][-l] * ngsolve.dx
            if model_options['river_boundary_treatment'] == 'exact':
                a += 0.5 * vertical_innerproduct(p,p) * f * (H+R) * Q_trialfunctions[-l] * normalalpha[p] * river_interpolant * vmom_testfunctions[p][-l] * ngsolve.dx
            # Barotropic pressure gradient
            a += 0.5 * (H+R) * g * G4(p) * ngsolve.grad(gamma_trialfunctions[-l])[0] * umom_testfunctions[p][-l] / x_scaling * ngsolve.dx
            a += 0.5 * (H+R) * g * G4(p) * ngsolve.grad(gamma_trialfunctions[-l])[1] * vmom_testfunctions[p][-l] / y_scaling * ngsolve.dx
            if model_options['sea_boundary_treatment'] == 'exact':
                a += 0.5 * (H+R) * g * G4(p) * A_trialfunctions[-l] * sea_interpolant_x * umom_testfunctions[p][-l] / x_scaling * ngsolve.dx
            # Vertical eddy viscosity
            if model_options['veddy_viscosity_assumption'] == 'constant':
                a += -0.5 * Av * G3(p, p) * alpha_trialfunctions[p][-l] * umom_testfunctions[p][-l] / (H+R) * ngsolve.dx # assumes that vertical basis consists of eigenfunctions of the vertical mixing operator
                a += -0.5 * Av * G3(p, p) * beta_trialfunctions[p][-l] * vmom_testfunctions[p][-l] / (H+R) * ngsolve.dx
                if model_options['river_boundary_treatment'] == 'exact':
                    a += -0.5 * Av * G3(p, p) * Q_trialfunctions[-l] * normalalpha[p] * river_interpolant * umom_testfunctions[p][-l] / (H+R) * ngsolve.dx
            elif model_options['veddy_viscosity_assumption'] == 'depth-scaled&constantprofile': 
                a += -0.5 * Av * G3(p, p) * alpha_trialfunctions[p][-l] * umom_testfunctions[p][-l] *ngsolve.dx # assumes that vertical basis consists of eigenfunctions of the vertical mixing operator
                a += -0.5 * Av * G3(p, p) * beta_trialfunctions[p][-l] * vmom_testfunctions[p][-l] * ngsolve.dx
                if model_options['river_boundary_treatment'] == 'exact':
                    a += -0.5 * Av * G3(p, p) * Q_trialfunctions[-l] * normalalpha[p] * river_interpolant * umom_testfunctions[p][-l] * ngsolve.dx
            # Horizontal eddy viscosity
            a += 0.5 * vertical_innerproduct(p, p) * Ah * (
                ngsolve.grad(alpha_trialfunctions[p][-l])[0] * ((H+R)*ngsolve.grad(umom_testfunctions[p][-l])[0] / (x_scaling**2) + umom_testfunctions[p][-l]*(Hx+Rx) / (x_scaling**2)) + \
                ngsolve.grad(alpha_trialfunctions[p][-l])[1] * ((H+R)*ngsolve.grad(umom_testfunctions[p][-l])[1] / (y_scaling**2) + umom_testfunctions[p][-l]*(Hy+Ry) / (y_scaling**2))
            ) * ngsolve.dx
            a += 0.5 * vertical_innerproduct(p, p) * Ah * (
                ngsolve.grad(beta_trialfunctions[p][-l])[0] * ((H+R)*ngsolve.grad(vmom_testfunctions[p][-l])[0] / (x_scaling**2) + vmom_testfunctions[p][-l]*(Hx+Rx) / (x_scaling**2)) + \
                ngsolve.grad(beta_trialfunctions[p][-l])[1] * ((H+R)*ngsolve.grad(vmom_testfunctions[p][-l])[1] / (y_scaling**2) + vmom_testfunctions[p][-l]*(Hy+Ry) / (y_scaling**2))
            ) * ngsolve.dx
            if model_options['river_boundary_treatment'] == 'exact':
                a += 0.5 * vertical_innerproduct(p, p) * Ah * (
                    Q_trialfunctions[-l] * normalalpha[p] * river_interpolant_x * ((H+R)*ngsolve.grad(umom_testfunctions[p][-l])[0] / (x_scaling**2) + umom_testfunctions[p][-l]*(Hx+Rx) / (x_scaling**2)) + \
                    Q_trialfunctions[-l] * normalalpha_y[p] * river_interpolant * ((H+R)*ngsolve.grad(umom_testfunctions[p][-l])[1] / (y_scaling**2) + umom_testfunctions[p][-l]*(Hy+Ry) / (y_scaling**2))
                ) * ngsolve.dx

            # Coriolis
            a += -0.5 * vertical_innerproduct(p, p) * f * (H+R) * beta_trialfunctions[p][l] * umom_testfunctions[p][l] * ngsolve.dx
            a += 0.5 * vertical_innerproduct(p, p) * f * (H+R) * alpha_trialfunctions[p][l] * vmom_testfunctions[p][l] * ngsolve.dx
            if model_options['river_boundary_treatment'] == 'exact':
                a += 0.5 * vertical_innerproduct(p,p) * f * (H+R) * Q_trialfunctions[l] * normalalpha[p] * river_interpolant * vmom_testfunctions[p][l] * ngsolve.dx
            # Barotropic pressure gradient
            a += 0.5 * (H+R) * g * G4(p) * ngsolve.grad(gamma_trialfunctions[l])[0] * umom_testfunctions[p][l] / x_scaling * ngsolve.dx
            a += 0.5 * (H+R) * g * G4(p) * ngsolve.grad(gamma_trialfunctions[l])[1] * vmom_testfunctions[p][l] / y_scaling * ngsolve.dx
            if model_options['sea_boundary_treatment'] == 'exact':
                a += 0.5 * (H+R) * g * G4(p) * A_trialfunctions[l] * sea_interpolant_x * umom_testfunctions[p][l] / x_scaling * ngsolve.dx
            # Vertical eddy viscosity
            if model_options['veddy_viscosity_assumption'] == 'constant':
                a += -0.5 * Av * G3(p, p) * alpha_trialfunctions[p][l] * umom_testfunctions[p][l] / (H+R) * ngsolve.dx # assumes that vertical basis consists of eigenfunctions of the vertical mixing operator
                a += -0.5 * Av * G3(p, p) * beta_trialfunctions[p][l] * vmom_testfunctions[p][l] / (H+R) * ngsolve.dx
                if model_options['river_boundary_treatment'] == 'exact':
                    a += -0.5 * Av * G3(p, p) * Q_trialfunctions[l] * normalalpha[p] * river_interpolant * umom_testfunctions[p][l] / (H+R) * ngsolve.dx
            elif model_options['veddy_viscosity_assumption'] == 'depth-scaled&constantprofile': 
                a += -0.5 * Av * G3(p, p) * alpha_trialfunctions[p][l] * umom_testfunctions[p][l] *ngsolve.dx # assumes that vertical basis consists of eigenfunctions of the vertical mixing operator
                a += -0.5 * Av * G3(p, p) * beta_trialfunctions[p][l] * vmom_testfunctions[p][l] * ngsolve.dx
                if model_options['river_boundary_treatment'] == 'exact':
                    a += -0.5 * Av * G3(p, p) * Q_trialfunctions[l] * normalalpha[p] * river_interpolant * umom_testfunctions[p][l] * ngsolve.dx
            # Horizontal eddy viscosity
            a += 0.5 * vertical_innerproduct(p, p) * Ah * (
                ngsolve.grad(alpha_trialfunctions[p][l])[0] * ((H+R)*ngsolve.grad(umom_testfunctions[p][l])[0] / (x_scaling**2) + umom_testfunctions[p][l]*(Hx+Rx) / (x_scaling**2)) + \
                ngsolve.grad(alpha_trialfunctions[p][l])[1] * ((H+R)*ngsolve.grad(umom_testfunctions[p][l])[1] / (y_scaling**2) + umom_testfunctions[p][l]*(Hy+Ry) / (y_scaling**2))
            ) * ngsolve.dx
            a += 0.5 * vertical_innerproduct(p, p) * Ah * (
                ngsolve.grad(beta_trialfunctions[p][l])[0] * ((H+R)*ngsolve.grad(vmom_testfunctions[p][l])[0] / (x_scaling**2) + vmom_testfunctions[p][l]*(Hx+Rx) / (x_scaling**2)) + \
                ngsolve.grad(beta_trialfunctions[p][l])[1] * ((H+R)*ngsolve.grad(vmom_testfunctions[p][l])[1] / (y_scaling**2) + vmom_testfunctions[p][l]*(Hy+Ry) / (y_scaling**2))
            ) * ngsolve.dx
            if model_options['river_boundary_treatment'] == 'exact':
                a += 0.5 * vertical_innerproduct(p, p) * Ah * (
                    Q_trialfunctions[l] * normalalpha[p] * river_interpolant_x * ((H+R)*ngsolve.grad(umom_testfunctions[p][l])[0] / (x_scaling**2) + umom_testfunctions[p][l]*(Hx+Rx) / (x_scaling**2)) + \
                    Q_trialfunctions[l] * normalalpha_y[p] * river_interpolant * ((H+R)*ngsolve.grad(umom_testfunctions[p][l])[1] / (y_scaling**2) + umom_testfunctions[p][l]*(Hy+Ry) / (y_scaling**2))
                ) * ngsolve.dx


def add_linearised_nonlinear_terms(a: ngsolve.BilinearForm, model_options: dict, numerical_information: dict, geometric_information: dict, constant_parameters, spatial_parameters,
                                   alpha_trialfunctions, alpha0, beta_trialfunctions, beta0, gamma_trialfunctions,
                                   gamma0, umom_testfunctions, vmom_testfunctions, DIC_testfunctions, 
                                   vertical_basis: TruncationBasis, time_basis: TruncationBasis, normalalpha, normalalpha_y=None, A_trialfunctions=None, A0=None,
                                   sea_boundary_testfunctions=None, Q_trialfunctions=None, Q0=None):
    """Adds the (Fréchet/Gâteaux) linearisation of the nonlinear terms (advection and/or surface_in_sigma) to a bilinear form.

    Arguments:
    
        - a (ngsolve.BilinearForm):                     bilinear form object on which the weak form is constructed;
        - model_options (dict):                         model options of the hydrodynamics object;
        - numerical_information (dict):                 numerical information dictionary of the hydrodynamics object;
        - geometric_information (dict):                 geometric information dictionary of the hydrodynamics object;
        - constant_parameters (dict):                   dictionary of constant physical parameters associated with the model;
        - spatial_parameters (dict):                    dictionary of spatial physical parameters (as SpatialParameter objects) associated with the model;
        - alpha_trialfunctions (dict):                  dictionary of trial functions representing the alpha-Fourier/eigenfunction coefficients (obtainable via hydrodynamics._setup_TnT)
        - alpha0 (dict):                                dictionary of gridfunctions containing the value of the alpha coefficients at the current Newton iteration, at which the form is linearised;
        - beta_trialfunctions (dict):                   dictionary of trial functions representing the beta-Fourier/eigenfunction coefficients;
        - beta0 (dict):                                 dictionary of gridfunctions containing the value of the beta coefficients at the current Newton iteration, at which the form is linearised;
        - gamma_trialfunctions (dict):                  dictionary of trial functions representing the gamma-Fourier/eigenfunction coefficients;
        - gamma0 (dict):                                dictionary of gridfunctions containing the value of the gamma coefficients at the current Newton iteration, at which the form is linearised;
        - umom_testfunctions (dict):                    dictionary of test functions for the along-channel momentum equations;
        - vmom_testfunctions (dict):                    dictionary of test functions for the lateral momentum equations;
        - DIC_testfunctions (dict):                     dictionary of test functions for the depth-integrated continuity equation;
        - vertical_basis (TruncationBasis):             vertical eigenfunction basis;
        - time_basis (TruncationBasis):                 temporal Fourier basis;
        - normalalpha (list):                           list of coefficient functions containing the lateral structure of the riverine along-channel flow for that boundary condition;
        - normalalpha_y (list):                         analytical y-derivatives (eta) for the normalalpha-functions;
        - A_trialfunctions (dict):                      dictionary of trial functions representing the computational seaward boundary condition of gamma (free surface);
        - A0 (dict):                                    dictionary of gridfunctions (constant value) containing the value of A at the current Newton iteration, at which the form is linearised;
        - sea_boundary_testfunctions (dict):            dictionary of test functions for the interpretable seaward boundary condition;
        - Q_trialfunctions (dict):                      dictionary of trial functions representing the computational riverine boundary condition for alpha (along-channel velocity);
        - Q0 (dict):                                    dictionary of gridfunctions (constant value) containing the value of Q at the current Newton iteration, at which the form is linearised.
    """

    # Defining shorthands of variables

    H = spatial_parameters['H'].cf
    Hx = spatial_parameters["H"].gradient_cf[0]
    Hy = spatial_parameters['H'].gradient_cf[1]
    rho = spatial_parameters['rho'].cf
    rhox = spatial_parameters['rho'].gradient_cf[0]
    rhoy = spatial_parameters['rho'].gradient_cf[1]
    R = spatial_parameters['R'].cf
    Rx = spatial_parameters['R'].gradient_cf[0]
    Ry = spatial_parameters['R'].gradient_cf[1]

    f = constant_parameters['f']
    g = constant_parameters['g']
    Av = constant_parameters['Av']
    sigma = constant_parameters['sigma']
    Q = constant_parameters['discharge']

    x_scaling = geometric_information['x_scaling']
    y_scaling = geometric_information['y_scaling']

    G1 = vertical_basis.tensor_dict['G1']
    G2 = vertical_basis.tensor_dict['G2']
    G3 = vertical_basis.tensor_dict['G3']
    G4 = vertical_basis.tensor_dict['G4']
    H3 = time_basis.tensor_dict['H3']
    H3_iszero = time_basis.tensor_dict['H3_iszero']

    advection_matrix = model_options['advection_influence_matrix']
    advection_epsilon = constant_parameters['advection_epsilon']

    M = numerical_information['M']
    imax = numerical_information['imax']

    # Construct the non-linear ramp
    if geometric_information['L_R_sea'] > 1e-16:
        ramp_sea = ngsolve.IfPos(
            -geometric_information['L_R_sea']/x_scaling - ngsolve.x,
            ngsolve.IfPos(
                -geometric_information['L_R_sea']/x_scaling - geometric_information['L_RA_sea']/x_scaling - ngsolve.x,
                0,
                0.5 * (1 + ngsolve_tanh((4 * (ngsolve.x + geometric_information['L_RA_sea']/x_scaling + 0.5 * geometric_information['L_R_sea']/x_scaling) / (geometric_information['L_R_sea']/x_scaling)) / 
                                        (1 - (2*(ngsolve.x + geometric_information['L_RA_sea']/x_scaling + 0.5*geometric_information['L_R_sea']/x_scaling) / (geometric_information['L_R_sea']/x_scaling))**2)))
            ),
            1
        )
    else:
        ramp_sea = ngsolve.CF(1)

    if geometric_information['L_R_river'] > 1e-16 and model_options['river_boundary_treatment'] == 'exact':
        ramp_river = ngsolve.IfPos(
                    -(geometric_information['riverine_boundary_x'] + geometric_information['L_RA_river']) / x_scaling + ngsolve.x,
                    ngsolve.IfPos(
                        -(geometric_information['riverine_boundary_x'] + geometric_information['L_R_river'] + geometric_information['L_RA_river']) / x_scaling + ngsolve.x,
                        0,
                        0.5 * (1 + ngsolve_tanh((-4 * (ngsolve.x - geometric_information['L_RA_river'] / x_scaling - 0.5 * geometric_information["L_R_river"] / x_scaling - geometric_information['riverine_boundary_x'] / x_scaling) / 
                                                (geometric_information['L_R_river'] / x_scaling)) / (1 - (2*(ngsolve.x- geometric_information['L_RA_river']/x_scaling - 0.5 * geometric_information["L_R_river"]/x_scaling - geometric_information['riverine_boundary_x']/x_scaling)/(geometric_information["L_R_river"]/x_scaling))**2)))
                    ),
                    1
                )
    elif geometric_information['L_R_river'] > 1e-16 and model_options['river_boundary_treatment'] == 'simple':
        ramp_river = ngsolve.IfPos(
            -(geometric_information['riverine_boundary_x'] - geometric_information['L_RA_river']) / x_scaling + ngsolve.x,
            ngsolve.IfPos(
                -(geometric_information['riverine_boundary_x'] - geometric_information['L_R_river'] - geometric_information['L_RA_river']) / x_scaling + ngsolve.x,
                0,
                0.5 * (1 + ngsolve_tanh((4 * (ngsolve.x - geometric_information['L_RA_river'] / x_scaling + 0.5 * geometric_information["L_R_river"] / x_scaling - geometric_information['riverine_boundary_x'] / x_scaling) / 
                                         (geometric_information['L_R_river'] / x_scaling)) / (1 - (2*(ngsolve.x- geometric_information['L_RA_river']/x_scaling + 0.5 * geometric_information["L_R_river"]/x_scaling - geometric_information['riverine_boundary_x']/x_scaling)/(geometric_information["L_R_river"]/x_scaling))**2)))
            ),
            1
        )
    else:
        ramp_river = ngsolve.CF(1)

    ramp = ramp_sea * ramp_river

    # make the linear interpolant functions to treat computational boundary conditions
    if model_options['sea_boundary_treatment'] == 'exact':
        sea_interpolant = ((geometric_information['riverine_boundary_x'] / x_scaling) + (geometric_information['L_BL_river']/x_scaling) + (geometric_information['L_R_river']/x_scaling) + \
                           (geometric_information['L_RA_river']/x_scaling) - ngsolve.x) / \
                           ((geometric_information['riverine_boundary_x']+geometric_information['L_BL_river']+geometric_information['L_R_river']+geometric_information['L_RA_river'] +
                            geometric_information['L_BL_sea'] + geometric_information['L_R_sea'] + geometric_information['L_RA_sea']) / x_scaling)
        sea_interpolant_x = -1 / ((geometric_information['riverine_boundary_x']+geometric_information['L_BL_river']+geometric_information['L_R_river']+geometric_information['L_RA_river'] +
                            geometric_information['L_BL_sea'] + geometric_information['L_R_sea'] + geometric_information['L_RA_sea']) / x_scaling)
    if model_options['river_boundary_treatment'] == 'exact':
        river_interpolant = (-(geometric_information['L_BL_sea']/x_scaling) - (geometric_information['L_R_sea']/x_scaling) - \
                             (geometric_information['L_RA_sea']/x_scaling) + ngsolve.x) / \
                             ((geometric_information['riverine_boundary_x']+geometric_information['L_BL_river']+geometric_information['L_R_river']+geometric_information['L_RA_river'] +
                             geometric_information['L_BL_sea'] + geometric_information['L_R_sea'] + geometric_information['L_RA_sea']) / x_scaling)
        river_interpolant_x = 1 / ((geometric_information['riverine_boundary_x']+geometric_information['L_BL_river']+geometric_information['L_R_river']+geometric_information['L_RA_river'] +
                             geometric_information['L_BL_sea'] + geometric_information['L_R_sea'] + geometric_information['L_RA_sea']) / x_scaling)
    # if model_options['sea_boundary_treatment'] == 'exact':
    #     sea_interpolant = 1
    #     sea_interpolant_x = 0
    # if geometric_information['L_R_river'] > 1e-16:
    #     river_interpolant = 1
    #     river_interpolant_x = 0


    # INTERPRETABLE SEAWARD BOUNDARY CONDITION ====================================================================================================================

    dirac_delta_width = 0.05 # this value is based on the fact that we have a domain scaled to approximately a unit square
    # dirac_delta_sea = ngsolve.IfPos((ngsolve.x - dirac_delta_width/2 - geometric_information['L_R_sea_boundary_layer_length']) * (model_options['sea'] - dirac_delta_width/2 - ngsolve.x), # if this function is positive, i.e. in [xsea - DDwidth/2, xsea + DDwidth/2],
    #                                 1 / dirac_delta_width, # then return 1/DDwidth as value
    #                                 0) # and otherwise zero

    
    dirac_delta_sea = ngsolve.IfPos((ngsolve.x - dirac_delta_width/2) * (-dirac_delta_width/2 - ngsolve.x),
                                    (4/(dirac_delta_width**2)) * (dirac_delta_width / 2 - ngsolve.sqrt((ngsolve.x)**2)),
                                    0) # Hat function Dirac Delta
    
    # So this is an approximation of the Dirac delta at x=xsea
    if model_options['sea_boundary_treatment'] == 'exact':
        # terms l != 0
        for l in range(1, imax + 1):
            a += dirac_delta_sea / ngsolve.sqrt((gamma0[-l]+A0[-l]*sea_interpolant)**2 + (gamma0[l]+A0[l]*sea_interpolant)**2) * \
                ((gamma0[l]+A0[l]*sea_interpolant)*(gamma_trialfunctions[l]+A_trialfunctions[l]*sea_interpolant) + (gamma0[-l]+A0[-l]*sea_interpolant)*(gamma_trialfunctions[-l]+A_trialfunctions[-l]*sea_interpolant)) * sea_boundary_testfunctions[-l] * ngsolve.dx

            a += dirac_delta_sea * (
                -(gamma_trialfunctions[-l]+A_trialfunctions[-l]*sea_interpolant) / ((gamma0[l]+A0[l]*sea_interpolant) * (1 + ((gamma0[-l]+A0[-l]*sea_interpolant)**2 / (gamma0[l]+A0[l]*sea_interpolant)**2))) + \
                (gamma_trialfunctions[l]+A_trialfunctions[l]*sea_interpolant) * (gamma0[-l]+A0[-l]*sea_interpolant) / ((gamma0[-l]+A0[-l]*sea_interpolant)**2 + (gamma0[l]+A0[l]*sea_interpolant)**2)
            ) * sea_boundary_testfunctions[l] * ngsolve.dx

    ## MOMENTUM EQUATIONS #######################################################################
    for p in range(M):
        for i in range(-imax, imax+1):
            for j in range(-imax, imax+1):
                if H3_iszero(i, j, 0):
                    continue
                else:
                    if advection_matrix[0, abs(i)] and advection_matrix[0, abs(j)]:
                        # Along-channel advection
                        a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,0) * G1(m,n,p) * umom_testfunctions[p][0] * alpha_trialfunctions[m][i] * ngsolve.grad(alpha0[n][j])[0] / x_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                        a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,0) * G1(m,n,p) * vmom_testfunctions[p][0] * alpha_trialfunctions[m][i] * ngsolve.grad(beta0[n][j])[0] / x_scaling *ngsolve.dx for n in range(M)]) for m in range(M)])
                        # Lateral advection
                        a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,0) * G1(m,n,p) * umom_testfunctions[p][0] * beta_trialfunctions[m][i] * ngsolve.grad(alpha0[n][j])[1] / y_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                        a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,0) * G1(m,n,p) * vmom_testfunctions[p][0] * beta_trialfunctions[m][i] * ngsolve.grad(beta0[n][j])[1] / y_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                        # Vertical advection
                        a += sum([sum([advection_epsilon * ramp * H3(i,j,0) * G2(m,n,p) * umom_testfunctions[p][0] * alpha_trialfunctions[m][i] * (
                            (Hx + Rx) * alpha0[n][j] / x_scaling + (H+R) * ngsolve.grad(alpha0[n][j])[0] / x_scaling + \
                            (Hy + Ry) * beta0[n][j] / y_scaling + (H+R) * ngsolve.grad(beta0[n][j])[1] / y_scaling
                        ) * ngsolve.dx for n in range(M)]) for m in range(M)])
                        a += sum([sum([advection_epsilon * ramp * H3(i,j,0) * G2(m,n,p) * vmom_testfunctions[p][0] * beta_trialfunctions[m][i] * (
                            (Hx + Rx) * alpha0[n][j] / x_scaling + (H+R) * ngsolve.grad(alpha0[n][j])[0] / x_scaling + \
                            (Hy + Ry) * beta0[n][j] / y_scaling + (H+R) * ngsolve.grad(beta0[n][j])[1] / y_scaling
                        ) * ngsolve.dx for n in range(M)]) for m in range(M)])
                        if model_options['river_boundary_treatment'] == 'exact':
                                # Along-channel advection
                                a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,0) * G1(m,n,p) * umom_testfunctions[p][0] * Q_trialfunctions[i] * normalalpha[m] * river_interpolant * ngsolve.grad(alpha0[n][j])[0] / x_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                                a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,0) * G1(m,n,p) * umom_testfunctions[p][0] * alpha_trialfunctions[m][i] *  Q0[j] * normalalpha[n] * river_interpolant_x / x_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                                a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,0) * G1(m,n,p) * umom_testfunctions[p][0] * Q_trialfunctions[i] * normalalpha[m] * river_interpolant * Q0[j] * normalalpha[n] * river_interpolant_x / x_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])

                                a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,0) * G1(m,n,p) * vmom_testfunctions[p][0] * Q_trialfunctions[i] * normalalpha[m] * river_interpolant * ngsolve.grad(beta0[n][j])[0] / x_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                                # Lateral advection
                                a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,0) * G1(m,n,p) * umom_testfunctions[p][0] * beta_trialfunctions[m][i] * Q0[j] * normalalpha_y[n] * river_interpolant / y_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                                # Vertical advection
                                a += sum([sum([advection_epsilon * ramp * H3(i,j,0) * G2(m,n,p) * umom_testfunctions[p][0] * Q_trialfunctions[i] * normalalpha[m] * river_interpolant * (H+R) * ngsolve.grad(alpha0[n][j])[0] / x_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                                a += sum([sum([advection_epsilon * ramp * H3(i,j,0) * G2(m,n,p) * umom_testfunctions[p][0] * Q_trialfunctions[i] * normalalpha[m] * river_interpolant * (Hx+Rx) * alpha0[n][j] / x_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                                a += sum([sum([advection_epsilon * ramp * H3(i,j,0) * G2(m,n,p) * umom_testfunctions[p][0] * Q_trialfunctions[i] * normalalpha[m] * river_interpolant * (H+R) * Q0[j] * normalalpha[n] * river_interpolant_x / x_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                                a += sum([sum([advection_epsilon * ramp * H3(i,j,0) * G2(m,n,p) * umom_testfunctions[p][0] * Q_trialfunctions[i] * normalalpha[m] * river_interpolant * (Hx+Rx) * Q0[j] * normalalpha[n] * river_interpolant / x_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                                a += sum([sum([advection_epsilon * ramp * H3(i,j,0) * G2(m,n,p) * umom_testfunctions[p][0] * alpha_trialfunctions[m][i] * (H+R) * Q0[j] * normalalpha[n] * river_interpolant_x / x_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                                a += sum([sum([advection_epsilon * ramp * H3(i,j,0) * G2(m,n,p) * umom_testfunctions[p][0] * alpha_trialfunctions[m][i] * (Hx+Rx) * Q0[j] * normalalpha[n] * river_interpolant / x_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])

                                a += sum([sum([advection_epsilon * ramp * H3(i,j,0) * G2(m,n,p) * vmom_testfunctions[p][0] * beta_trialfunctions[m][i] * (H+R) * Q0[j] * normalalpha[n] * river_interpolant_x / x_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                                a += sum([sum([advection_epsilon * ramp * H3(i,j,0) * G2(m,n,p) * vmom_testfunctions[p][0] * beta_trialfunctions[m][i] * (Hx+Rx) * Q0[j] * normalalpha[n] * river_interpolant / x_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])


                        # Along-channel advection
                        a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,0) * G1(m,n,p) * umom_testfunctions[p][0] * alpha0[m][i] * ngsolve.grad(alpha_trialfunctions[n][j])[0] / x_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                        a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,0) * G1(m,n,p) * vmom_testfunctions[p][0] * alpha0[m][i] * ngsolve.grad(beta_trialfunctions[n][j])[0] / x_scaling *ngsolve.dx for n in range(M)]) for m in range(M)])
                        # Lateral advection
                        a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,0) * G1(m,n,p) * umom_testfunctions[p][0] * beta0[m][i] * ngsolve.grad(alpha_trialfunctions[n][j])[1] / y_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                        a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,0) * G1(m,n,p) * vmom_testfunctions[p][0] * beta0[m][i] * ngsolve.grad(beta_trialfunctions[n][j])[1] / y_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                        # Vertical advection
                        a += sum([sum([advection_epsilon * ramp * H3(i,j,0) * G2(m,n,p) * umom_testfunctions[p][0] * alpha0[m][i] * (
                            (Hx + Rx) * alpha_trialfunctions[n][j] / x_scaling + (H+R) * ngsolve.grad(alpha_trialfunctions[n][j])[0] / x_scaling + \
                            (Hy + Ry) * beta_trialfunctions[n][j] / y_scaling + (H+R) * ngsolve.grad(beta_trialfunctions[n][j])[1] / y_scaling
                        ) * ngsolve.dx for n in range(M)]) for m in range(M)])
                        a += sum([sum([advection_epsilon * ramp * H3(i,j,0) * G2(m,n,p) * vmom_testfunctions[p][0] * beta0[m][i] * (
                            (Hx + Rx) * alpha_trialfunctions[n][j] / x_scaling + (H+R) * ngsolve.grad(alpha_trialfunctions[n][j])[0] / x_scaling + \
                            (Hy + Ry) * beta_trialfunctions[n][j] / y_scaling + (H+R) * ngsolve.grad(beta_trialfunctions[n][j])[1] / y_scaling
                        ) * ngsolve.dx for n in range(M)]) for m in range(M)])
                        if model_options['river_boundary_treatment'] == 'exact':
                            # Along-channel advection
                            a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,0) * G1(m,n,p) * umom_testfunctions[p][0] * Q0[i] * normalalpha[m] * river_interpolant * ngsolve.grad(alpha_trialfunctions[n][j])[0] / x_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                            a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,0) * G1(m,n,p) * umom_testfunctions[p][0] * alpha0[m][i] *  Q_trialfunctions[j] * normalalpha[n] * river_interpolant_x / x_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                            a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,0) * G1(m,n,p) * umom_testfunctions[p][0] * Q0[i] * normalalpha[m] * river_interpolant * Q_trialfunctions[j] * normalalpha[n] * river_interpolant_x / x_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])

                            a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,0) * G1(m,n,p) * vmom_testfunctions[p][0] * Q0[i] * normalalpha[m] * river_interpolant * ngsolve.grad(beta_trialfunctions[n][j])[0] / x_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                            # Lateral advection
                            a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,0) * G1(m,n,p) * umom_testfunctions[p][0] * beta0[m][i] * Q_trialfunctions[j] * normalalpha_y[n] * river_interpolant / y_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                            # Vertical advection
                            a += sum([sum([advection_epsilon * ramp * H3(i,j,0) * G2(m,n,p) * umom_testfunctions[p][0] * Q0[i] * normalalpha[m] * river_interpolant * (H+R) * ngsolve.grad(alpha_trialfunctions[n][j])[0] / x_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                            a += sum([sum([advection_epsilon * ramp * H3(i,j,0) * G2(m,n,p) * umom_testfunctions[p][0] * Q0[i] * normalalpha[m] * river_interpolant * (Hx+Rx) * alpha_trialfunctions[n][j] / x_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                            a += sum([sum([advection_epsilon * ramp * H3(i,j,0) * G2(m,n,p) * umom_testfunctions[p][0] * Q0[i] * normalalpha[m] * river_interpolant * (H+R) * Q_trialfunctions[j] * normalalpha[n] * river_interpolant_x / x_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                            a += sum([sum([advection_epsilon * ramp * H3(i,j,0) * G2(m,n,p) * umom_testfunctions[p][0] * Q0[i] * normalalpha[m] * river_interpolant * (Hx+Rx) * Q_trialfunctions[j] * normalalpha[n] * river_interpolant / x_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                            a += sum([sum([advection_epsilon * ramp * H3(i,j,0) * G2(m,n,p) * umom_testfunctions[p][0] * alpha0[m][i] * (H+R) * Q_trialfunctions[j] * normalalpha[n] * river_interpolant_x / x_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                            a += sum([sum([advection_epsilon * ramp * H3(i,j,0) * G2(m,n,p) * umom_testfunctions[p][0] * alpha0[m][i] * (Hx+Rx) * Q_trialfunctions[j] * normalalpha[n] * river_interpolant / x_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])

                            a += sum([sum([advection_epsilon * ramp * H3(i,j,0) * G2(m,n,p) * vmom_testfunctions[p][0] * beta0[m][i] * (H+R) * Q_trialfunctions[j] * normalalpha[n] * river_interpolant_x / x_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                            a += sum([sum([advection_epsilon * ramp * H3(i,j,0) * G2(m,n,p) * vmom_testfunctions[p][0] * beta0[m][i] * (Hx+Rx) * Q_trialfunctions[j] * normalalpha[n] * river_interpolant / x_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])

        
        for l in range(1, imax+1):
            for i in range(-imax, imax + 1):
                for j in range(-imax, imax + 1):
                    if H3_iszero(i,j,-l) and H3_iszero(i,j,l):
                        continue 
                    else:
                        if advection_matrix[l, abs(i)] and advection_matrix[l, abs(j)]:
                            # term -l
                            # Along-channel advection
                            a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,-l) * umom_testfunctions[p][-l] * G1(m,n,p) * alpha_trialfunctions[m][i] * ngsolve.grad(alpha0[n][j])[0] / x_scaling * ngsolve.dx for n in range(0, M)]) for m in range(0, M)])
                            a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,-l) * vmom_testfunctions[p][-l] * G1(m,n,p) * alpha_trialfunctions[m][i] * ngsolve.grad(beta0[n][j])[0] / x_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                            # Lateral advection
                            a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,-l) * umom_testfunctions[p][-l] * G1(m,n,p) * beta_trialfunctions[m][i] * ngsolve.grad(alpha0[n][j])[1] / y_scaling * ngsolve.dx for n in range(0, M)]) for m in range(0, M)])
                            a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,-l) * vmom_testfunctions[p][-l] * G1(m,n,p) * beta_trialfunctions[m][i] * ngsolve.grad(beta0[n][j])[1] / y_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                            # Vertical advection
                            a += sum([sum([advection_epsilon * ramp * H3(i,j,-l) * G2(m,n,p) * umom_testfunctions[p][-l] * alpha_trialfunctions[m][i] * (
                                (H+R) * ngsolve.grad(alpha0[n][j])[0] / x_scaling + (Hx+Rx) * alpha0[n][j] / x_scaling + \
                                (H+R) * ngsolve.grad(beta0[n][j])[1] / y_scaling + (Hy+Ry) * beta0[n][j] / y_scaling
                            ) * ngsolve.dx for n in range(M)]) for m in range(M)])
                            a += sum([sum([advection_epsilon * ramp * H3(i,j,-l) * G2(m,n,p) * vmom_testfunctions[p][-l] * beta_trialfunctions[m][i] * (
                                (H+R) * ngsolve.grad(alpha0[n][j])[0] / x_scaling + (Hx+Rx) * alpha0[n][j] / x_scaling + \
                                (H+R) * ngsolve.grad(beta0[n][j])[1] / y_scaling + (Hy+Ry) * beta0[n][j] / y_scaling
                            ) * ngsolve.dx for n in range(M)]) for m in range(M)])

                            # Along-channel advection
                            a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,-l) * umom_testfunctions[p][-l] * G1(m,n,p) * alpha0[m][i] * ngsolve.grad(alpha_trialfunctions[n][j])[0] / x_scaling * ngsolve.dx for n in range(0, M)]) for m in range(0, M)])
                            a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,-l) * vmom_testfunctions[p][-l] * G1(m,n,p) * alpha0[m][i] * ngsolve.grad(beta_trialfunctions[n][j])[0] / x_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                            # Lateral advection
                            a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,-l) * umom_testfunctions[p][-l] * G1(m,n,p) * beta0[m][i] * ngsolve.grad(alpha_trialfunctions[n][j])[1] / y_scaling * ngsolve.dx for n in range(0, M)]) for m in range(0, M)])
                            a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,-l) * vmom_testfunctions[p][-l] * G1(m,n,p) * beta0[m][i] * ngsolve.grad(beta_trialfunctions[n][j])[1] / y_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                            # Vertical advection
                            a += sum([sum([advection_epsilon * ramp * H3(i,j,-l) * G2(m,n,p) * umom_testfunctions[p][-l] * alpha0[m][i] * (
                                (H+R) * ngsolve.grad(alpha_trialfunctions[n][j])[0] / x_scaling + (Hx+Rx) * alpha_trialfunctions[n][j] / x_scaling + \
                                (H+R) * ngsolve.grad(beta_trialfunctions[n][j])[1] / y_scaling + (Hy+Ry) * beta_trialfunctions[n][j] / y_scaling
                            ) * ngsolve.dx for n in range(M)]) for m in range(M)])
                            a += sum([sum([advection_epsilon * ramp * H3(i,j,-l) * G2(m,n,p) * vmom_testfunctions[p][-l] * beta0[m][i] * (
                                (H+R) * ngsolve.grad(alpha_trialfunctions[n][j])[0] / x_scaling + (Hx+Rx) * alpha_trialfunctions[n][j] / x_scaling + \
                                (H+R) * ngsolve.grad(beta_trialfunctions[n][j])[1] / y_scaling + (Hy+Ry) * beta_trialfunctions[n][j] / y_scaling
                            ) * ngsolve.dx for n in range(M)]) for m in range(M)])

                            # term +l
                            # Along-channel advection
                            a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,l) * umom_testfunctions[p][l] * G1(m,n,p) * alpha0[m][i] * ngsolve.grad(alpha_trialfunctions[n][j])[0] / x_scaling * ngsolve.dx for n in range(0, M)]) for m in range(0, M)])
                            a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,l) * vmom_testfunctions[p][l] * G1(m,n,p) * alpha0[m][i] * ngsolve.grad(beta_trialfunctions[n][j])[0] / x_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                            # Lateral advection
                            a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,l) * umom_testfunctions[p][l] * G1(m,n,p) * beta0[m][i] * ngsolve.grad(alpha_trialfunctions[n][j])[1] / y_scaling * ngsolve.dx for n in range(0, M)]) for m in range(0, M)])
                            a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,l) * vmom_testfunctions[p][l] * G1(m,n,p) * beta0[m][i] * ngsolve.grad(beta_trialfunctions[n][j])[1] / y_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                            # Vertical advection
                            a += sum([sum([advection_epsilon * ramp * H3(i,j,l) * G2(m,n,p) * umom_testfunctions[p][l] * alpha0[m][i] * (
                                (H+R) * ngsolve.grad(alpha_trialfunctions[n][j])[0] / x_scaling + (Hx+Rx) * alpha_trialfunctions[n][j] / x_scaling + \
                                (H+R) * ngsolve.grad(beta_trialfunctions[n][j])[1] / y_scaling + (Hy+Ry) * beta_trialfunctions[n][j] / y_scaling
                            ) * ngsolve.dx for n in range(M)]) for m in range(M)])
                            a += sum([sum([advection_epsilon * ramp * H3(i,j,-l) * G2(m,n,p) * vmom_testfunctions[p][l] * beta0[m][i] * (
                                (H+R) * ngsolve.grad(alpha_trialfunctions[n][j])[0] / x_scaling + (Hx+Rx) * alpha_trialfunctions[n][j] / x_scaling + \
                                (H+R) * ngsolve.grad(beta_trialfunctions[n][j])[1] / y_scaling + (Hy+Ry) * beta_trialfunctions[n][j] / y_scaling
                            ) * ngsolve.dx for n in range(M)]) for m in range(M)])

                            # Along-channel advection
                            a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,l) * umom_testfunctions[p][l] * G1(m,n,p) * alpha0[m][i] * ngsolve.grad(alpha_trialfunctions[n][j])[0] / x_scaling * ngsolve.dx for n in range(0, M)]) for m in range(0, M)])
                            a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,l) * vmom_testfunctions[p][l] * G1(m,n,p) * alpha0[m][i] * ngsolve.grad(beta_trialfunctions[n][j])[0] / x_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                            # Lateral advection
                            a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,l) * umom_testfunctions[p][l] * G1(m,n,p) * beta0[m][i] * ngsolve.grad(alpha_trialfunctions[n][j])[1] / y_scaling * ngsolve.dx for n in range(0, M)]) for m in range(0, M)])
                            a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,l) * vmom_testfunctions[p][l] * G1(m,n,p) * beta0[m][i] * ngsolve.grad(beta_trialfunctions[n][j])[1] / y_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                            # Vertical advection
                            a += sum([sum([advection_epsilon * ramp * H3(i,j,l) * G2(m,n,p) * umom_testfunctions[p][l] * alpha0[m][i] * (
                                (H+R) * ngsolve.grad(alpha_trialfunctions[n][j])[0] / x_scaling + (Hx+Rx) * alpha_trialfunctions[n][j] / x_scaling + \
                                (H+R) * ngsolve.grad(beta_trialfunctions[n][j])[1] / y_scaling + (Hy+Ry) * beta_trialfunctions[n][j] / y_scaling
                            ) * ngsolve.dx for n in range(M)]) for m in range(M)])
                            a += sum([sum([advection_epsilon * ramp * H3(i,j,l) * G2(m,n,p) * vmom_testfunctions[p][l] * beta0[m][i] * (
                                (H+R) * ngsolve.grad(alpha_trialfunctions[n][j])[0] / x_scaling + (Hx+Rx) * alpha_trialfunctions[n][j] / x_scaling + \
                                (H+R) * ngsolve.grad(beta_trialfunctions[n][j])[1] / y_scaling + (Hy+Ry) * beta_trialfunctions[n][j] / y_scaling
                            ) * ngsolve.dx for n in range(M)]) for m in range(M)])

                            if model_options['river_boundary_treatment'] == 'exact':
                                    # First the products where the second variable is replaced by its previous iterate
                                    # along-channel advection
                                    a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,-l) * umom_testfunctions[p][-l] * G1(m,n,p) * Q_trialfunctions[i] * normalalpha[m] * river_interpolant * ngsolve.grad(alpha0[n][j])[0] / x_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                                    a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,-l) * umom_testfunctions[p][-l] * G1(m,n,p) * Q_trialfunctions[i] * normalalpha[m] * river_interpolant * Q0[j] * normalalpha[n] * river_interpolant_x / x_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                                    a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,-l) * umom_testfunctions[p][-l] * G1(m,n,p) * alpha_trialfunctions[m][i] * Q0[j] * normalalpha[n] * river_interpolant_x / x_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                                    
                                    a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,-l) * vmom_testfunctions[p][-l] * G1(m,n,p) * Q_trialfunctions[i] * normalalpha[m] * river_interpolant * ngsolve.grad(beta0[n][j])[0] / x_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                                    # lateral advection
                                    a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,-l) * umom_testfunctions[p][-l] * G1(m,n,p) * beta_trialfunctions[m][i] * Q0[j] * normalalpha_y[n] * river_interpolant / y_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                                    # vertical advection
                                    a += sum([sum([advection_epsilon * ramp * H3(i,j,-l) * G2(m,n,p) * umom_testfunctions[p][-l] * Q_trialfunctions[i] * normalalpha[m] * river_interpolant * (H+R) * ngsolve.grad(alpha0[n][j])[0] / x_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                                    a += sum([sum([advection_epsilon * ramp * H3(i,j,-l) * G2(m,n,p) * umom_testfunctions[p][-l] * Q_trialfunctions[i] * normalalpha[m] * river_interpolant * (Hx+Rx) * alpha0[n][j][0] / x_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                                    a += sum([sum([advection_epsilon * ramp * H3(i,j,-l) * G2(m,n,p) * umom_testfunctions[p][-l] * alpha_trialfunctions[m][i] * (H+R) * Q0[j] * normalalpha[n] * river_interpolant_x / x_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                                    a += sum([sum([advection_epsilon * ramp * H3(i,j,-l) * G2(m,n,p) * umom_testfunctions[p][-l] * alpha_trialfunctions[m][i] * (Hx+Rx) * Q0[j] * normalalpha[n] * river_interpolant / x_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                                    a += sum([sum([advection_epsilon * ramp * H3(i,j,-l) * G2(m,n,p) * umom_testfunctions[p][-l] * Q_trialfunctions[i] * normalalpha[m] * river_interpolant * (H+R) * Q0[j] * normalalpha[n] * river_interpolant_x / x_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                                    a += sum([sum([advection_epsilon * ramp * H3(i,j,-l) * G2(m,n,p) * umom_testfunctions[p][-l] * Q_trialfunctions[i] * normalalpha[m] * river_interpolant * (Hx+Rx) * Q0[j] * normalalpha[n] * river_interpolant / x_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                                    a += sum([sum([advection_epsilon * ramp * H3(i,j,-l) * G2(m,n,p) * umom_testfunctions[p][-l] * Q_trialfunctions[i] * normalalpha[m] * river_interpolant * (H+R) * ngsolve.grad(beta0[n][j])[1] / y_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                                    a += sum([sum([advection_epsilon * ramp * H3(i,j,-l) * G2(m,n,p) * umom_testfunctions[p][-l] * Q_trialfunctions[i] * normalalpha[m] * river_interpolant * (Hy+Ry) * beta0[n][j]/ y_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                                    a += sum([sum([advection_epsilon * ramp * H3(i,j,-l) * G2(m,n,p) * vmom_testfunctions[p][-l] * beta_trialfunctions[m][i] * (H+R) * Q0[j] * normalalpha[n] * river_interpolant_x / x_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                                    a += sum([sum([advection_epsilon * ramp * H3(i,j,-l) * G2(m,n,p) * vmom_testfunctions[p][-l] * beta_trialfunctions[m][i] * (Hx+Rx) * Q0[j] * normalalpha[n] * river_interpolant / x_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])

                                    # along-channel advection
                                    a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,l) * umom_testfunctions[p][l] * G1(m,n,p) * Q_trialfunctions[i] * normalalpha[m] * river_interpolant * ngsolve.grad(alpha0[n][j])[0] / x_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                                    a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,l) * umom_testfunctions[p][l] * G1(m,n,p) * Q_trialfunctions[i] * normalalpha[m] * river_interpolant * Q0[j] * normalalpha[n] * river_interpolant_x / x_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                                    a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,l) * umom_testfunctions[p][l] * G1(m,n,p) * alpha_trialfunctions[m][i] * Q0[j] * normalalpha[n] * river_interpolant_x / x_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                                    
                                    a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,l) * vmom_testfunctions[p][l] * G1(m,n,p) * Q_trialfunctions[i] * normalalpha[m] * river_interpolant * ngsolve.grad(beta0[n][j])[0] / x_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                                    # lateral advection
                                    a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,l) * umom_testfunctions[p][l] * G1(m,n,p) * beta_trialfunctions[m][i] * Q0[j] * normalalpha_y[n] * river_interpolant / y_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                                    # vertical advection
                                    a += sum([sum([advection_epsilon * ramp * H3(i,j,l) * G2(m,n,p) * umom_testfunctions[p][l] * Q_trialfunctions[i] * normalalpha[m] * river_interpolant * (H+R) * ngsolve.grad(alpha0[n][j])[0] / x_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                                    a += sum([sum([advection_epsilon * ramp * H3(i,j,l) * G2(m,n,p) * umom_testfunctions[p][l] * Q_trialfunctions[i] * normalalpha[m] * river_interpolant * (Hx+Rx) * alpha0[n][j][0] / x_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                                    a += sum([sum([advection_epsilon * ramp * H3(i,j,l) * G2(m,n,p) * umom_testfunctions[p][l] * alpha_trialfunctions[m][i] * (H+R) * Q0[j] * normalalpha[n] * river_interpolant_x / x_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                                    a += sum([sum([advection_epsilon * ramp * H3(i,j,l) * G2(m,n,p) * umom_testfunctions[p][l] * alpha_trialfunctions[m][i] * (Hx+Rx) * Q0[j] * normalalpha[n] * river_interpolant / x_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                                    a += sum([sum([advection_epsilon * ramp * H3(i,j,l) * G2(m,n,p) * umom_testfunctions[p][l] * Q_trialfunctions[i] * normalalpha[m] * river_interpolant * (H+R) * Q0[j] * normalalpha[n] * river_interpolant_x / x_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                                    a += sum([sum([advection_epsilon * ramp * H3(i,j,l) * G2(m,n,p) * umom_testfunctions[p][l] * Q_trialfunctions[i] * normalalpha[m] * river_interpolant * (Hx+Rx) * Q0[j] * normalalpha[n] * river_interpolant / x_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                                    a += sum([sum([advection_epsilon * ramp * H3(i,j,l) * G2(m,n,p) * umom_testfunctions[p][l] * Q_trialfunctions[i] * normalalpha[m] * river_interpolant * (H+R) * ngsolve.grad(beta0[n][j])[1] / y_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                                    a += sum([sum([advection_epsilon * ramp * H3(i,j,l) * G2(m,n,p) * umom_testfunctions[p][l] * Q_trialfunctions[i] * normalalpha[m] * river_interpolant * (Hy+Ry) * beta0[n][j]/ y_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                                    a += sum([sum([advection_epsilon * ramp * H3(i,j,l) * G2(m,n,p) * vmom_testfunctions[p][l] * beta_trialfunctions[m][i] * (H+R) * Q0[j] * normalalpha[n] * river_interpolant_x / x_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                                    a += sum([sum([advection_epsilon * ramp * H3(i,j,l) * G2(m,n,p) * vmom_testfunctions[p][l] * beta_trialfunctions[m][i] * (Hx+Rx) * Q0[j] * normalalpha[n] * river_interpolant / x_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])

                                    # Next where the first variable is replaced by the previous iterate
                                    # along-channel advection
                                    a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,-l) * umom_testfunctions[p][-l] * G1(m,n,p) * Q0[i] * normalalpha[m] * river_interpolant * ngsolve.grad(alpha_trialfunctions[n][j])[0] / x_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                                    a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,-l) * umom_testfunctions[p][-l] * G1(m,n,p) * Q0[i] * normalalpha[m] * river_interpolant * Q_trialfunctions[j] * normalalpha[n] * river_interpolant_x / x_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                                    a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,-l) * umom_testfunctions[p][-l] * G1(m,n,p) * alpha0[m][i] * Q_trialfunctions[j] * normalalpha[n] * river_interpolant_x / x_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                                    
                                    a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,-l) * vmom_testfunctions[p][-l] * G1(m,n,p) * Q0[i] * normalalpha[m] * river_interpolant * ngsolve.grad(beta_trialfunctions[n][j])[0] / x_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                                    # lateral advection
                                    a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,-l) * umom_testfunctions[p][-l] * G1(m,n,p) * beta0[m][i] * Q_trialfunctions[j] * normalalpha_y[n] * river_interpolant / y_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                                    # vertical advection
                                    a += sum([sum([advection_epsilon * ramp * H3(i,j,-l) * G2(m,n,p) * umom_testfunctions[p][-l] * Q0[i] * normalalpha[m] * river_interpolant * (H+R) * ngsolve.grad(alpha_trialfunctions[n][j])[0] / x_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                                    a += sum([sum([advection_epsilon * ramp * H3(i,j,-l) * G2(m,n,p) * umom_testfunctions[p][-l] * Q0[i] * normalalpha[m] * river_interpolant * (Hx+Rx) * alpha_trialfunctions[n][j][0] / x_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                                    a += sum([sum([advection_epsilon * ramp * H3(i,j,-l) * G2(m,n,p) * umom_testfunctions[p][-l] * alpha0[m][i] * (H+R) * Q_trialfunctions[j] * normalalpha[n] * river_interpolant_x / x_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                                    a += sum([sum([advection_epsilon * ramp * H3(i,j,-l) * G2(m,n,p) * umom_testfunctions[p][-l] * alpha0[m][i] * (Hx+Rx) * Q_trialfunctions[j] * normalalpha[n] * river_interpolant / x_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                                    a += sum([sum([advection_epsilon * ramp * H3(i,j,-l) * G2(m,n,p) * umom_testfunctions[p][-l] * Q0[i] * normalalpha[m] * river_interpolant * (H+R) * Q_trialfunctions[j] * normalalpha[n] * river_interpolant_x / x_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                                    a += sum([sum([advection_epsilon * ramp * H3(i,j,-l) * G2(m,n,p) * umom_testfunctions[p][-l] * Q0[i] * normalalpha[m] * river_interpolant * (Hx+Rx) * Q_trialfunctions[j] * normalalpha[n] * river_interpolant / x_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                                    a += sum([sum([advection_epsilon * ramp * H3(i,j,-l) * G2(m,n,p) * umom_testfunctions[p][-l] * Q0[i] * normalalpha[m] * river_interpolant * (H+R) * ngsolve.grad(beta_trialfunctions[n][j])[1] / y_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                                    a += sum([sum([advection_epsilon * ramp * H3(i,j,-l) * G2(m,n,p) * umom_testfunctions[p][-l] * Q0[i] * normalalpha[m] * river_interpolant * (Hy+Ry) * beta_trialfunctions[n][j]/ y_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                                    a += sum([sum([advection_epsilon * ramp * H3(i,j,-l) * G2(m,n,p) * vmom_testfunctions[p][-l] * beta0[m][i] * (H+R) * Q_trialfunctions[j] * normalalpha[n] * river_interpolant_x / x_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                                    a += sum([sum([advection_epsilon * ramp * H3(i,j,-l) * G2(m,n,p) * vmom_testfunctions[p][-l] * beta0[m][i] * (Hx+Rx) * Q_trialfunctions[j] * normalalpha[n] * river_interpolant / x_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])

                                    # along-channel advection
                                    a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,l) * umom_testfunctions[p][l] * G1(m,n,p) * Q0[i] * normalalpha[m] * river_interpolant * ngsolve.grad(alpha_trialfunctions[n][j])[0] / x_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                                    a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,l) * umom_testfunctions[p][l] * G1(m,n,p) * Q0[i] * normalalpha[m] * river_interpolant * Q_trialfunctions[j] * normalalpha[n] * river_interpolant_x / x_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                                    a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,l) * umom_testfunctions[p][l] * G1(m,n,p) * alpha0[m][i] * Q_trialfunctions[j] * normalalpha[n] * river_interpolant_x / x_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                                    
                                    a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,l) * vmom_testfunctions[p][l] * G1(m,n,p) * Q0[i] * normalalpha[m] * river_interpolant * ngsolve.grad(beta_trialfunctions[n][j])[0] / x_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                                    # lateral advection
                                    a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,l) * umom_testfunctions[p][l] * G1(m,n,p) * beta0[m][i] * Q_trialfunctions[j] * normalalpha_y[n] * river_interpolant / y_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                                    # vertical advection
                                    a += sum([sum([advection_epsilon * ramp * H3(i,j,l) * G2(m,n,p) * umom_testfunctions[p][l] * Q0[i] * normalalpha[m] * river_interpolant * (H+R) * ngsolve.grad(alpha_trialfunctions[n][j])[0] / x_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                                    a += sum([sum([advection_epsilon * ramp * H3(i,j,l) * G2(m,n,p) * umom_testfunctions[p][l] * Q0[i] * normalalpha[m] * river_interpolant * (Hx+Rx) * alpha_trialfunctions[n][j][0] / x_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                                    a += sum([sum([advection_epsilon * ramp * H3(i,j,l) * G2(m,n,p) * umom_testfunctions[p][l] * alpha0[m][i] * (H+R) * Q_trialfunctions[j] * normalalpha[n] * river_interpolant_x / x_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                                    a += sum([sum([advection_epsilon * ramp * H3(i,j,l) * G2(m,n,p) * umom_testfunctions[p][l] * alpha0[m][i] * (Hx+Rx) * Q_trialfunctions[j] * normalalpha[n] * river_interpolant / x_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                                    a += sum([sum([advection_epsilon * ramp * H3(i,j,l) * G2(m,n,p) * umom_testfunctions[p][l] * Q0[i] * normalalpha[m] * river_interpolant * (H+R) * Q_trialfunctions[j] * normalalpha[n] * river_interpolant_x / x_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                                    a += sum([sum([advection_epsilon * ramp * H3(i,j,l) * G2(m,n,p) * umom_testfunctions[p][l] * Q0[i] * normalalpha[m] * river_interpolant * (Hx+Rx) * Q_trialfunctions[j] * normalalpha[n] * river_interpolant / x_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                                    a += sum([sum([advection_epsilon * ramp * H3(i,j,l) * G2(m,n,p) * umom_testfunctions[p][l] * Q0[i] * normalalpha[m] * river_interpolant * (H+R) * ngsolve.grad(beta_trialfunctions[n][j])[1] / y_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                                    a += sum([sum([advection_epsilon * ramp * H3(i,j,l) * G2(m,n,p) * umom_testfunctions[p][l] * Q0[i] * normalalpha[m] * river_interpolant * (Hy+Ry) * beta_trialfunctions[n][j]/ y_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                                    a += sum([sum([advection_epsilon * ramp * H3(i,j,l) * G2(m,n,p) * vmom_testfunctions[p][l] * beta0[m][i] * (H+R) * Q_trialfunctions[j] * normalalpha[n] * river_interpolant_x / x_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                                    a += sum([sum([advection_epsilon * ramp * H3(i,j,l) * G2(m,n,p) * vmom_testfunctions[p][l] * beta0[m][i] * (Hx+Rx) * Q_trialfunctions[j] * normalalpha[n] * river_interpolant / x_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
  

     