"""This file contains functions to generate along and across channel coordinates
23-12-2022: Added early continue"""

import ngsolve

from NiFlow.geometry import create_geometry as cg


def count_free_dofs(fes: ngsolve.H1):
    i = 0
    for isFree in fes.FreeDofs():
        i = i + isFree
    return i


def generate_bfc(mesh, order, method, alpha=1.0):
    """Wrapper to generate the boundary fitted coordinates
    laplace method:     Generate bfc by solving a Laplace equation once
    diffusion method:   Generate bfc by first solving a Laplace equation and then a diffusion equation with
                        fitting parameter alpha
    Args:
    - mesh: the finite element mesh
    - order: the order of the finite elements for the bfc
    - method: string with the method: "laplace" or "iterative"
    - alpha: constant fitting parameter for the diffusivity method
    """
    if method == "laplace":
        xi = along_channel_coordinates(mesh, order)
        eta = across_channel_coordinates(mesh, order)
    elif method == "diffusion":
        xi, eta = along_and_across_channel_coordinates(mesh, order, alpha)
    else:
        raise Exception("Boundary fitted coordinates method not recognised, please use \"laplace\" or \"diffusion\".")

    return BoundaryFittedCoordinates(xi, eta)


def along_channel_coordinates(mesh, order):
    """Generates along channel coordinates

    These coordinates are generated by solving a Laplace equation.

    Solve:               int_Omega -Grad(phi)*Grad(xi) dOmega = 0
    Boundary conditions: xi=0 at sea, xi=1 at river, xi_n = 0 at wall up and wall down"""

    # FEM space
    fes = ngsolve.H1(mesh, order=order, dirichlet=cg.BOUNDARY_DICT[cg.SEA] + "|" + cg.BOUNDARY_DICT[cg.RIVER])

    # Solution variable
    xi = ngsolve.GridFunction(fes, name="along_channel_coordinate")

    # Boundary conditions
    boundary_values = {cg.BOUNDARY_DICT[cg.SEA]: 0, cg.BOUNDARY_DICT[cg.RIVER]: 1}
    values_list = [boundary_values[boundary]
                   if boundary in boundary_values else 0
                   for boundary in mesh.GetBoundaries()]
    xi.Set(ngsolve.CoefficientFunction(values_list), ngsolve.BND)

    # FEM formulation
    u = fes.TrialFunction()
    phi = fes.TestFunction()

    a = ngsolve.BilinearForm(fes, symmetric=True)
    a += - ngsolve.Grad(u) * ngsolve.Grad(phi) * ngsolve.dx

    f = ngsolve.LinearForm(fes)

    # print("DOFS: ", count_free_dofs(fes))

    # Solve for xi
    ngsolve.solvers.BVP(bf=a, lf=f, gf=xi, inverse='pardiso')

    return xi


def across_channel_coordinates(mesh, order):
    """Generates across channel coordinates

    These coordinates are generated by solving a Laplace equation.

    Solve:               int_Omega -Grad(phi)*Grad(eta) dOmega = 0
    Boundary conditions: eta_n=0 at sea and river, eta=-1 at wall down, eta=1 at wall up"""

    # FEM space
    fes = ngsolve.H1(mesh, order=order, dirichlet=cg.BOUNDARY_DICT[cg.WALLDOWN] + "|" + cg.BOUNDARY_DICT[cg.WALLUP])

    # Solution variable
    eta = ngsolve.GridFunction(fes, name="across_channel_coordinate")

    # Boundary conditions
    boundary_values = {cg.BOUNDARY_DICT[cg.WALLDOWN]: -1, cg.BOUNDARY_DICT[cg.WALLUP]: 1}
    values_list = [boundary_values[boundary]
                   if boundary in boundary_values else 0
                   for boundary in mesh.GetBoundaries()]
    eta.Set(ngsolve.CoefficientFunction(values_list), ngsolve.BND)

    # FEM fomulation
    u = fes.TrialFunction()
    phi = fes.TestFunction()

    a = ngsolve.BilinearForm(fes, symmetric=True)
    a += - ngsolve.Grad(u) * ngsolve.Grad(phi) * ngsolve.dx

    f = ngsolve.LinearForm(fes)

    # print("DOFS: ", count_free_dofs(fes))

    # Solve for eta
    ngsolve.solvers.BVP(bf=a, lf=f, gf=eta, inverse='pardiso')

    return eta


def along_and_across_channel_coordinates(mesh, order, alpha):
    """Generates along and across channel coordinates

    These coordinates are generated by solving a diffusion style equations.


    Solve:               int_Omega -Innerproduct(Grad(phi),D*Grad(xi)) dOmega = 0
        Boundary conditions: xi=0 at sea, xi=1 at river, xi_n = 0 at wall up and wall down

    and

    Solve:               int_Omega -Innerproduct(Grad(phi),D*Grad(eta)) dOmega = 0
       Boundary conditions: detadn=0 at sea and river, eta=-1 at wall down, eta=1 at wall up

    Parameters:
        mesh: ngsolve mesh object
        order: order of the basis functions
        alpha: fitting parameter to make the along-channel coordinate represent a real distance, typically alpha=1 works
                well
    """

    # FEM spaces
    fes_xi = ngsolve.H1(mesh, order=order, dirichlet=cg.BOUNDARY_DICT[cg.SEA] + "|" + cg.BOUNDARY_DICT[cg.RIVER])
    fes_eta = ngsolve.H1(mesh, order=order, dirichlet=cg.BOUNDARY_DICT[cg.WALLDOWN] + "|" + cg.BOUNDARY_DICT[cg.WALLUP])

    # Boundary conditions
    boundary_values_xi = {cg.BOUNDARY_DICT[cg.SEA]: 0, cg.BOUNDARY_DICT[cg.RIVER]: 1}
    values_list_xi = [boundary_values_xi[boundary]
                   if boundary in boundary_values_xi else 0
                   for boundary in mesh.GetBoundaries()]

    boundary_values_eta = {cg.BOUNDARY_DICT[cg.WALLDOWN]: -1, cg.BOUNDARY_DICT[cg.WALLUP]: 1}
    values_list_eta = [boundary_values_eta[boundary]
                   if boundary in boundary_values_eta else 0
                   for boundary in mesh.GetBoundaries()]

    # FEM formulation
    u_xi = fes_xi.TrialFunction()
    phi_xi = fes_xi.TestFunction()

    u_eta = fes_eta.TrialFunction()
    phi_eta = fes_eta.TestFunction()




    xi_gf = {}
    eta_gf = {}
    for n in range(2):
        if n == 0:
            D = 1
        else:
            D = ngsolve.Norm(ngsolve.Grad(xi_gf[n-1])) ** alpha


        # The xi_gf problem
        a_xi = ngsolve.BilinearForm(fes_xi, symmetric=True)
        a_xi += -ngsolve.InnerProduct(ngsolve.Grad(u_xi), D * ngsolve.Grad(phi_xi)) * ngsolve.dx
        f_xi = ngsolve.LinearForm(fes_xi)
        # print("DOFS xi: ", pp.count_free_dofs(fes_xi))

        # Along channel coordinate
        xi_gf[n] = ngsolve.GridFunction(fes_xi, name="along_channel_coordinate")
        xi_gf[n].Set(ngsolve.CoefficientFunction(values_list_xi), ngsolve.BND)

        # Solve for xi_gf[n]
        ngsolve.solvers.BVP(bf=a_xi, lf=f_xi, gf=xi_gf[n], inverse="pardiso")

        # For the initial run only xi is needed for the diffusion coefficient computation, thus we continue early
        if n==0:
            continue

        ## The eta_gf problem
        a_eta = ngsolve.BilinearForm(fes_eta, symmetric=True)
        a_eta += - ngsolve.InnerProduct(ngsolve.Grad(u_eta), D*ngsolve.Grad(phi_eta)) * ngsolve.dx
        f_eta = ngsolve.LinearForm(fes_eta)
        # print("DOFS eta: ", pp.count_free_dofs(fes_eta))

        # Across channel coordinate
        eta_gf[n] = ngsolve.GridFunction(fes_eta, name="across_channel_coordinate")
        eta_gf[n].Set(ngsolve.CoefficientFunction(values_list_eta), ngsolve.BND)

        # Solve for eta_gf[n]
        ngsolve.solvers.BVP(bf=a_eta, lf=f_eta, gf=eta_gf[n], inverse="pardiso")

    return xi_gf[n], eta_gf[n]


class BoundaryFittedCoordinates:
    """
    Simple class to store the boundary fitted coordintes
    """

    def __init__(self, xi_gf, eta_gf):
        """ Set the xi and eta grid functions """
        self.xi_gf = xi_gf
        self.eta_gf = eta_gf
        self.h1, self.h2 = self.compute_h1_h2()

    # We make functions to compute the scale factors
    def compute_Jinv(self):
        """
        Returns:
            Jinv: the inverse Jacobian of the transformation
        """
        # Create the defined Jinv
        Jinv = ngsolve.CoefficientFunction((ngsolve.Grad(self.xi_gf), ngsolve.Grad(self.eta_gf)), dims=(2, 2))
        return Jinv



    def compute_h1_h2(self):
        """
        Returns:
            h1, h2: tuple of scale factors h1 and h2
        """
        Jinv = self.compute_Jinv()
        abs_det_Jinv = ngsolve.Norm(ngsolve.Det(Jinv))

        h1 = 1/abs_det_Jinv * ngsolve.Norm(ngsolve.Grad(self.eta_gf))
        h2 = 1/abs_det_Jinv * ngsolve.Norm(ngsolve.Grad(self.xi_gf))
        return h1, h2