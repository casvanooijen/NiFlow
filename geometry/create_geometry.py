""""
File used to generate a NGSolve geometry object

In this file, four methods are defined to create a NGSolve geometry object:
  1) parametric_geometry      -- Given parametric curves, use these curves as a boundary
  2) parametric_wall_geometry -- Given two parametric wall curves, use these curves as wall boundary and automatically generate the sea and river boundaries
  3) general_spline_geometry  -- Given boundary points, fit a spline and use it as a boundary
  4) general_spline3_geometry -- Given boundary points, fit a spline, create smaller NGSolve spline3 curves and use these as boundary

TODO make an object of this script, such that we can more easily work with its properties
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import netgen.geom2d as geom2d
import ngsolve

SEA = 1
WALL = 2
RIVER = 3
WALLDOWN = 4
WALLUP = 5
BOUNDARY_DICT = {1: "sea", 2: "wall", 3: "river", 4: "walldown", 5: "wallup"}

POINTS_ORGINAL = 1
POINTS_UNIFORMPOINTS = 2
POINTS_NORMFIRSTDERIVATIVE = 3

########## Parametric methods (1,2) ###########
def parametric_geometry(geometrycurves, boundary_parameter_partitition_dict=None, boundary_maxh_dict=None):
    """
    Creates a NGSolve geometry based on parametric curves.
    The curves parameterise the geometry in a counterclockwise fashion.

    The curves should enclose an area.
    The distance between a begin and end point is maximal 1e-5.
    Else the meshing fails.

    # 2023-02-06: MPR: Added boundary_parameter_partitition_dict and boundary_maxh_dict

    Arguments:
        geometrycurves -- list of parametric curves f(t) = [x, y] with the parameter t between 0 and 1 and
                          the boundary condition [f(t), bc_type]
        boundary_parameter_partitition_dict -- dictionary containing the partition of the parameter per boundary condition.
                                               For example, {cg.WALLDOWN: [0, 0.5, 1]}
        boundary_maxh_dict -- dictionary containing the maximal grid spacing for each boundary partition per boundary condition.
                              For example, {cg.WALLDOWN: [2e3, 1e3]}
    """

    curve_list = [curve for curve, _ in geometrycurves]
    boundary_condition_list = [boundary_condition for _, boundary_condition in geometrycurves]

    # Create geometry
    geometry = geom2d.SplineGeometry()

    append_splines_to_geometry(curve_list, boundary_condition_list, geometry, boundary_parameter_partitition_dict, boundary_maxh_dict)

    points = [] # list used to prevent duplicate points from being added
    for curve in curve_list:
        can_we_add = True
        for point in points:
            if abs(curve(0)[0] - point[0]) < 1e-12 and abs(curve(0)[1] - point[1]) < 1e-12:
                can_we_add = False
                break
        if can_we_add:
            geometry.AppendPoint(curve(0)[0], curve(0)[1])
            points.append([curve(0)[0], curve(0)[1]])

        can_we_add = True
        for point in points:
            if abs(curve(1)[0] - point[0]) < 1e-12 and abs(curve(1)[1] - point[1]) < 1e-12:
                can_we_add = False
                break
        if can_we_add:
            geometry.AppendPoint(curve(1)[0], curve(1)[1])
            points.append([curve(1)[0], curve(1)[1]])


    return geometry


def parametric_wall_geometry(geometrywallcurves):
    """
    Create NGSolve geometry based on two wall curves.
    The other boundaries are a straight river and sea boundary, defined between the end points of the wall curves.

    This function is a simplified interface for the parameteric_geometry function.
    Arguments:
        geometrywallcurves -- list of two parametric curves f(t) = np.array([x, y]) with the parameter t between 0 and 1
                              Describing the bottom and top wall of the geometry.
    """
    wallbottom, walltop = geometrywallcurves

    def river(t):
        return wallbottom(1) + (walltop(0)-wallbottom(1))*t

    def sea(t):
        return walltop(1) + (wallbottom(0)-walltop(1))*t

    geometrycurves = [[wallbottom, WALL], [river, RIVER], [walltop, WALL], [sea, SEA]]
    geometry = parametric_geometry(geometrycurves)

    return geometry


def debug_parametric_geometry(geometrycurves):
    """
    Plot to debug the parameteric curves.

    Arguments:
        geometrycurves -- list of parametric curves f(t) = [x, y] with the parameter t between 0 and 1 and
                          the boundary condition [f(t), bc_type]
    """
    curve_list = [curve for curve, _ in geometrycurves]
    t = np.linspace(0, 1, 1000)
    for curve in curve_list:
        line = np.array([curve(t_) for t_ in t])
        plt.plot(line[:, 0], line[:, 1])
    plt.gca().axis('equal')
    return plt

############ Spline methods (3,4) ###############
def general_spline_geometry(geometrydata, degree_spline_fit, smoothness_spline_fit,
                            boundary_parameter_partitition_dict=None, boundary_maxh_dict=None):
    """
    Creates a piecewise smooth spline geometry using counter clockwise geometry data.
    The spline interpolator of the geometry data per segment is used directly as boundary.

    The geometry data is split into segments based on the boundary condition type.
    For each segment, a spline interpolater of the geometry data is created.
    The spine interpolator is directly used as the boundary curve for each segment.
    The boundary parameter and boundary maxh dict are related.
    Arguments:
        geometrydata -- numpy array with [[x1, y2, bc_type1], [x2, y2, bc_type2], ... ] structure
        degree_spline_fit -- polynomial degree of spline fit
        smoothness_spline_fit -- smoothness_spline_fit of spline fit
        boundary_parameter_partitition_dict -- dictionary containing the partition of the parameter per boundary condition.
                                               For example, {cg.WALLDOWN: [0, 0.5, 1]}
        boundary_maxh_dict -- dictionary containing the maximal grid spacing for each boundary partition per boundary condition.
                              For example, {cg.WALLDOWN: [2e3, 1e3]}
    """

    # Split geometrydata based on bc_type
    points_segment_list, boundary_conditions_list = split_on_boundary_condition_type(geometrydata)

    # Spline interpolation per segment
    spline_list = spline_segments(points_segment_list, degree_spline_fit, smoothness_spline_fit)

    # Create geometry
    geometry = geom2d.SplineGeometry()

    # Append scaled splines to geometry
    append_splines_to_geometry(spline_list, boundary_conditions_list, geometry, boundary_parameter_partitition_dict, boundary_maxh_dict)

    return geometry


def general_spline_geometry_automatic(geometrydata, degree_spline_fit, smoothness_spline_fit, cells_per_crossection):
    """
    Creates a piecewise smooth spline geometry using counterclockwise geometry data.
    The spline interpolator of the geometry data per segment is used directly as boundary.

    The geometry data is split into segments based on the boundary condition type.
    For each segment, a spline interpolater of the geometry data is created.
    The spine interpolator is directly used as the boundary curve for each segment.
    The boundary parameter and boundary maxh dict are related.

    Difference with above: the estimate of the maxh is determined based on a estimate of the width
    Arguments:
        geometrydata -- numpy array with [[x1, y2, bc_type1], [x2, y2, bc_type2], ... ] structure
        degree_spline_fit -- polynomial degree of spline fit
        smoothness_spline_fit -- smoothness_spline_fit of spline fit
        boundary_parameter_partitition_dict -- dictionary containing the partition of the parameter per boundary condition.
                                               For example, {cg.WALLDOWN: [0, 0.5, 1]}
        boundary_maxh_dict -- dictionary containing the maximal grid spacing for each boundary partition per boundary condition.
                              For example, {cg.WALLDOWN: [2e3, 1e3]}
    """

    # Split geometrydata based on bc_type
    points_segment_list, boundary_conditions_list = split_on_boundary_condition_type(geometrydata)

    # Spline interpolation per segment
    spline_list = spline_segments(points_segment_list, degree_spline_fit, smoothness_spline_fit)

    boundary_parameter_partitition_dict, boundary_maxh_dict = generate_partition_and_maxh_dict(spline_list, boundary_conditions_list, cells_per_crossection)

    # Print automatic subdivision
    print("boundary_maxh_dict", boundary_maxh_dict)

    # Create geometry
    geometry = geom2d.SplineGeometry()

    # Append scaled splines to geometry
    append_splines_to_geometry(spline_list, boundary_conditions_list, geometry, boundary_parameter_partitition_dict, boundary_maxh_dict)

    return geometry


# TODO write
def general_spline_geometry_interpolated(geometrydata_1, geometrydata_2, alpha_interpolate, degree_spline_fit, smoothness_spline_fit, cells_per_crossection):
    """Function to interpolate between geometrydata1 and geometrydata2 based on alpha_interpolate"""

    # We create two splines fits and lerp between them
    # Split geometrydata based on bc_type
    points_segment_list_1, boundary_conditions_list_1 = split_on_boundary_condition_type(geometrydata_1)
    points_segment_list_2, boundary_conditions_list_2 = split_on_boundary_condition_type(geometrydata_2)

    # Spline interpolation per segment
    spline_list_1 = spline_segments(points_segment_list_1, degree_spline_fit, smoothness_spline_fit)
    spline_list_2 = spline_segments(points_segment_list_2, degree_spline_fit, smoothness_spline_fit)

    # Check if same boundary condition list, else raise error
    if boundary_conditions_list_1 == boundary_conditions_list_2:
        boundary_conditions_list = boundary_conditions_list_1
    else:
        raise Exception("Boundary list of geometry 1 does not match boundary condition list of geometry 2. Cannot perform lerp.")

    # Lerp between spline segments list
    spline_list_lerp = [lambda t, spline_1=spline_1, spline_2=spline_2: (1-alpha_interpolate) * np.array(spline_1(t)) + alpha_interpolate * np.array(spline_2(t)) for spline_1, spline_2 in zip(spline_list_1, spline_list_2)]

    # Automatically generate partion and maxh
    boundary_parameter_partitition_dict, boundary_maxh_dict = generate_partition_and_maxh_dict(spline_list_lerp,
                                                                                               boundary_conditions_list,
                                                                                               cells_per_crossection)

    # Print automatic subdivision
    print("boundary_maxh_dict", boundary_maxh_dict)


    # Create geometry
    geometry = geom2d.SplineGeometry()

    # Append scaled splines to geometry
    append_splines_to_geometry(spline_list_lerp, boundary_conditions_list, geometry, boundary_parameter_partitition_dict, boundary_maxh_dict)

    return geometry



    # TODO
def generate_partition_and_maxh_dict(spline_list, boundary_conditions_list, cells_per_crossection):
    """ Function that automatically generates boundary_parameter_partitition_dict and boundary_maxh_dict based on width """
    # We estimate the width based on the spline list distance between wallup and walldown
    width_estimate = estimate_width(spline_list, boundary_conditions_list)

    # #TODO we create a simple boundary parmater partitiuon dict TODO CHECK
    # The boundary is subdivided into n segments, where a maxh h is prescribed
    n_segments = 10  # TODO was 10, 20 appears to be more unstable
    t_walldown = np.linspace(0, 1, n_segments)
    t_wallup = np.append(0, np.cumsum(np.flip(np.diff(t_walldown))))
    t_avg_walldown = (t_walldown[1:] + t_walldown[:-1]) / 2
    t_avg_wallup = np.flip((t_wallup[1:] + t_wallup[:-1]) / 2)

    t_river = [0, 1]

    boundary_parameter_partitition_dict = {WALLDOWN: t_walldown, WALLUP: t_wallup, RIVER: t_river}
    boundary_maxh_dict = {WALLDOWN: width_estimate(t_avg_walldown) / cells_per_crossection,
                          WALLUP: width_estimate(t_avg_wallup) / cells_per_crossection,
                          RIVER: [width_estimate(t_avg_walldown[-1]) / cells_per_crossection / 2]}
    return boundary_parameter_partitition_dict, boundary_maxh_dict


# TODO move
def toint(float):
    """Function that more robustly sets a float to an int"""
    return int(round(float))

# TODO move
def estimate_width(spline_list, boundary_conditions_list):
    """Function that estimates the width of the channel. We assume a single channel only,
    with WALLDOWN and WALLUP as boundary condition types.

    Arg:
        spline_list - list of splines
    Returns:
        estimation of width as function of t
    """
    smoothness_width = 2e8
    min_width = 1e-5

    def reverse_parametrisation(spline, t):
        """Function to reverse the parametrisation of a spline"""
        t_reversed = 1 - t
        return spline(t_reversed)

    # Seek WALLUP and DOWN
    for spline, boundary_condition in zip(spline_list, boundary_conditions_list):
        if boundary_condition == toint(WALLDOWN):
            spline_walldown = spline
        if boundary_condition == toint(WALLUP):
            spline_wallup = lambda t, spline=spline: reverse_parametrisation(spline, t)

    width_estimate = lambda t: np.linalg.norm(np.array(spline_wallup(t)) - np.array(spline_walldown(t)), 2, axis=0)

    # We smooth the given width using a spline fit (again)
    # We discretize the approximate width and fit a spline
    t_samples = np.linspace(0, 1, 200)
    width_estimate_sample = width_estimate(t_samples)

    # The last point is given more weight to make sure it is correctly approximated.
    # The *1 makes a boolean array a binary array
    weight_samples = np.ones_like(t_samples) + 1e4 * t_samples[(t_samples > 0.95)*1]

    width_estimate_smooth = interpolate.UnivariateSpline(t_samples, width_estimate_sample, weight_samples, s=smoothness_width)

    # Make sure the width estimate stays above zero
    min_width_sample = width_estimate_sample.min()
    width_estimate_positive = lambda t: np.maximum(width_estimate_smooth(t), np.maximum(min_width_sample, min_width))


    return width_estimate_positive

#TODO
def debug_width_estimate(width_estimate):
    """ Plotting the estimate of the width of the estuary """
    t = np.linspace(0, 1, 1000)
    plt.plot(t, width_estimate(t), label="Estimate of width")


def append_splines_to_geometry(spline_list, boundary_conditions_list, geometry, boundary_parameter_partitition_dict, boundary_maxh_dict):
    """
    Appends the spline segments to the geometry and sets boundary conditions

    Note: mutates geometry
    Arguments:
        spline_list                 -- list of the splines per segment
        boundary_conditions_list    -- list of the boundary conditions
        geometry                    -- the geometry object
        boundary_parameter_partitition_dict -- dictionary containing the partition of the parameter per boundary condition.
        boundary_maxh_dict                  -- dictionary containing the maximal grid spacing for boundary condition.
    """
    def rescale_spline(spline, t, t_left, t_right):
        """Rescales spline to the desired interval
        - t parameter [0,1]
        - t_left left bound
        - t_right right bound
        - Returns the rescaled spline"""

        t_scaled = t_left + (t_right - t_left) * t
        return spline(t_scaled)

    # Iterate over the splines
    for spline, boundary_condition in zip(spline_list, boundary_conditions_list):
        boundary_type = int(boundary_condition)
        # If spline is subdivided
        if boundary_maxh_dict is not None and boundary_type in boundary_maxh_dict:
            for i, maxh in enumerate(boundary_maxh_dict[boundary_type]):
                t_left, t_right = boundary_parameter_partitition_dict[boundary_type][i:i + 2]

                def scaled_spline(t):
                    return rescale_spline(spline, t, t_left, t_right)

                geometry.AddCurve(scaled_spline, bc=BOUNDARY_DICT[boundary_type], maxh=maxh)
        else:
            geometry.AddCurve(spline, bc=BOUNDARY_DICT[boundary_type])
    return None


def spline_segments(points_segment_list, degree_spline_fit, smoothness_spline_fit):
    """"
    Creates spline fit of segment

    Arguments:
        points_segment      -- numpy array with points along segment
        degree_spline_fit       -- polynomial degree of spline fit. Bounded between 1 and 5.
        smoothness_spline_fit         -- smoothness_spline_fit of spline fit
        includeEnds         -- boolean to determine to keep the ends or not
    """

    def spline_segment(t, knots_coefficients_degree):
        return interpolate.splev(t, knots_coefficients_degree)

    spline_segment_list = []
    for i, points_segment in enumerate(points_segment_list):
        # If there are not enough equations for the number of unknowns, use lower order polynomial
        if len(points_segment) <= degree_spline_fit:
            # Use highest possible approximation
            degree = len(points_segment)-1
        else:
            degree = degree_spline_fit

        knots_coefficients_degree, parameter_original_points = interpolate.splprep(points_segment.T,
                                                                                   k=degree, s=smoothness_spline_fit)
        # Trick to get the current knots_coefficients_degree in as argument
        spline_segment_list.append(lambda t, knots_coefficients_degree=knots_coefficients_degree:
                                   spline_segment(t, knots_coefficients_degree))

    return spline_segment_list


#################### Spline3 method ###################
def general_spline3_geometry(geometrydata, polynomial_degree, smoothness, method=POINTS_ORGINAL):
    """
    Creates a piecewise smooth spline geometry using counterclockwise geometry data.
    Between two data points a spline3 curve is used as boundary.

    The geometry data is split into segments based on the boundary condition type.
    For each segment, a spline interpolation of the geometry data is preformed.
    NGSolve's Append geometry only takes line and spline3 as input arguments.
    For spline3 the start and end point of the curve section are required
    as well as the bounding or intersection point of the derivatives in these points.
    Lastly, all the points are added to the geometry and the line or spline3 segments
    between these points are introduced.
    Arguments:
        geometrydata -- numpy array with [x, y, bc_type] structure
        degree_spline_fit -- integer with the desired order of the polynomial fit
        smoothness_spline_fit -- float to indicate the smoothness_spline_fit of the fit
        method -- integer to indicate which method to use to generate the NGSolve spline3 curves from the spline fit
    """

    # Split geometrydata based on bc_type
    points_segment_list, boundary_conditions_list = split_on_boundary_condition_type(geometrydata)

    # Spline fit each segment data
    points_spline_list, points_intersection_list = spline_and_intersection_points(
        points_segment_list, polynomial_degree, smoothness, method)

    # Create geometry
    geometry = geom2d.SplineGeometry()

    points_spline_index_list = append_points_list_to_geometry(points_spline_list, geometry, True)
    points_intersection_index_list = append_points_list_to_geometry(points_intersection_list, geometry, False)

    append_segments_to_geometry(points_spline_index_list, points_intersection_index_list, boundary_conditions_list, geometry)

    return geometry 


def append_segments_to_geometry(points_spline_index_list, points_intersection_index_list, boundary_conditions_list, geometry):
    """"
    Append the curver defined on each segment to the geomentry object

    Walks through each section and appends corresponding curves to the geometry.
    Note: Mutates geometry
    Arguments:
        points_spline_index_list           -- list consisting of the spline point indeces per segment
        points_intersection_index_list     -- list consisting of the intersection point indeces per segment
        boundary_conditions_list           -- list of the boundary conditions
        geometry                           -- geometry object
    """

    for points_spline_index, points_intersection_index, boundary_condition in zip(
            points_spline_index_list, points_intersection_index_list, boundary_conditions_list):
        append_curves(points_spline_index, points_intersection_index, boundary_condition, geometry)
    return None


def append_curves(points_spline_index, points_intersection_index, boundary_condition, geometry):
    """"
    Appends curves to the geometry object

    Arguments:
        points_spline_index           -- the spline point indeces per segment
        points_intersection_index     -- the intersection point indeces per segment
        boundary_condition            -- boundary condition
        geometry                      -- geometry object
    """

    if len(points_spline_index) == len(points_intersection_index)+1:
        # Append splines
        [geometry.Append(["spline3", point_spline_index1, point_intersection_index, point_spline_index2], bc=BOUNDARY_DICT[int(boundary_condition)]) for
         point_spline_index1, point_intersection_index, point_spline_index2 in
         zip(points_spline_index[:-1], points_intersection_index, points_spline_index[1:])]
    else:
        # Append straight lines
        [geometry.Append(["line", point_spline_index1, point_spline_index2], bc=BOUNDARY_DICT[int(boundary_condition)]) for
         point_spline_index1, point_spline_index2 in
         zip(points_spline_index[:-1], points_spline_index[1:])]
    return None


def append_points_list_to_geometry(points_list, geometry, is_cyclic):
    """"
    Appends unique points to geometry per segment

    We walk through the each point in the list and add it to the geometry, so it knowns that it exists.
    Note: 1) the last point of each segment is excluded because else we would have double points
          2) mutates geometry
    Arguments:
        points_list -- list consisting of the points per segment
        geometry    -- geometry object
        is_cyclic  -- is the end of each segment included
    """

    def rotate(lst, n):
        return lst[n:] + lst[:n]

    points_index_list = []

    if is_cyclic:
        points_index_unique_list = []
        for points in points_list:
            points_index_unique = [geometry.AppendPoint(*point) for point in points[:-1, :]]
            points_index_unique_list.append(points_index_unique)

        # Add reference to last point in segment back
        for points_index_unique1, points_index_unique2 in zip(points_index_unique_list, rotate(points_index_unique_list, 1)):
            points_index = []
            points_index.extend(points_index_unique1)
            points_index.append(points_index_unique2[0])
            points_index_list.append(points_index)

    else:
        for points in points_list:
            points_index = [geometry.AppendPoint(*point) for point in points]
            points_index_list.append(points_index)

    return points_index_list


def split_on_boundary_condition_type(geometrydata):
    """"
    Splits geometry data into sections determined by the last column of geometrydata

    Creates a list of segments and list of boundary conditions based on the boundary condition type of geometry data.
    Each section consists of points which includes the begin and endpoints of the segment. The inclusion of end points
    is to prepare the data for a spline fit.
    Arguments:
        geometrydata (numpy array) -- matrix with the dimensions [x, y, bc_type]
    """

    points_geometry = geometrydata[:, 0:2].tolist()
    boundary_conditions_geometry = geometrydata[:, 2].tolist()

    # Create two lists: points_segment_list = [p_BC1, P_BC2, P_BC3, ...]
    # and boundary_conditions_list = [BC1, BC2, BC3, ...] which are directly linked to the boundary condition type
    points_segment_list = []
    points_segment = []
    boundary_conditions_list = [boundary_conditions_geometry[0]]

    for i, boundary_condition in enumerate(boundary_conditions_geometry):

        points_segment.append(points_geometry[i])

        # If the current boundary condition is different then
        if boundary_condition != boundary_conditions_list[-1]:
            boundary_conditions_list.append(boundary_condition)
            points_segment_list.append(np.array(points_segment))
            points_segment = [points_geometry[i]]

            # and if last element
            if i + 1 == len(boundary_conditions_geometry):
                points_segment.append(points_geometry[0])
                points_segment_list.append(np.array(points_segment))

        # if boundary condition is the same and last element
        elif i + 1 == len(boundary_conditions_geometry):
            points_segment.append(points_geometry[0])
            points_segment_list.append(np.array(points_segment))

    return points_segment_list, boundary_conditions_list


def spline_and_intersection_points(points_segment_list, polynomial_degree, smoothness, method):
    """"
    Returns spline and intersection points of all segments

     Arguments:
        points_segment_list -- list of numpy array of points of each segment
        degree_spline_fit   -- polynomial degree of spline fit. Bounded between 1 and 5.
        smoothness_spline_fit          -- smoothness_spline_fit of spline fit
    """
    points_spline_list = []
    points_intersection_list = []
    for points_segment in points_segment_list:
        points_spline, points_intersection = spline_and_intersection_points_segment(
            points_segment, polynomial_degree, smoothness, method, includeEnds=True)

        points_spline_list.append(points_spline)
        points_intersection_list.append(points_intersection)
    return points_spline_list, points_intersection_list


def spline_and_intersection_points_segment(points_segment, degree_spline_fit, smoothness_spline_fit, method, includeEnds=True):
    """"
    Creates spline and intersection points based on spline fit

    Arguments:
        points_segment      -- numpy array with points along segment
        degree_spline_fit   -- polynomial degree of spline fit. Bounded between 1 and 5.
        smoothness_spline_fit          -- smoothness_spline_fit of spline fit
        includeEnds         -- boolean to determine to keep the ends or not
    """

    # If there are not enough equations for the number of unknowns, use a straight line
    if len(points_segment) <= degree_spline_fit or degree_spline_fit == 1:
        points_spline = points_segment
        points_intersection = []
    else:
        if method == POINTS_ORGINAL:
            knots_coefficients_degree, parameter_original_points = interpolate.splprep(points_segment.T, k=degree_spline_fit, s=smoothness_spline_fit)
            points_spline, points_intersection = refine_points(parameter_original_points, knots_coefficients_degree)

        elif method == POINTS_UNIFORMPOINTS:
            knots_coefficients_degree, parameter_original_points = interpolate.splprep(points_segment.T,
                                                                                       k=degree_spline_fit,
                                                                                       s=smoothness_spline_fit)
            parameter_uniform = np.linspace(0, 1, 100)
            points_spline, points_intersection = refine_points(parameter_uniform, knots_coefficients_degree)
        elif method == POINTS_NORMFIRSTDERIVATIVE:
            knots_coefficients_degree, parameter_original_points = interpolate.splprep(points_segment.T,
                                                                                       k=degree_spline_fit,
                                                                                       s=smoothness_spline_fit)
            # If you want to use this method you should generate a better guess for parameter_scaled.
            # Although, I advise to use the general_spline_geometry instead
            parameter_uniform = np.linspace(0, 1, 100)
            derivative_spline = interpolate.splev(parameter_uniform, knots_coefficients_degree, der=1)
            norm = np.linalg.norm(derivative_spline)
            parameter_scaled = parameter_uniform*norm

            points_spline, points_intersection = refine_points(parameter_scaled, knots_coefficients_degree)

    if includeEnds:
        return points_spline, points_intersection
    else:
        return points_spline[1:-1], points_intersection


def refine_points(parameter_original_points, knots_coefficients_degree):
    """
    Creates spline points and intersection points along the spline fit

    Takes two parameter values along the spline and checks if there is a valid intersection between them.
    It can be the case that the valid intersection has a new right parameter. As long as this new right parameter is
    unequal to the original right parameter, continue to keep track of the spline and
    intersection points. This process is know as the refinement.
    Arguments:
        parameter_original_points   -- List containing the parameter points of the original spline fit
        knots_coefficients_degree   -- knots, coefficients and polynomial degree of spline fit
    """
    points_spline = []
    points_intersection = []
    parameter_left = parameter_original_points[0]
    i = 1

    while i < len(parameter_original_points):
        parameter_right = parameter_original_points[i]

        point_intersection, parameter_intersection = find_valid_intersection(parameter_left, parameter_right, knots_coefficients_degree)

        points_spline.append(interpolate.splev(parameter_left, knots_coefficients_degree))
        points_intersection.append(point_intersection)

        parameter_left = parameter_intersection

        # If the intersection parameter equals the right parameter, then go on to next right parameter.
        if parameter_intersection == parameter_right:
            i += 1

    points_spline.append(interpolate.splev(parameter_left, knots_coefficients_degree))

    return np.array(points_spline), np.array(points_intersection)


def find_valid_intersection(parameter1, parameter2, knots_coefficients_degree):
    """
    Recursively finds a valid intersection of parametric derivative of spline fit

    A divide-and-conquer algorithm is applied. If there is no valid intersection point,
    then the second parameter is changed to the average of the first and second parameter
    while the first parameter is fixed. This method is somewhat similar to the bisection method.
    Arguments:
        parameter1                  --
        parameter2                  --
        knots_coefficients_degree   --
    """
    point_parameter1 = np.array(interpolate.splev(parameter1, knots_coefficients_degree))
    gradient_parameter1 = np.array(interpolate.splev(parameter1, knots_coefficients_degree, der=1))

    point_parameter2 = np.array(interpolate.splev(parameter2, knots_coefficients_degree))
    gradient_parameter2 = np.array(interpolate.splev(parameter2, knots_coefficients_degree, der=1))

    state, point_intersection = intersection2(point_parameter1, gradient_parameter1, point_parameter2, gradient_parameter2)

    if state == 0:
        # Ok
        return point_intersection, parameter2
    elif state == 1:
        # Linearly dependent. Return average instead.
        point_average = (point_parameter1 + point_parameter2) / 2
        return point_average, parameter2
    else:
        # state == 2. No valid intersection. Try again.
        return find_valid_intersection(parameter1, (parameter1 + parameter2) / 2, knots_coefficients_degree)



def intersection2(point1, gradient1, point2, gradient2):
    """"
    Finds the intersection between two lines

    This functions computes the intersection point given by the lines
    L1: x(t1) = gradient1[0] * t1 + point1[0],    L2: x(t2) = gradient2[0] * t2 + point2[0],
        y(t1) = gradient1[1] * t1 + point1[1],        y(t2) = gradient2[1] * t2 + point2[1]
    The intersection has to occur for positive t1 and negative t2 to ensure the direction of the curve is respected
    Arguments:
        point1      -- the x,y coordinates of point1
        gradient1   -- the gradient at point1
        point2      -- the x,y coordinates of point2
        gradient2   -- the gradient at point2
    """
    state = 0  # Status: 0=OK, 1=LINEARLY DEPENDENT, 2=NO VALID INTERSECTION

    A = np.column_stack((gradient1, -gradient2))
    b = np.array(point2-point1)

    try:
        t1, t2 = np.linalg.solve(A, b)
    except:
        # Linearly dependent
        state = 1
        return state, np.array([])

    point_intersection = gradient1 * t1 + point1

    if t1 > 0 > t2:
        # Found intersection
        return state, point_intersection
    else:
        # No valid intersection
        state = 2
        return state, np.array([])


def debug_geometry_data(geometrydata):
    """
    Plot to debug the geometry data

    Arguments:
        geometrydata
    """
    points_segment_list, boundary_conditions_list = split_on_boundary_condition_type(geometrydata)
    for points_segment, boundary_condition in zip(points_segment_list, boundary_conditions_list):
        boundary_condition_int = toint(boundary_condition)
        plt.plot(points_segment[:, 0], points_segment[:, 1], '.-', color="C{}".format(boundary_condition_int),
                 label=BOUNDARY_DICT[boundary_condition_int] + ": " + str(boundary_condition_int))
    plt.gca().axis('equal')
    plt.legend()
    plt.show()
    return plt


def get_refined_boundarypoints(geometrydata, polynomial_degree, smoothness, num_points_per_segment=500):
    """
    Function to obtain the splines of the boundary

    This function contains code from spline_and_intersection_points_segment which is not so nice. This should be refractored.


    Arguments:
        geometrydata        -- the geometrydata object
        points_segment      -- numpy array with points along segment
        degree_spline_fit   -- polynomial degree of spline fit. Bounded between 1 and 5.
        smoothness_spline_fit          -- smoothness_spline_fit of spline fit
    """
    points_segment_list, boundary_conditions_list = split_on_boundary_condition_type(geometrydata)

    points_segmentrefined_list = []
    for points_segment in points_segment_list:
        # If there are not enough equations for the number of unknowns, use a straight line
        if len(points_segment) <= polynomial_degree or polynomial_degree == 1:
            points_segmentrefined_list.append(points_segment)
        else:
            knots_coefficients_degree, parameter_original_points = interpolate.splprep(points_segment.T,
                                                                                       k=polynomial_degree,
                                                                                       s=smoothness)
            parameter = np.linspace(0, 1, num_points_per_segment)
            points_segment_spline = np.column_stack(interpolate.splev(parameter, knots_coefficients_degree))
            points_segmentrefined_list.append(points_segment_spline)

    return points_segmentrefined_list




