""""
This file contains some predefined geometries for ease of use
"""

import numpy as np
from NiFlow.geometry.create_geometry import WALLDOWN, RIVER, WALLUP, SEA
import ngsolve


############################### Geometry data ###########################
# Geometries defined in terms of points. Useful for geometries with straight sides.


def rectangle(B, L, Lextension=0):
    """
    Rectangular domain.

    Rectangular domain of length L+Lextension and width B

        y=B/2 |---------------------------|-----|
              |                           |     |
    (Sea) y=0 |                           |     | (River)
              |                           |     |
       y=-B/2 |---------------------------|-----|
             x=0                         x=L   x=L+Lextension
    """
    if Lextension==0:
        geometrydata = np.array(
            [[0, -B / 2, WALLDOWN], [L, -B / 2, RIVER], [L, B / 2, WALLUP],
             [0, B / 2, SEA]])
    else:
        geometrydata = np.array(
            [[-Lextension, -B / 2, WALLDOWN], [L, -B / 2,RIVER], [L, B / 2, WALLUP], [-Lextension, B / 2, SEA]])
    return geometrydata


def trapezoid(B, L1, L2):
    """
    Trapezoid domain.

    A trapezoidal domain with nonparallel sides at the Sea and River boundaries.
    The lower parallel side has length L1, the upper parallel L2 and the width is B.
    The sloped sides have the same slope.

                       <-------L2------>
                     x=(L1-L2)/2        x=(L1+L2)/2
                  y=B |------------------|
                    |                     |
    (Sea)         |                        |    (River)
                |                           |
          y=0 |------------------------------|
             x=0                            x=L1
    """

    geometrydata = np.array(
            [[0, 0, WALLDOWN], [L1, 0, RIVER], [(L1+L2)/2, B, WALLUP],
             [(L1-L2)/2, B, SEA]])
    return geometrydata




###################################### geometrycurves #########################################
# Geometries defined in terms of parametric curves. Useful for geometries with curved sides.

def parametric_rectangle(L, B, L_BL_sea=0., L_R_sea=0., L_RA_sea=0., L_BL_river=0., L_R_river=0., L_RA_river=0.):

    """
    
    Parametric version of rectangular domain with width B0 and length L.
    
            y=B0/2  |--------------------------------------------------|
                    |                                                  |
    (Sea)   y=0     |                                                  | (River)
                    |                                                  |
            y=-B0/2 |--------------------------------------------------|
                    x=0                                                x=L
    
                    

    """

    def linearly_connect(t, p1, p2):
        return p1 + (p2 - p1) * t
    

    def top(t, L=L, B=B): # added L and B as arguments for this function to make sure that these functions do not rely on external context; that might become an issue in pickling/unpickling from remote computers
        return linearly_connect(t, np.array([L+L_RA_river+L_R_river+L_BL_river, B/2]), np.array([-L_BL_sea-L_R_sea-L_RA_sea, B/2]))
    
    def bottom(t, L=L, B=B):
        return linearly_connect(t, np.array([-L_BL_sea-L_R_sea-L_RA_sea, -B/2]), np.array([L+L_RA_river+L_R_river+L_BL_river,-B/2]))
    
    def leftside(t, L=L, B=B):
        return linearly_connect(t, np.array([-L_BL_sea-L_R_sea-L_RA_sea, B/2]), np.array([-L_BL_sea-L_R_sea-L_RA_sea, -B/2]))
    
    def rightside(t, L=L, B=B):
        return linearly_connect(t, np.array([L+L_RA_river+L_R_river+L_BL_river, -B/2]), np.array([L+L_RA_river+L_R_river+L_BL_river, B/2]))
    
    geometrycurves = [[bottom, WALLDOWN], [top, WALLUP], [leftside, SEA], [rightside, RIVER]]
    
    return geometrycurves


def exponential(B0, L, Lc):
    """
    Exponential domain.

    Exponential convergent domain with initial width B0, width convergence length Lc and length L.
    The spatially varying width is given by B(x) = B0 * exp(-x/Lc) for 0<x<L

        y=B0/2 |----\
               |     \-------\
               |              \------------|
    (Sea)  y=0 |                           | (River)
               |              /------------|
               |     /-------/
       y=-B0/2 |----/
             x=0                         x=L
    """

    def expside(t, B0, Lc, L1, L2, isUpper):
        """ Parametrised exponential side with initial width B0 and convergence length Lc. """
        xtilde = L1 + (L2 - L1) * t
        if isUpper:
            factor = 1
        else:
            factor = -1
        return np.array([xtilde, factor * B0 / 2 * np.exp(-xtilde / Lc)])

    def bottomexp(t):
        return expside(t, B0, Lc, 0, L, False)

    def topexp(t):
        return expside(t, B0, Lc, L, 0, True)


    def linearly_connect(t, p1, p2):
        return p1 + (p2 - p1) * t

    def rightside(t):
        return linearly_connect(t, bottomexp(1), topexp(0))

    def leftside(t):
        return linearly_connect(t, topexp(1), bottomexp(0))

    geometrycurves = [[bottomexp, WALLDOWN], [rightside, RIVER], [topexp, WALLUP], [leftside, SEA]]

    # The inverse of the parametric boundaries so that boundary conditions on these boundaries may be prescribed more easily

    t_river = (ngsolve.x - rightside(0)[0]) / (rightside(1)[0] - rightside(0)[1]) + \
             (ngsolve.y - rightside(0)[1]) / (rightside(1)[1] - rightside(0)[1])
    
    t_sea = (ngsolve.x - leftside(0)[0]) / (leftside(1)[0] - leftside(0)[1]) + \
             (ngsolve.y - leftside(0)[1]) / (leftside(1)[1] - leftside(0)[1])

    return geometrycurves, t_river, t_sea


def exponential_rational(C1, C2, B0, L):
    """
    Exponential rational domain.

    Exponential rational convergent domain where the width is dependent on the composition between a rational and exponential function.

    The polynomials are constructed as in numpy.polyval, e.g.,
    The numerator of the rational function is constructed for N1 = len(C1) as:
        n(x) =  C1[0]*x^(N1-1) + C1[1]*x^(N1-2) ... + C1[N1-1]
    The denominator of the rational function is constructed for N2 = len(C2) as:
        d(x) =  C2[0]*x^(N2-1) + C2[1]*x^(N2-2) ... + C2[N2-1]
    The rational function is then given by
        r(x) = n(x)/d(x)

    The spatially varying exponential rational width reads
        B(x) = B0 * exp(-r(0)) * exp(r(x)),      for 0<x<L,
    with initial width B0.

        y=B0/2 |--\
               |    \---\
               |         \----\
               |               \--------------|
    (Sea)  y=0 |                              | (River)
               |                /-------------|
               |          /----/
               |    /---/
       y=-B0/2 |--/
              x=0                            x=L
    """
    def exp_rat_side(t, x1, x2, isUpper):
        """ Parameterised side of the rational exponential domain. The side represents the half width."""
        xtilde = x1 + (x2 - x1) * t
        if isUpper:
            factor = 1
        else:
            factor = -1

        def side(xtilde):
            def rational(xtilde):
                return np.polyval(C1, xtilde)/np.polyval(C2, xtilde)

            return factor * B0 * np.exp(-rational(0)) / 2 * np.exp(rational(xtilde))

        return np.array([xtilde, side(xtilde)])


    def bottomside(t):
        return exp_rat_side(t, 0, L, False)

    def topside(t):
        return exp_rat_side(t, L, 0, True)

    def linearly_connect(t, p1, p2):
        return p1 + (p2 - p1) * t

    def rightside(t):
        return linearly_connect(t, bottomside(1), topside(0))

    def leftside(t):
        return linearly_connect(t, topside(1), bottomside(0))

    geometrycurves = [[bottomside, WALLDOWN], [rightside, RIVER], [topside, WALLUP], [leftside, SEA]]

    t_river = (ngsolve.x - rightside(0)[0]) / (rightside(1)[0] - rightside(0)[1]) + \
             (ngsolve.y - rightside(0)[1]) / (rightside(1)[1] - rightside(0)[1])
    
    t_sea = (ngsolve.x - leftside(0)[0]) / (leftside(1)[0] - leftside(0)[1]) + \
             (ngsolve.y - leftside(0)[1]) / (leftside(1)[1] - leftside(0)[1])

    return geometrycurves, t_river, t_sea


def annulus(r1, r2, theta1, theta2, Lextension=0):
    """
    Extended annular domain.

    Annular domain from theta1 to theta2 with inner radius r1 and outer radius r2.
    If Lextension==0, then returns normal annulus else an extended annulus is returned.

             _theta=theta2
             _____________
    (River)  |   |        \
             |___|___       \
     Lextension^     \       |
                      |_____ | _theta=theta1
                     r=r1    r=r2
                        (Sea)
    """

    def arc(t, r, theta1, theta2):
        """"Arc from theta1 to theta2 with radius r"""
        return np.array([r * np.cos((theta2 - theta1) * t + theta1), r * np.sin((theta2 - theta1) * t + theta1)])

    def arc1(t):
        return arc(t, r1, theta2, theta1)

    def arc2(t):
        return arc(t, r2, theta1, theta2)

    p2extension = np.array([-Lextension, arc2(1)[1]])
    p1extension = np.array([-Lextension, arc1(0)[1]])

    def linearly_connect(t, p1, p2):
        return p1 + (p2 - p1) * t

    def bottom(t):
        return linearly_connect(t, arc1(1), arc2(0))

    def arc2_extended(t):
        return linearly_connect(t, arc2(1), p2extension)

    def side(t):
        return linearly_connect(t, p2extension, p1extension)

    def arc1_extended(t):
        return linearly_connect(t, p1extension, arc1(0))

    def side_arcs(t):
        return linearly_connect(t, arc2(1), arc1(0))


    if Lextension == 0:
        geometrycurves = [[arc1, WALLDOWN], [bottom, RIVER], [arc2, WALLUP], [side_arcs, SEA]]

        t_sea = (ngsolve.x - side_arcs(0)[0]) / (side_arcs(1)[0] - side_arcs(0)[1]) + \
             (ngsolve.y - side_arcs(0)[1]) / (side_arcs(1)[1] - side_arcs(0)[1])
    else:
        geometrycurves = [[arc1, WALLDOWN], [bottom, RIVER], [arc2, WALLUP], [arc2_extended, WALLUP], [side, SEA],
                        [arc1_extended, WALLDOWN]]
        
        t_sea = (ngsolve.x - side(0)[0]) / (side(1)[0] - side(0)[1]) + \
             (ngsolve.y - side(0)[1]) / (side(1)[1] - side(0)[1])
        
    t_river = (ngsolve.x - bottom(0)[0]) / (bottom(1)[0] - bottom(0)[1]) + \
             (ngsolve.y - bottom(0)[1]) / (bottom(1)[1] - bottom(0)[1])
    
    return geometrycurves, t_river, t_sea


def linearly_converging(r1, r2, theta, isSeaBoundaryCurved=True, isRiverBoundaryCurved=True):
    """
    linearly converging channel with possibly curved sides


    Linearly converging channel with angle _theta from r1 to r2

    _theta=_theta/2    /----\
                    /       \----\
                   |              \----|
            (Sea) |                   |   o  (River)
                   \              /----|
                    \      /----/
    _theta=-_theta/2   \----/
                    r=r1              r=r2

    """

    def arc(t, r, theta1, theta2):
        """"Arc from theta1 to theta2 with radius r"""
        return np.array([r * np.cos((theta2 - theta1) * t + theta1), r * np.sin((theta2 - theta1) * t + theta1)])

    def arc_sea(t):
        return arc(t, r1, np.pi - theta/2, np.pi + theta/2)

    def arc_river(t):
        return arc(t, r2, np.pi + theta / 2, np.pi - theta / 2)

    def linearly_connect(t, p1, p2):
        return p1 + (p2 - p1) * t

    def bottom(t):
        return linearly_connect(t, arc_sea(1), arc_river(0))

    def top(t):
        return linearly_connect(t, arc_river(1), arc_sea(0))

    # Straight sections
    def straight_sea(t):
        return linearly_connect(t, top(1), bottom(0))

    def straight_river(t):
        return linearly_connect(t, bottom(1), top(0))


    # Logic set sea and river boundaries
    if isSeaBoundaryCurved:
        # Curved sea boundary
        side_sea = arc_sea
    else:
        # Straight sea boundary
        side_sea = straight_sea

    if isRiverBoundaryCurved:
        #Curved river boundary
        side_river = arc_river
    else:
        # Straight river boundary
        side_river = straight_river


    geometrycurves = [[bottom, WALLDOWN], [side_river, RIVER], [top, WALLUP], [side_sea, SEA]]

    return geometrycurves



def side_channel_rectangle(L, B, Lchannel, Bchannel, rconnect=None, circular_channel_end=False):
    """                     
                              Bchannel
                           <------------>
                           
                                          y = B/2 + Lchannel
           
                           ______________                 
                           |            |                 
                           |            |                 
                           |            |                 
     y=B/2    -------------/            \-------------    
              |                                      |
              |                                      |
              |                                      |
     y=-B/2   ----------------------------------------

             x=0                                     x=L
    
             
    Rectangular domain with side channel in the middle of the long side of the rectangle. Side channel is connected to main estuary
    via quarter circles of radius rconnect and the top of the side channel is a semicircle of radius Bchannel/2. It follows that 
    2*rconnect+Bchannel < Lchannel. The topside is parametrised in such a way that each segment gets 1/7th of the parametrisation length. The 
    segments are: left rectangular side, left connection quarter circle, left side channel side, top semicircle, right side channel side, 
    right connection quarter circle, right rectangular side.

    """

    if rconnect is None:
        rconnect = Bchannel

    if 2 * rconnect + Bchannel >= L:
        print('Proposed side channel too wide')
        return

    if (Lchannel <= rconnect + Bchannel and circular_channel_end) or (Lchannel <= rconnect and not circular_channel_end):
        print('Proposed side channel too short: cannot fit straight segment')
        return
    
    def linearly_connect(t, p1, p2):
        return p1 + (p2 - p1) * t
    
    def bottom(t):
        return linearly_connect(t, np.array([0, -B/2]), np.array([L,-B/2]))
    
    def leftside(t):
        return linearly_connect(t, np.array([0, B/2]), np.array([0, -B/2]))
    
    def rightside(t):
        return linearly_connect(t, np.array([L, -B/2]), np.array([L, B/2]))
    
    def right_rect_side(t):
        return linearly_connect(t, np.array([L, B/2]), np.array([L/2+Bchannel/2+rconnect, B/2]))
    
    def right_connect_circle(t):
        return right_rect_side(1) + np.array([0, rconnect]) + np.array([rconnect * np.cos(-np.pi/2 * (1+t)),
                                                                          rconnect * np.sin(-np.pi/2 * (1+t))])
    
    def right_channel_side(t):
        if circular_channel_end:
            return linearly_connect(t, right_connect_circle(1), right_connect_circle(1) + np.array([0, Lchannel-rconnect-Bchannel]))
        else:
            return linearly_connect(t, right_connect_circle(1), right_connect_circle(1) + np.array([0, Lchannel - rconnect]))
    
    if circular_channel_end:
        def top_semicircle(t):
            return right_channel_side(1) + np.array([-Bchannel/2, 0]) + np.array([Bchannel/2 * np.cos(np.pi*t), Bchannel/2 * np.sin(np.pi*t)])
    else:
        def channelend(t):
            return linearly_connect(t, right_channel_side(1), right_channel_side(1) + np.array([-Bchannel, 0]))
        
    def left_channel_side(t):
        if circular_channel_end:
            return linearly_connect(t, top_semicircle(1), np.array([L/2-Bchannel/2, B/2+rconnect]))
        else:
            return linearly_connect(t, channelend(1), channelend(1) - np.array([0, Lchannel-rconnect]))
    
    def left_connect_circle(t):
        return left_channel_side(1) + np.array([-rconnect, 0]) + np.array([rconnect * np.cos(-np.pi/2 * t), rconnect * np.sin(-np.pi/2 * t)])
    
    def left_rect_side(t):
        return linearly_connect(t, left_connect_circle(1), np.array([0, B/2]))
    
    def top(t):
        if 0 <= t < 1/7:
            return right_rect_side(7*t)
        elif 1/7 <= t < 2/7:
            return right_connect_circle(7*(t-1/7))
        elif 2/7 <= t < 3/7:
            return right_channel_side(7*(t-2/7))
        elif 3/7 <= t < 4/7:
            if circular_channel_end:
                return top_semicircle(7*(t-3/7))
            else:
                return channelend(7*(t-3/7))
        elif 4/7 <= t < 5/7:
            return left_channel_side(7*(t-4/7))
        elif 5/7 <= t < 6/7:
            return left_connect_circle(7*(t-5/7))
        elif 6/7 <= t <= 1:
            return left_rect_side(7*(t-6/7))

    
    geometrycurves = [[bottom, WALLDOWN], [top, WALLUP], [leftside, SEA], [rightside, RIVER]]
    
    return geometrycurves


def downward_side_channel_rectangle(L, B, Lchannel, Bchannel, rconnect=None, circular_channel_end=False):
    """                     
                    
     y=B/2    ----------------------------------------    
              |                                      |
              |                                      |
              |                                      |
     y=-B/2   -------------\            /-------------
                           |            |
                           |            |
                           |            |
                           ______________           y = -B/2-Lchannel
                                          
             x=0                                     x=L

                           <------------>
                              Bchannel
    
             
    Rectangular domain with side channel in the middle of the long side of the rectangle. Side channel is connected to main estuary
    via quarter circles of radius rconnect and the top of the side channel is a semicircle of radius Bchannel/2. It follows that 
    2*rconnect+Bchannel < Lchannel. The topside is parametrised in such a way that each segment gets 1/7th of the parametrisation length. The 
    segments are: left rectangular side, left connection quarter circle, left side channel side, top semicircle, right side channel side, 
    right connection quarter circle, right rectangular side.

    """

    if rconnect is None:
        rconnect = Bchannel

    if 2 * rconnect + Bchannel >= L:
        print('Proposed side channel too wide')
        return

    if (Lchannel <= rconnect + Bchannel and circular_channel_end) or (Lchannel <= rconnect and not circular_channel_end):
        print('Proposed side channel too short: cannot fit straight segment')
        return
    
    def linearly_connect(t, p1, p2):
        return p1 + (p2 - p1) * t
    
    def top(t):
        return linearly_connect(t, np.array([L, B/2]), np.array([0,B/2]))
    
    def leftside(t):
        return linearly_connect(t, np.array([0, B/2]), np.array([0, -B/2]))
    
    def rightside(t):
        return linearly_connect(t, np.array([L, -B/2]), np.array([L, B/2]))
    
    def left_rect_side(t):
        return linearly_connect(t, np.array([0, -B/2]), np.array([L/2-Bchannel/2-rconnect, -B/2]))
    
    def left_connect_circle(t):
        return left_rect_side(1) + np.array([0, -rconnect]) + np.array([rconnect * np.cos(np.pi/2 * (1-t)),
                                                                        rconnect * np.sin(np.pi/2 * (1-t))])
    
    def left_channel_side(t):
        if circular_channel_end:
            return linearly_connect(t, left_connect_circle(1), left_connect_circle(1) - np.array([0, Lchannel-rconnect-Bchannel]))
        else:
            return linearly_connect(t, left_connect_circle(1), left_connect_circle(1) - np.array([0, Lchannel-rconnect]))
        
    if circular_channel_end:
        def bottom_semicircle(t):
            return left_channel_side(1) + np.array([Bchannel/2, 0]) + np.array([Bchannel/2 * np.cos(np.pi*(1+t)), Bchannel/2 * np.sin(np.pi(1+t))])
    else:
        def channelend(t):
            return linearly_connect(t, left_channel_side(1), left_channel_side(1) + np.array([Bchannel, 0]))
        
    def right_channel_side(t):
        if circular_channel_end:
            return linearly_connect(t, bottom_semicircle(1), bottom_semicircle(1) + np.array([0, Lchannel-rconnect]))
        else:
            return linearly_connect(t, channelend(1), channelend(1) + np.array([0, Lchannel-rconnect]))
    
    def right_connect_circle(t):
        return left_channel_side(1) + np.array([rconnect, 0]) + np.array([rconnect * np.cos(np.pi/2 * (2-t)), rconnect * np.sin(np.pi/2 * (2-t))])

    def right_rect_side(t):
        return linearly_connect(t, right_connect_circle(1), np.array([L, -B/2]))
    
    def bottom(t):
        if 0 <= t < 1/7:
            return left_rect_side(7*t)
        elif 1/7 <= t < 2/7:
            return left_connect_circle(7*(t-1/7))
        elif 2/7 <= t < 3/7:
            return left_channel_side(7*(t-2/7))
        elif 3/7 <= t < 4/7:
            if circular_channel_end:
                return bottom_semicircle(7*(t-3/7))
            else:
                return channelend(7*(t-3/7))
        elif 4/7 <= t < 5/7:
            return right_channel_side(7*(t-4/7))
        elif 5/7 <= t < 6/7:
            return right_connect_circle(7*(t-5/7))
        elif 6/7 <= t <= 1:
            return right_rect_side(7*(t-6/7))

    
    geometrycurves = [[bottom, WALLDOWN], [top, WALLUP], [leftside, SEA], [rightside, RIVER]]
    
    return geometrycurves



