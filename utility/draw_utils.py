import cv2
import numpy as np

from utility import geometry_utils as geometry

DEFAULT_COLOR = (0, 0, 255)
DEFAULT_THICKNESS = 1
DEFAULT_LENGTH = 2

def draw_cross(frame, point, color=DEFAULT_COLOR, thickness=DEFAULT_THICKNESS, length=DEFAULT_LENGTH):
    """
    Draw a cross on the frame at the given point.

    Parameters
    ----------
    frame : ndarray
        The image on which to draw the cross.
    
    point : array_like
        The (x, y) coordinates at which to draw the cross. 
        If not a subscriptable object, an attempt is made to convert it to a flat numpy array.
        The first two elements of the array are used as the coordinates.
    
    color : tuple, optional, default = (0, 0, 255)
        The color of the cross. The default color is (0, 0, 255).
    
    thickness : int, optional, default = 1
        The thickness of the lines that make up the cross. The default thickness is 1 px.
    
    length : int, optional, default = 2
        The length of the lines that make up the cross. The default length is 2 px.
    """
    # Convert the point to a usable format
    point = np.array(point).flatten()

    x, y, size = int(point[0]), int(point[1]), int(length / 2)
    cv2.line(frame, (x - size, y), (x + size, y), color, thickness)
    cv2.line(frame, (x, y - size), (x, y + size), color, thickness)


def draw_all_crosses(frame, points, color=DEFAULT_COLOR, thickness=DEFAULT_THICKNESS, length=DEFAULT_LENGTH):
    """
    Draw a cross on the frame at each point in the points list.

    Parameters
    ----------
    frame : ndarray
        The image on which to draw the cross.

    points : array_like
        The list of (x, y) coordinates at which to draw the crosses. If only one point is given, 
        the draw_cross() function is called automatically. 

    color : tuple, optional, default = (0, 0, 255)
        The color of the cross. The default color is (0, 0, 255).
    
    thickness : int, optional, default = 1
        The thickness of the lines that make up the cross. The default thickness is 1 px.

    length : int, optional, default = 2
        The length of the lines that make up the cross. The default length is 2 px.

    See Also
    ----------
    draw_cross() : Draw a cross at a single point.  
    """

    # Check if the points list is correctly formatted
    if np.ndim(points) == 1:
        draw_cross(frame, points, color=color, thickness=thickness, length=length)
        return
    
    for point in points:
        draw_cross(frame, point, color=color, thickness=thickness, length=length)


def draw_line(frame, point1, point2, color=DEFAULT_COLOR, thickness=DEFAULT_THICKNESS):
    """
    Draw a line on the frame joining the two points.
    Helper function for not forgetting to convert the coordinates to integers.

    Parameters
    ----------
    frame : ndarray
        The image on which to draw the line.
    
    point1 : Any
        First point of the line segment

    point2 : Any
        Second point of the line segment
        
    color : tuple, optional, default = (0, 0, 255)
        The color of the line. The default color is (0, 0, 255).
    
    thickness : int, optional, default = 1
        The thickness of the line. The default thickness is 1 px.
    
    length : int, optional, default = 2
        The length of the line. The default length is 2 px.
    """
    pt1 = (int(point1[0]), int(point1[1]))
    pt2 = (int(point2[0]), int(point2[1]))
    cv2.line(frame, pt1, pt2, color, thickness)


def draw_equation_line(frame, domain_range, linear_equation: callable, color=DEFAULT_COLOR, thickness=DEFAULT_THICKNESS):
    """
    Draw a line on the frame according to the given linear equation, and within the domain range of the equation.
    Returns the coordinates of the two points at the extremities of the line.

    Parameters
    ----------

    frame : ndarray
        The image on which to draw the line.

    domain_range : Any
        A set of two numbers representing the domain of the function described by the linear equation.

    linear_equation : callable
        A function that takes a number as input and returns a number as output.
    
    color : tuple, optional, default = (0, 0, 255)
        The color of the line. The default color is (0, 0, 255).
    
    thickness : int, optional, default = 1
        The thickness of the line. The default thickness is 1 px.
    """

    point1 = domain_range[0], linear_equation(domain_range[0])
    point2 = domain_range[1], linear_equation(domain_range[1])

    draw_line(frame, point1, point2, color, thickness)

    return point1, point2


def draw_joined_segments(frame, vertices, closed=True, color=DEFAULT_COLOR, thickness=DEFAULT_THICKNESS):
    """
    Draw a line joining each vertex in the points list to the next one in the list, with the last point joined to the first one by default.
    
    Parameters
    ----------
    frame : ndarray
        The image on which to draw the line.
    
    vertices : array_like
        The list of (x, y) coordinates connecting each segment.
    
    closed : bool, optional, default = True
        If True, the last point is joined to the first one. Set to True by default.
    
    color : tuple, optional, default = (0, 0, 255)
        The color of the line. The default color is (0, 0, 255).
    
    thickness : int, optional, default = 1
        The thickness of the line. The default thickness is 1 px.
    
    See Also
    ----------
    draw_line : Draw a line between two points.
    draw_polygon : Draw non-intersecting segments between the points in the list.
    draw_convex_hull : Draw non-intersecting segments between the points in the list, such as the segments form a convex polygon.
    """

    for i in range(len(vertices) - 1):
        draw_line(frame, vertices[i], vertices[i + 1], color, thickness)

    if closed:
        draw_line(frame, vertices[0], vertices[-1], color, thickness)


def draw_polygon(frame, vertices, closed=True,  color=DEFAULT_COLOR, thickness=DEFAULT_THICKNESS):
    """
    Draw a polygon on the frame by joining the vertices in the points list such as the lines joining each vertex to the next one in the list do not intersect.

    Parameters
    ----------
    frame : ndarray
        The image on which to draw the line.
    
    vertices : array_like
        The list of (x, y) coordinates connecting each segment.
    
    closed : bool, optional, default = True
        If True, the last point is joined to the first one. Set to True by default.
    
    color : tuple, optional, default = (0, 0, 255)
        The color of the line. The default color is (0, 0, 255).
    
    thickness : int, optional, default = 1
        The thickness of the line. The default thickness is 1 px.

    See Also
    ----------
    draw_line : Draw a line between two points.
    draw_joined_segments : Draw non-intersecting segments between the points in the list.
    draw_convex_hull : Draw non-intersecting segments between the points in the list, such as the segments form a convex polygon.
    """
    sorted_vertices = geometry.get_non_intersecting_vertices(vertices)
    draw_joined_segments(frame, sorted_vertices, closed=closed, color=color, thickness=thickness)


def draw_convex_hull(frame, vertices, closed=True, color=DEFAULT_COLOR, thickness=DEFAULT_THICKNESS):
    """
    Draw a convex polygon on the frame by joining the vertices in the points list.

    Parameters
    ----------
    frame : ndarray
        The image on which to draw the line.
    
    vertices : array_like
        The list of (x, y) coordinates connecting each segment.
    
    closed : bool, optional, default = True
        If True, the last point is joined to the first one. Set to True by default.
    
    color : tuple, optional, default = (0, 0, 255)
        The color of the line. The default color is (0, 0, 255).
    
    thickness : int, optional, default = 1
        The thickness of the line. The default thickness is 1 px.
    
    See Also
    ----------
    draw_line : Draw a line between two points.
    draw_joined_segments : Draw non-intersecting segments between the points in the list.
    draw_polygon : Draw non-intersecting segments between the points in the list.
    """
    sorted_vertices = geometry.get_convex_hull(vertices)
    draw_polygon(frame, sorted_vertices, closed=closed, color=color, thickness=thickness)


def draw_min_enclosing_rectangle(frame, vertices, color=DEFAULT_COLOR, thickness=DEFAULT_THICKNESS):
    """
    Draw a rectangle on the frame that encloses the given vertices.
    
    Parameters
    ----------
    frame : ndarray
        The image on which to draw the rectangle.
    
    vertices : array_like
        The list of (x, y) coordinates connecting each segment.
    
    color : tuple, optional, default = (0, 0, 255)
        The color of the rectangle. The default color is (0, 0, 255).
    
    thickness : int, optional, default = 1
        The thickness of the rectangle. The default thickness is 1 px.
    
    Returns
    ----------
    The coordinates of the top left and bottom right corners of the rectangle, in the form (x_min, y_min), (x_max, y_max).
    """
    rectangle = geometry.get_min_enclosing_rectangle(vertices)
    cv2.rectangle(frame, rectangle[0], rectangle[1], color, thickness)
    return rectangle

    