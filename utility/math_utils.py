import numpy as np
import math

def calculate_slope(point1, point2):
    """
    Calculates the slope of line intersecting the two points.
    Returns math.inf if the line is vertical (division by zero)
    """

    if (point2[0] - point1[0]) == 0: return math.inf
    return (point2[1] - point1[1]) / (point2[0] - point1[0])



def calculate_intercept(slope, point):
    """
    Calculates the intercept of the line with the given slope and the given point
    """
    return point[1] - slope * point[0]


def get_linear_equation(point1, point2):
    """
    Returns a callable that represents the linear equation of the line intersecting the two points.
    """
    slope = calculate_slope(point1, point2)
    intercept = calculate_intercept(slope, point1)
    
    return lambda x: slope * x + intercept

def get_non_intersecting_vertices(vertices):
    # Sort the vertices by their x coordinate in ascending order
    sorted_vertices = sorted(vertices, key=lambda x: x[0])

    # Get the leftmost and rightmost points
    leftmost_vertex = sorted_vertices[0]
    rightmost_vertex = sorted_vertices[-1]

    # Get the equation of the line joining the leftmost and rightmost points
    equation = get_linear_equation(leftmost_vertex, rightmost_vertex)

    # Separate each point in two lists: one for the points above the line, and one for the points below the line
    points_above_line, points_below_line = [], []
    for vertex in sorted_vertices:
        points_above_line.append(vertex) if vertex[1] < equation(vertex[0]) else points_below_line.append(vertex)

    points_above_line.sort(key=lambda x: x[0])
    points_below_line.sort(key=lambda x: x[0], reverse=True)

    # Concatenate the two lists
    return points_above_line + points_below_line 

def get_convex_hull(vertices):
    """
    Compute the convex hull of the given vertices using the Graham Scan algorithm.
    """
    convex_hull = []
    sorted_vertices = sorted(vertices, key=lambda x: [x[0], x[1]])

    start = sorted_vertices.pop(0)
    convex_hull.append(start)

    sorted_vertices = sorted(sorted_vertices, key=lambda p: (calculate_slope(start, p), -p[1], p[0]))

    for vertex in sorted_vertices:
        convex_hull.append(vertex)


        while len(convex_hull) > 2 and get_cross_product(convex_hull[-3], convex_hull[-2], convex_hull[-1]) < 0:
            convex_hull.pop(-2)
 
    return convex_hull


def calculate_angle_degrees(point1, point2):
    """
    Returns angle (in degrees) of the vector intersecting the two points.
    """
    return np.arctan(calculate_slope(point1, point2)) * 180 / np.pi


def calculate_magnitude(point1, point2):
    """
    Calculated the magnitude of a vector intersecting the two points.
    """
    x1, y1 = point1
    x2, y2 = point2
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def get_cross_product(point1, point2, point3):
    """
    Returns the cross product of the two vectors intersecting the two points.
    """
    return ((point2[0] - point1[0]) * (point3[1] - point1[1])) - ((point2[1] - point1[1]) * (point3[0] - point1[0]))


def as_geometric_vector(initial_point, terminal_point):
    """
    A vector representation of a line that starts at the initial point and ends at the terminal point. 
    Returns a tuple containing the vector's magnitude and angle with respect to the horizontal axis.
    """
    return calculate_magnitude(initial_point, terminal_point), calculate_angle_degrees(initial_point, terminal_point)



def map_by_column(list, callables):
    """
    Maps the given list by the given callables. The callable of index n is applied to nth element of each column of the list.
    Example :
    list = [[1,2,3], [4,5,6], [7,8,9]]
    callables = [lambda x: x * 1, lambda x: x * 0, lambda x: x * 1]
    map_by_column(list, callables)
    >> [[1,0,3], [4,0,6], [7,0,9]]
    """
    flat_array = np.array(list).flatten()

    mapping_function = lambda i: callables[i % len(callables)](flat_array[i])

    flat_array = np.array([mapping_function(i) for i in range(len(flat_array))])

    return flat_array.reshape(-1, len(list[0]))