import cv2
import numpy as np

import utils.math_utils


def draw_cross(frame, point, color=(0, 0, 255), width=2, thickness=1):
    x, y, size = int(point[0]), int(point[1]), int(width / 2)
    cv2.line(frame, (x - size, y), (x + size, y), color, thickness)
    cv2.line(frame, (x, y - size), (x, y + size), color, thickness)


def draw_points(frame, points, color=(0, 0, 255), width=2):
    for point in points:
        draw_cross(frame, point, color=color, width=width)


def draw_line(frame, domain_range, linear_equation: callable, color=(0, 0, 255), thickness=2):
    """
    Draw a line on the frame according to the given linear equation, and within the domain range of the equation.

    @param frame: The image on which to draw the line
    @param domain_range: A tuple containing the domain of the function
    @param linear_equation: A callable that takes a number as input and returns a number as output
    """

    point1 = (int(domain_range[0]), int(linear_equation(domain_range[0])))
    point2 = (int(domain_range[1]), int(linear_equation(domain_range[1])))

    cv2.line(frame, point1, point2, color, thickness)


def draw_joined_points(frame, points, closed=True, color=(0, 0, 255), width=2, thickness=1):
    """
    Draw a line joining each point in the points list to the next one in the list, with the last point joined to the first one by default.
    See draw_polygon() for non-intersecting lines.
    """
    for i in range(len(points) - 1):
        cv2.line(frame, points[i], points[i + 1], color, thickness)

    if closed:
        cv2.line(frame, points[-1], points[0], color, thickness)


def draw_polygon(frame, vertices, closed=True,  color=(0, 0, 255), thickness=1):
    """
    Draw a polygon on the frame by joining the vertices in the points list such as the lines joining each vertex to the next one in the list do not intersect.
    """
        # Sort the vertices by their x coordinate in ascending order
    sorted_vertices = sorted(vertices, key=lambda x: x[0])

    # Get the leftmost and rightmost points
    leftmost_vertex = sorted_vertices[0]
    rightmost_vertex = sorted_vertices[-1]

    # Get the equation of the line joining the leftmost and rightmost points
    equation = utils.math_utils.calculate_linear_equation(leftmost_vertex, rightmost_vertex)

    # Separate each point in two lists: one for the points above the line, and one for the points below the line
    points_above_line, points_below_line = [], []
    for vertex in sorted_vertices:
        points_above_line.append(vertex) if vertex[1] < equation(vertex[0]) else points_below_line.append(vertex)

    # Sort the points in each list by their x coordinate in ascending order
    points_above_line.sort(key=lambda x: x[0])
    points_below_line.sort(key=lambda x: x[0])

    sorted_vertices = points_above_line + points_below_line[::-1]

    draw_joined_points(frame, sorted_vertices, closed=closed, color=color, thickness=thickness)



# ------------------------------
# EXAMPLE 
# ------------------------------
if __name__ == '__main__':

    
    frame = np.zeros((500, 500, 3), dtype=np.uint8)

    # random points
    vertices = []

    for i in range(100):
        x = np.random.randint(0, 500)
        y = np.random.randint(0, 500)

        vertices.append((x, y))



    draw_points(frame, vertices, width=5, color=(0, 255, 0))
    draw_polygon(frame, vertices, color=(0, 255, 0), thickness=2)


    while True:
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) == ord('q'):  
            break

    