import cv2
import numpy as np

DEFAULT_COLOR = (0, 0, 255)
DEFAULT_THICKNESS = 1
DEFAULT_LENGTH = 2


def draw_cross(frame, point, color=DEFAULT_COLOR, thickness=DEFAULT_THICKNESS, length=DEFAULT_LENGTH):
    """
    Draw a cross on the frame at the given point. A correctly formatted point is subscriptable object
    who's first two elements are the x and y coordinates of the point, respectively. If incorrectly formatted,
    the point is converted to a flattened numpy array, and the first two elements are used as the coordinates.
    """
    # Convert the point to a usable format
    point = np.array(point).flatten()

    x, y, size = int(point[0]), int(point[1]), int(length / 2)
    cv2.line(frame, (x - size, y), (x + size, y), color, thickness)
    cv2.line(frame, (x, y - size), (x, y + size), color, thickness)


def draw_points(frame, points, color=DEFAULT_COLOR, thickness=DEFAULT_THICKNESS, length=DEFAULT_LENGTH):
    """
    Draw a cross on the frame at each point in the points list.
    """
    for point in points:
        draw_cross(frame, point, color=color, thickness=thickness, length=length)

def draw_line(frame, point1, point2, color=DEFAULT_COLOR, thickness=DEFAULT_THICKNESS, length=DEFAULT_LENGTH):
    """
    Draw a line on the frame joining the two points.
    Helper function for not forgetting to convert the coordinates to integers.
    """
    pt1 = (int(point1[0]), int(point1[1]))
    pt2 = (int(point2[0]), int(point2[1]))
    cv2.line(frame, pt1, pt2, color, thickness)

def draw_equation_line(frame, domain_range, linear_equation: callable, color=DEFAULT_COLOR, thickness=DEFAULT_THICKNESS):
    """
    Draw a line on the frame according to the given linear equation, and within the domain range of the equation.
    Returns the coordinates of the two points at the extremities of the line.

    @param frame: The image on which to draw the line
    @param domain_range: A tuple containing the domain of the function
    @param linear_equation: A callable that takes a number as input and returns a number as output
    """

    point1 = domain_range[0], linear_equation(domain_range[0])
    point2 = domain_range[1], linear_equation(domain_range[1])

    draw_line(frame, point1, point2, color, thickness)

    return point1, point2


def draw_joined_segments(frame, points, closed=True, color=DEFAULT_COLOR, width=2, thickness=DEFAULT_THICKNESS):
    """
    Draw a line joining each point in the points list to the next one in the list, with the last point joined to the first one by default.
    See draw_polygon() for non-intersecting lines.
    """

    for i in range(len(points) - 1):
        draw_line(frame, points[i], points[i + 1], color, thickness)

    if closed:
        draw_line(frame, points[0], points[-1], color, thickness)


def draw_polygon(frame, vertices, closed=True,  color=DEFAULT_COLOR, thickness=DEFAULT_THICKNESS):
    """
    Draw a polygon on the frame by joining the vertices in the points list such as the lines joining each vertex to the next one in the list do not intersect.
    """
    sorted_vertices = math_utils.get_non_intersecting_vertices(vertices)
    draw_joined_segments(frame, sorted_vertices, closed=closed, color=color, thickness=thickness)

def draw_convex_hull(frame, vertices, closed=True, color=DEFAULT_COLOR, thickness=DEFAULT_THICKNESS):
    """
    Draw a convex polygon on the frame by joining the vertices in the points list.
    """
    sorted_vertices = math_utils.get_convex_hull(vertices)
    draw_polygon(frame, sorted_vertices, closed=closed, color=color, thickness=thickness)

def draw_min_enclosing_rectangle(frame, vertices, color=DEFAULT_COLOR, thickness=DEFAULT_THICKNESS):
    """
    Draw a rectangle on the frame that encloses the given vertices.
    Returns the coordinates of the top left and bottom right vertices of the rectangle.
    """
    # Get the coordinates of the top left corner
    x_min = int(min(vertices, key=lambda x: x[0])[0])
    y_min = int(min(vertices, key=lambda x: x[1])[1])

    # Get the coordinates of the bottom right corner
    x_max = int(max(vertices, key=lambda x: x[0])[0])
    y_max = int(max(vertices, key=lambda x: x[1])[1])

    # Draw the rectangle
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, thickness)

    return (x_min, y_min), (x_max, y_max)
    
# ------------------------------
# EXAMPLE USAGE
# ------------------------------
if __name__ == '__main__':

    frame = np.zeros((500, 500, 3), dtype=np.uint8)

    vertices = []
    for i in range(100):
        x = np.random.randint(0, 500)
        y = np.random.randint(0, 500)

        vertices.append((x, y))

    draw_points(frame, vertices, color=(255,0,0), length=5)
    draw_convex_hull(frame, vertices, color=(0, 255, 0))
    #draw_polygon(frame, vertices, color=(0, 0, 255))

    while True:
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) == ord('q'):  
            break

    