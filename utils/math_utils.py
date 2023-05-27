import numpy as np

def calculate_slope(point1, point2):
    """
    Calculates the slope of line intersecting the two points
    """
    return (point2[1] - point1[1]) / (point2[0] - point1[0])

def calculate_intercept(slope, point):
    """
    Calculates the intercept of the line with the given slope and the given point
    """
    return point[1] - slope * point[0]

def calculate_linear_equation(point1, point2):
    slope = calculate_slope(point1, point2)
    intercept = calculate_intercept(slope, point1)
    
    return lambda x: slope * x + intercept

