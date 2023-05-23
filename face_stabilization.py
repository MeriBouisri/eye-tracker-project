import cv2
import mediapipe as mp
import numpy as np
import pyautogui

# ========== LANDMARK INDEX CONSTANTS ==========

# The following landmarks will be used as guideline vertices

LOWER_CENTER_NOSE_RIDGE_LANDMARK = 197
MIDDLE_CENTER_NOSE_RIDGE_LANDMARK = 168
UPPER_CENTER_NOSE_RIDGE_LANDMARK = 8

OUTER_LEFT_CORNER_LANDMARK = 33
OUTER_RIGHT_CORNER_LANDMARK = 263

LEFT_IRIS_LANDMARK = [474, 475, 476, 477]
RIGHT_IRIS_LANDMARK = [468, 469, 470, 471]

face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

def main():
    cam = cv2.VideoCapture(0)
    screen_w, screen_h = pyautogui.size()

    # initialize bounding box shapes so that we can maintain shape later
    valid_eye_box_boundaries = False
    valid_face_box_boundaries = False

    while True:
        _, frame = cam.read()
        frame = cv2.flip(frame, 1)

        # ========== INITIAL FACE MESH BEFORE IMAGE STABILIZATION ==========

        landmark_points = apply_face_mesh(frame)

        if landmark_points:
            global rotation_vector
            landmarks = landmark_points[0].landmark

            # We will use the line that crosses the outer corners of the eyes as a reference to the face rotation
            # The angle of rotation is the angle between that line and the horizontal axis
            # So, the face is pointing forward along the horizontal axis when the slope of that line is null
            rotation_vector = vector_of(frame, landmarks, OUTER_LEFT_CORNER_LANDMARK, OUTER_RIGHT_CORNER_LANDMARK)

        # Rotate frame to stabilize it along with face rotation
        stable_frame = rotate_frame(frame, rotation_vector[1])
        stable_frame_h, stable_frame_w = stable_frame.shape[:2]

        # Then get landmarks again to get the updated coordinates without applying a rotation matrix on those coordinates
        stable_landmark_points = apply_face_mesh(stable_frame)

        if stable_landmark_points:

            stable_landmarks = stable_landmark_points[0].landmark

            center_right = average_landmark_coordinates(stable_frame, stable_landmarks, RIGHT_IRIS_LANDMARK)
            cv2.circle(stable_frame, center_right, 2, (0, 0, 255), -1)
        
            center_left = average_landmark_coordinates(stable_frame, stable_landmarks, LEFT_IRIS_LANDMARK)
            cv2.circle(stable_frame, center_left, 2, (0, 0, 255), -1)

            # Vertices that will be used to make rotation and translation calculations
            eye_area_vertices = calculate_face_vertices(stable_frame, stable_landmarks, True)

            # We will use bounding boxes to crop the eyes and the face
            # The eye area is the most important. Face area is just for vizualisation purposes

            # ===== EYE AREA BOUNDING BOX =====

            eye_bounding_box = minimal_bounding_box_from_coordinates(stable_frame, eye_area_vertices)
    
            cv2.rectangle(stable_frame, eye_bounding_box[0], eye_bounding_box[1], (255,0, 0), 1)
            
            if not valid_eye_box_boundaries:
                eye_bounding_box_shape = rectangle_dimensions(eye_bounding_box)
                valid_eye_box_boundaries = True
        
            eye_box = resize_dimensions(eye_bounding_box_shape, 5)

            eye_frame = crop_frame(stable_frame, eye_bounding_box)
            eye_frame = cv2.resize(eye_frame, eye_box, 3)

            # put frame at top left of stable frame for vizualisation
            stable_frame[:eye_box[1], :eye_box[0]] = eye_frame


            # ===== FACE AREA BOUNDING BOX =====

            face_bounding_box = minimal_bounding_box_from_landmarks(stable_frame, stable_landmarks)
            cv2.rectangle(stable_frame, face_bounding_box[0], face_bounding_box[1], (0, 0, 255), 1)
            
            if not valid_face_box_boundaries:
                face_bounding_box_shape = rectangle_dimensions(face_bounding_box)
                valid_face_box_boundaries = True

            face_box = resize_dimensions(face_bounding_box_shape, 0.5)
    
            face_frame = crop_frame(stable_frame, face_bounding_box)
            face_frame = cv2.resize(face_frame, face_box, 3)

            # put frame at bottom right of stable frame for vizualisation
            stable_frame[stable_frame_h - face_box[1]:, stable_frame_w - face_box[0]:] = face_frame


            cv2.imshow('Stable frame', stable_frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()



# ========== LANDMARK RELATED FUNCTIONS ==========

def apply_face_mesh(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = face_mesh.process(rgb_frame)
    landmark_points = output.multi_face_landmarks

    return landmark_points

def average_landmark_coordinates(frame, landmarks, index_list):
    frame_h, frame_w = frame.shape[:2]

    center_x, center_y = 0, 0

    # Iterate through indices instead of landmarks
    for index in index_list:
        # Accumulate coordinates for center calculation
        center_x += landmarks[index].x * frame_w
        center_y += landmarks[index].y * frame_h
    
    # Calculate the average of the coordinates
    center_x = int(center_x / len(index_list))
    center_y = int(center_y / len(index_list))

    return center_x, center_y

def map_landmark_to_frame(frame, landmarks, index):
    return int(landmarks[index].x * frame.shape[1]), int(landmarks[index].y * frame.shape[0])

"""
Vertices on the face that will be useful at some point. Set show_vectors to true
to draw the vectors on the frame. The points chosen are my attempt at finding the best
bounding box for the area of interest (the eyes).
"""
def calculate_face_vertices(frame, landmarks, show_vertices=False):

    # Vertices that cross the middle of the nose ridge
    # necessary to calculate the angle of rotation of the face
    vertexA = map_landmark_to_frame(frame, landmarks, LOWER_CENTER_NOSE_RIDGE_LANDMARK)
    vertexB = map_landmark_to_frame(frame, landmarks, UPPER_CENTER_NOSE_RIDGE_LANDMARK)
    vertexC = map_landmark_to_frame(frame, landmarks, MIDDLE_CENTER_NOSE_RIDGE_LANDMARK)

    # Outer corners of the eyes
    left_corner = map_landmark_to_frame(frame, landmarks, OUTER_LEFT_CORNER_LANDMARK)
    right_corner = map_landmark_to_frame(frame, landmarks, OUTER_RIGHT_CORNER_LANDMARK)

    # Function of the line that passes through the two outer corners of the eys
    corner_line_function = line_equation(left_corner, right_corner)

    # Midpoint of the line that passes through the two outer corners of the eyes
    corner_midpoint = vertexC[0], int(corner_line_function(vertexC[0]))

    if show_vertices:

        # Draw line from tip of the nose to outer eye corners
        cv2.line(frame, vertexA, left_corner, (0, 255, 0), 1)
        cv2.line(frame, vertexA, right_corner, (0, 255, 0), 1)

        # Draw line from top of the nose ridge to outer eye corners
        cv2.line(frame, vertexB, left_corner, (0, 255, 0), 1)
        cv2.line(frame, vertexB, right_corner, (0, 255, 0), 1)

        # Draw line from middle of nose ridge, which separate the sides of the face
        cv2.line(frame, vertexA, corner_midpoint, (0, 0, 255), 1)
        cv2.line(frame, vertexB, corner_midpoint, (0, 0, 255), 1)

        cv2.line(frame, left_corner, right_corner, (0, 0, 255), 1)
        cv2.line(frame, vertexA, vertexB, (255, 0, 0), 1)
    
    return vertexA, vertexB, left_corner, right_corner, corner_midpoint

"""
Returns a vector (magnitude, angle) representation of the segment passing through 
the landmarks at index1 and index2, mapped to the given frame.
"""
def vector_of(frame, landmarks, index1, index2):

    point1 = map_landmark_to_frame(frame, landmarks, index1)
    point2 = map_landmark_to_frame(frame, landmarks, index2)

    return calculate_magnitude(point1, point2), calculate_angle(point1, point2)

# ========== BOUNDING BOX-RELATED FUNCTIONS ==========

def minimal_bounding_box_from_coordinates(frame, coordinates, map_to_frame=False):
    frame_h, frame_w = frame.shape[:2]

    x_max, y_max = 0, 0
    x_min, y_min = frame_w, frame_h

    for point in coordinates:
        x, y = point

        if x > x_max and x < frame_w:
            x_max = x

        elif x < x_min and x > 0:
            x_min = x
        
        if y > y_max and y < frame_h:
            y_max = y 

        elif y < y_min and y > 0:
            y_min = y 

    # Option to map to frame if not already done
    if map_to_frame:
        x_min, y_min = int(x_min * frame_w), int(y_min * frame_h)
        x_max, y_max = int(x_max * frame_w), int(y_max * frame_h)

    return (x_min, y_max), (x_max, y_min)



"""
Returns the minimum bounding box that contains the face.
The bounding box is a set of two integer coordinatesthat corresponds to
(lower_left_corner, upper_right_corner). 
"""
def minimal_bounding_box_from_landmarks(frame, landmarks):
    frame_h, frame_w = frame.shape[:2]

    x_max, y_max = 0, 0
    x_min, y_min = frame_w, frame_h
    
    # Find the minimum and maximum x and y coordinates of the landmarks
    for landmark in landmarks:

        # This one is slightly different from the minimal_bounding_box_with_coordinates function because of this
        # Temporary solution for now
        x, y = landmark.x, landmark.y

        if x < x_min and x > 0:
            x_min = x
        
        elif x > x_max and x < frame_w:
            x_max = x

        elif y < y_min and y > 0 :
            y_min = y 
        
        elif y > y_max and y < frame_h:
            y_max = y
         

    # Map coordinates to frame dimensions
    lower_left_corner = int(x_min * frame_w), int(y_min * frame_h)
    upper_right_corner = int(x_max * frame_w), int(y_max * frame_h)
    
    return lower_left_corner, upper_right_corner

"""
Returns the dimensions (height, width) of the rectangle defined by
the bounding box. The bounding box is a set of two points that should correspond to
opposite corners of the rectangle.
"""
def rectangle_dimensions(bounding_box):
    lower_left_corner, upper_right_corner = bounding_box

    box_h = abs(upper_right_corner[1] - lower_left_corner[1])
    box_w = abs(upper_right_corner[0] - lower_left_corner[0])

    return box_h, box_w

"""
Returns the frame section contained in the bounding box
The bounding box is a set of two points that should correspond to
opposite corners of the rectangle.
"""
def crop_frame(frame, bounding_box):
    # Sort coordinates by axis
    corner_x = bounding_box[0][0], bounding_box[1][0]
    corner_y = bounding_box[0][1], bounding_box[1][1]
    
    lower_corner_x = min(corner_x)
    lower_corner_y = min(corner_y)

    upper_corner_x = max(corner_x)
    upper_corner_y = max(corner_y)
    
    return frame[lower_corner_y:upper_corner_y, lower_corner_x:upper_corner_x]

# ========== IMAGE ROTATION FUNCTIONS ==========

def rotate_frame(frame, angle):
    frame_h, frame_w = frame.shape[:2]
    center = (frame_w // 2, frame_h // 2)

    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_frame = cv2.warpAffine(frame, rotation_matrix, (frame_w, frame_h))

    return rotated_frame

# ========== GEOMETRY FUNCTIONS ==========

"""
Calculate angle between the vector (point1, point2) and the horizontal axis (?)
in radians 
"""
def calculate_angle(point1, point2):
    # Lot of jittering when the rotation is too big
    # TODO: reduce sensitivity? 

    # it is also jitterey because when the face is rotated too much, the landmarks are not detected properly
    # a solution could be to put angle limits on the frame rotation

    slope = calculate_slope(point1, point2) 
    return np.arctan(slope) * 180 / np.pi


"""
Calcul de la distance entre deux points (ou norme d'un vecteur)
"""
def calculate_magnitude(point1, point2):
    x1, y1 = point1
    x2, y2 = point2

    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

"""
Calcul de la pente d'une droite
"""
def calculate_slope(point1, point2):
    x1, y1 = point1
    x2, y2 = point2

    return (y2 - y1) / (x2 - x1)

"""
Calcul de l'ordonnée à l'origine d'une droite passant par un point
"""
def calculate_intercept(point, slope):
    x, y = point
    return y - slope * x

"""
Returns a lambda function that can be used to calculate a points that lies on the line
that passes through the two points given as arguments
"""
def line_equation(point1, point2):
    slope = calculate_slope(point1, point2)
    intercept = calculate_intercept(point1, slope)
    return lambda x: slope * x + intercept


"""
Returns the dimensions (height, width) resized by the given factor while keeping the aspect ratio
"""
def resize_dimensions(original_dimensions, resize_factor):
    ratio = original_dimensions[0] / original_dimensions[1]

    new_height = int(original_dimensions[0] * resize_factor)
    new_width = int(new_height * ratio)

    return new_height, new_width



if __name__ == "__main__":
    main()