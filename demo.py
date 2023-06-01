import cv2
import numpy as np

from face_frame import FaceFrame
from stable_face_frame import StableFaceFrame
from face_mesh import FaceNotFound


from utility.geometry_utils import *
from utility.draw_utils import *
from landmark_constants import *

"""
Quick demo of what we have so far, and how we can use these tools for advancing the project.
"""

cam = cv2.VideoCapture(0)

# Initialize the FaceFrame and StableFaceFrame instances
# The FaceFrame instance is used to detect the face and landmarks
# The StableFaceFrame instance is the result of filtering out unnecessary movement from the FaceFrame
face_frame = FaceFrame()

# Dont forget to bind the stable frame to the necessary face frame
stable_frame = StableFaceFrame(face_frame)

while True:
    # Read the frame from the camera
    _, frame = cam.read()
    frame = cv2.flip(frame, 1)

    # Create a blank frame to draw on
    blank = np.zeros((frame.shape), dtype=np.uint8)

    try:
        # 
        stable_frame.update_frame(frame)

        # Determine the coordinates of the landmarks that you need
        # You can access them through pre-defined methods
        stable_landmarks = stable_frame.face_mesh.get_landmarks()
        stable_iris_landmarks = stable_frame.get_iris_landmarks()

        # We can distinguish between stable landmarks (after stabilization) and raw landmarks (before)
        stable_pupil_landmarks = stable_frame.get_pupil_center_landmarks()
        raw_pupil_landmarks = stable_frame.parent_frame.get_pupil_center_landmarks()

        # Or you can access them directly through the face mesh instance
        stable_eye_zone_landmarks = stable_frame.face_mesh.get_landmarks(EYE_AREA_KEYPOINTS)
        
        # You can also extract data about the movement of the stable frame
        stable_h_vector = stable_frame.horizontal_vector
        stable_v_vector = stable_frame.vertical_vector

        # Or the data from its parent frame
        parent_h_vector = stable_frame.parent_frame.horizontal_vector.get_geometric_vector()
        parent_v_vector = stable_frame.parent_frame.vertical_vector.get_geometric_vector()

        # Extract different areas from the stable mesh, and get their exact bounding vertices
        face_convex_hull = get_convex_hull(stable_landmarks)
        eye_convex_hull = get_convex_hull(stable_eye_zone_landmarks)

        # You can also get the minimum bounding rectangle instead of polygons
        face_rectangle = stable_frame.get_face_rectangle()
        eye_rectangle = stable_frame.get_eye_zone_rectangle()

        # Now, let's start drawing and displaying the info we have so far 
        cv2.putText(blank, f'left pupil coordinates (raw) : {raw_pupil_landmarks[0]}', (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
        cv2.putText(blank, f'right pupil coordinates (raw) : {raw_pupil_landmarks[1]}', (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
        cv2.putText(blank, f'left pupil coordinates (stable) : {stable_pupil_landmarks[0]}', (0, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
        cv2.putText(blank, f'right pupil coordinates (stable) : {stable_pupil_landmarks[1]}', (0, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
        
        cv2.putText(blank, f'Horizontal rotation: {parent_h_vector[1]}', (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
        cv2.putText(blank, f'Horizontal magnitude : {parent_h_vector[0]}', (0, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
        cv2.putText(blank, f'Scale factor : {stable_frame.parent_frame.get_scale_factor()}', (0, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
        

        # We can draw other stuff on the frame with the data we have, for visualization purposes
        draw_all_crosses(blank, stable_pupil_landmarks, color=(0, 0, 255), length=5)
        draw_cross(blank, stable_h_vector.get_center_coordinates(), color=(255, 255, 255), length=5)
        stable_h_vector.draw_landmark_vector(blank, color=(0, 255, 0))
        stable_v_vector.draw_landmark_vector(blank, color=(0, 255, 0))

        stable_iris_landmarks = np.array(stable_iris_landmarks).astype(np.int32)

        left_iris = cv2.fitEllipse(stable_iris_landmarks[0])
        right_iris = cv2.fitEllipse(stable_iris_landmarks[1])

        cv2.ellipse(blank, left_iris, (255, 255, 255), 1)
        cv2.ellipse(blank, right_iris, (255, 255, 255), 1)

        # Draw the polygons and rectangles on the frame of your choice
        draw_polygon(blank, face_convex_hull, color=(255, 255, 255))
        draw_polygon(blank, eye_convex_hull, color=(255, 0, 0))
        draw_polygon(blank, face_rectangle, color=(0, 255, 0))
        draw_polygon(blank, eye_rectangle, color=(0, 0, 255))

        draw_convex_hull(blank, stable_landmarks, color=(255, 255, 255))
        draw_convex_hull(blank, stable_frame.face_mesh.get_landmarks(EYE_AREA_KEYPOINTS), color=(255, 255, 255))
        draw_all_crosses(blank, stable_frame.get_pupil_center_landmarks(), color=(0, 0, 255))

    except FaceNotFound:
            pass

    cv2.imshow('frame', blank)

    if cv2.waitKey(1) == ord('q'):
            break
    
cam.release()
cv2.destroyAllWindows()
