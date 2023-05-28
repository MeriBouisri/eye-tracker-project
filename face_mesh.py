import mediapipe as mp
import numpy as np
import cv2

from enum import Enum

import draw_utils
import math_utils


class FaceNotFound(Exception):
    pass

class Landmark():
    """
    Refer to the following link for the landmark indices: https://i.stack.imgur.com/T1ypF.jpg
    """

    # TODO: Organize landmark constants better. Dictionary maybe

    LEFT = 0
    RIGHT = 1

    LOWER_CENTER_NOSE_RIDGE_LANDMARK = 197
    MIDDLE_CENTER_NOSE_RIDGE_LANDMARK = 168
    UPPER_CENTER_NOSE_RIDGE_LANDMARK = 8

    OUTER_EYE_CORNER_LANDMARKS = [33, 263]

    IRIS_LANDMARKS = [[474, 475, 476, 477], 
                      [469, 470, 471, 472]]
    

class FaceMesh():

    face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

    def __init__(self):
        pass

    def update_frame(self, frame):
        """
        Update the frame to be processed.
        """
        self.frame = frame
        self.frame_h, self.frame_w, _ = self.frame.shape

    def apply_face_mesh(self):
        """
        Apply the face mesh to the frame. Throws a FaceNotFound exception if landmarks are not found.
        """
        rgb_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        output = self.face_mesh.process(rgb_frame)
        self.landmark_points = output.multi_face_landmarks

        if not self.landmark_points: raise FaceNotFound()


    def get_landmark(self, index):
        """
        Returns the (x, y) coordinates of the gace mesh landmark at the given index. 
        The coordinates are not yet scaled to the frame's dimensions.
        Helper function for getting the coordinates as a tuple.
        """
        landmark = self.landmark_points[0].landmark[index]
        return landmark.x, landmark.y
    
    
    def map_landmarks_to_frame(self, indices=[]):
        """
        
        """

        if len(indices) == 0: indices = range(len(self.landmark_points[0].landmark))
        callables = [lambda x: x * self.frame_w, lambda x: x * self.frame_h]
        landmarks = [self.get_landmark(index) for index in indices] 
        return  math_utils.map_by_column(landmarks, callables)



# ------------------------------
# EXAMPLE USAGE
# ------------------------------

if __name__ == '__main__':
    cam = cv2.VideoCapture(0)
    mesh = FaceMesh()

    while True:
        _, frame = cam.read()
        frame = cv2.flip(frame, 1)
        mesh.update_frame(frame)

        try:
            mesh.apply_face_mesh()
            f = np.array(Landmark.OUTER_EYE_CORNER_LANDMARKS).flatten()
            flat = np.array(Landmark.IRIS_LANDMARKS).flatten()

            flat = np.concatenate((f, flat))

            draw_utils.draw_convex_hull(frame, mesh.map_landmarks_to_frame())



        except FaceNotFound:
            pass

        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()
                
    


    
