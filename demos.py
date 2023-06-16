import cv2
import numpy as np

from camera import Camera
from face_frame import FaceFrame
from face_mesh import RawFaceMesh, FaceNotFound
from iris_extraction import IrisEllipse, NoIrisFound
from eye_keypoints import eye_dict

from utils import draw_utils
from utils import geometry_utils
from utils import image_utils


def iris_ellipse_demo():
    """
    The following demo will show the iris ellipse fitting algorithm.
    Call this function in main.py to run the demo.

    Notes
    ----------
    There is an issue with the ellipse fitting for the right eye. The ellipse often extends way beyond the
    iris region, even though the region should be confined to the iris landmarks. This does not seem to happen
    with the left eye.
    I haven't looked into it yet, but I will soon.
    """
    cam = Camera(0)
    face_frame = FaceFrame()
    iris_ellipse = IrisEllipse(face_frame)

    while True:
        frame = cam.get_frame()
        
        try:
            face_frame.update_frame(frame)

            left_eye_frame = face_frame.get_iris_frame(0)
            right_eye_frame = face_frame.get_iris_frame(1)

            left_iris_ellipse, _ = iris_ellipse.fit_ellipse_to_iris(0)
            right_iris_ellipse, _ = iris_ellipse.fit_ellipse_to_iris(1)

            cv2.ellipse(left_eye_frame, left_iris_ellipse, (0, 255, 0), 1)
            cv2.ellipse(right_eye_frame, right_iris_ellipse, (0, 255, 0), 1)

            left_iris_center = left_iris_ellipse[0]
            right_iris_center = right_iris_ellipse[0]

            draw_utils.draw_cross(left_eye_frame, left_iris_center, length=5)
            draw_utils.draw_cross(right_eye_frame, right_iris_center, length=5)

            left_iris_landmarks = face_frame.face_mesh.get_landmarks(eye_dict.left.iris)
            right_iris_landmarks = face_frame.face_mesh.get_landmarks(eye_dict.right.iris)

            draw_utils.draw_all_crosses(frame, left_iris_landmarks, (0, 255, 0))
            draw_utils.draw_all_crosses(frame, right_iris_landmarks, (0, 255, 0))

            cv2.imshow('frame', frame) 

        except FaceNotFound:
            pass

        except NoIrisFound:
            pass

        if cv2.waitKey(1) == ord('q'):
            break

    cam.release()

def gaze_vector_2d_demo():
    """
    The following code was the one that I showed during the lab meeting of 15/06/2023 .
    It showed a vector going from the center of the iris to the rest of the screen, which allowed us
    to approximate the direction of the gaze vector in a 2D space.

    Due to changes to the iris_extraction module, this code is no longer valid. 
    I will be updating it soon.
    """
    raise NotImplementedError("This demo is no longer functional.")

    cam = Camera()
    face_frame = FaceFrame()
    left_pupil = Pupil(face_frame, 0)

    while True:
        frame = cam.get_frame()

        try:
            face_frame.update_frame(frame)
            left_pupil.gaze_vectors()

            cv2.ellipse(frame, left_pupil.scaled_pupil_ellipse, (0, 255, 0), 1)
            cv2.line(frame, left_pupil.iris_center, left_pupil.eye_center, (0, 255, 0),2)
        
            eq = geometry_utils.get_linear_equation(left_pupil.iris_center, left_pupil.eye_center)

            start_range_x = left_pupil.iris_center[0]
            end_range_x = left_pupil.eye_center[0]

            x_diff = left_pupil.iris_center[0] - left_pupil.eye_center[0]
            y_diff = left_pupil.iris_center[1] - left_pupil.eye_center[1]

            # temporary, for testing purposes

            delta = 1000
            if x_diff > 0 :
                start_range_x = left_pupil.eye_center[0]
                end_range_x = left_pupil.iris_center[0] + delta

            elif x_diff < 0 :
                start_range_x = left_pupil.iris_center[0] - delta
                end_range_x = left_pupil.eye_center[0]

            draw_utils.draw_equation_line(frame,(start_range_x, end_range_x), eq, (0, 0, 255), 1)

            cv2.imshow('frame', frame)

        except FaceNotFound:
            pass

        

        if cv2.waitKey(1) == ord('q'):
            break

    cam.release()