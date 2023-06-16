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

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
