import cv2
import numpy as np

from eye_tracking.face_frame import FaceFrame
from eye_tracking.face_mesh import RawFaceMesh, FaceNotFound
from eye_tracking.iris_extraction import IrisEllipse, NoIrisFound, NoFittingEllipseFound
from eye_tracking.eye_keypoints import eye_dict

from eye_tracking.utils import draw_utils
from eye_tracking.utils import geometry_utils
from eye_tracking.utils import image_utils

"""
The following demo will show the iris ellipse fitting algorithm.
Call this function in main.py to run the demo.
"""
def main():
    cam = cv2.VideoCapture(0)
    face_frame = FaceFrame()
    iris_ellipse = IrisEllipse(face_frame)

    while True:
        _, frame = cam.read()
        frame = cv2.flip(frame, 1)

        
        try:
            face_frame.update_frame(frame)

            left_eye_frame = face_frame.get_iris_frame(0)
            right_eye_frame = face_frame.get_iris_frame(1)

            try:

                left_iris_ellipse, _ = iris_ellipse.fit_ellipse_to_iris(0)
                left_iris_center = left_iris_ellipse[0]
                left_iris_landmarks = face_frame.face_mesh.get_landmarks(eye_dict.left.iris)

                cv2.ellipse(left_eye_frame, left_iris_ellipse, (0, 255, 0), 1)
                draw_utils.draw_cross(left_eye_frame, left_iris_center, length=5)
                draw_utils.draw_all_crosses(frame, left_iris_landmarks, (0, 255, 0))

            except NoIrisFound:
                pass

            except NoFittingEllipseFound:
                pass

            try:

                right_iris_ellipse, _ = iris_ellipse.fit_ellipse_to_iris(1)
                right_iris_center = right_iris_ellipse[0]
                right_iris_landmarks = face_frame.face_mesh.get_landmarks(eye_dict.right.iris)

                cv2.ellipse(right_eye_frame, right_iris_ellipse, (0, 255, 0), 1)
                draw_utils.draw_cross(right_eye_frame, right_iris_center, length=5)
                draw_utils.draw_all_crosses(frame, right_iris_landmarks, (0, 255, 0))

            except NoIrisFound:
                pass

            except NoFittingEllipseFound:
                pass

            
            cv2.imshow('frame', frame)

        except FaceNotFound:
            pass

        if cv2.waitKey(1) == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()