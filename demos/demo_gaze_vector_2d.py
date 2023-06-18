import cv2
import numpy as np

from eye_tracking.face_frame import FaceFrame
from eye_tracking.face_mesh import RawFaceMesh, FaceNotFound
from eye_tracking.iris_extraction import IrisEllipse, NoIrisFound, NoFittingEllipseFound
from eye_tracking.eye_keypoints import eye_dict

from eye_tracking.utils import draw_utils
from eye_tracking.utils import geometry_utils
from eye_tracking.utils import image_utils

from eye_tracking.camera.camera import Camera



def main():
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

if __name__ == '__main__':
    main()