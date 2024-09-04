from eye_tracking.face_frame import FaceFrame
from eye_tracking.iris_ellipse_extractor import IrisEllipseExtractor
from eye_tracking.exceptions import *

from eye_tracking.utils import draw_utils, geometry_utils

import cv2
import numpy as np

NOSE_TIP = 4

red = (0, 0, 255)
green = (0, 255, 0)
blue = (255, 0, 0)

if __name__ == '__main__':
    cam = cv2.VideoCapture(0)

    prev_ref_point_landmark = 0, 0
    prev_ref_vector = 0, 0
    eye_center = 0, 0
    range_end_x = 0
    ref_vector_equation =  lambda x: x

    delta_x = 0
    delta_y = 0

    face_frame = FaceFrame()
    left_iris_extractor = IrisEllipseExtractor(face_frame, 'left')

    while True:
        _, frame = cam.read()

        try:
            face_frame.update_frame(frame)
            iris_ellipse = left_iris_extractor.get_iris_ellipse(relative_to_roi=False, draw_ellipse=True)
            

            iris_center = iris_ellipse[0]
            ref_point_landmark = face_frame.face_mesh.get_landmarks(NOSE_TIP)

            ref_vector = geometry_utils.as_geometric_vector(ref_point_landmark, iris_center)

            new_eye_center = ref_point_landmark[0] + delta_x, ref_point_landmark[1] + delta_y # type: ignore

            draw_utils.draw_line(frame, ref_point_landmark, iris_center, color=blue)
            draw_utils.draw_line(frame, ref_point_landmark, new_eye_center, color=green)
            draw_utils.draw_line(frame, new_eye_center, iris_center)

        
            # range_start_x = ref_point_landmark[0] # type: ignore

 


            # # draw_utils.draw_line(frame, ref_point_landmark, iris_center)
            # # draw_utils.draw_line(frame, ref_point_landmark, eye_center, color=blue)
            # # draw_utils.draw_line(frame, eye_center, iris_center, color=green)

            # draw_utils.draw_equation_line(frame, [range_start_x, range_end_x], ref_vector_equation, color=red)




            draw_utils.draw_all_crosses(frame, [iris_center, ref_point_landmark, eye_center])

            if cv2.waitKey(1) == ord('c'):
                ref_vector_equation = geometry_utils.get_linear_equation(ref_point_landmark, iris_center)
                delta_x = iris_center[0] - ref_point_landmark[0] # type: ignore
                delta_y = iris_center[1] - ref_point_landmark[1] # type: ignore
                range_end_x = ref_point_landmark[0] + delta_x # type: ignore
                prev_ref_point_landmark = ref_point_landmark
                prev_ref_vector = ref_vector

        except (NoFaceFound, NoFittingEllipseFound, NoIrisFound):
            pass

        cv2.imshow('frame', frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()
