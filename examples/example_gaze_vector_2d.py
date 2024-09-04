import cv2
import numpy as np

from eye_tracking.face_frame import FaceFrame
from eye_tracking.iris_ellipse_extractor import IrisEllipseExtractor
from eye_tracking.corneal_reflection_extractor import CornealReflectionExtractor
from eye_tracking.exceptions import NoFaceFound, NoIrisFound, NoFittingEllipseFound, NoCornealReflectionFound

from eye_tracking.utils import draw_utils
from eye_tracking.utils import image_utils
from eye_tracking.utils import geometry_utils

def example_gaze_vector_2d():
    cam = cv2.VideoCapture(0)

    face_frame = FaceFrame()

    left_iris_extractor = IrisEllipseExtractor(face_frame, 'left')
    left_reflection_extractor = CornealReflectionExtractor(left_iris_extractor)

    right_iris_extractor = IrisEllipseExtractor(face_frame, 'right')
    right_reflection_extractor = CornealReflectionExtractor(right_iris_extractor)

    while True:
        _, frame = cam.read()
        frame = cv2.flip(frame, 1)

        try:
            face_frame.update_frame(frame)

            try:
                left_iris_frame = face_frame.get_iris_roi_frame('left')
                right_iris_frame = face_frame.get_iris_roi_frame('right')

                try:

                    left_iris = left_iris_extractor.get_iris_ellipse(relative_to_roi=False, draw_ellipse=True)
                    left_reflection = left_reflection_extractor.get_corneal_reflection(relative_to_roi=False, draw_rect=True)

                    left_iris_center = left_iris[0]
                    left_reflection_center = left_reflection[0]

                    vector_equation = geometry_utils.get_linear_equation(left_iris_center, left_reflection_center)

                    delta_x = left_iris_center[0] - left_reflection_center[0]

                    start_range_x = 0
                    end_range_x = left_reflection_center[0]

                    if delta_x > 0:
                        start_range_x = left_reflection_center[0]
                        end_range_x = frame.shape[1]

                    domain_range = [start_range_x, end_range_x]
                    draw_utils.draw_equation_line(frame, domain_range, vector_equation, color=(0, 255, 0))

                    draw_utils.draw_cross_lines(left_iris_frame, left_iris_center, color=(0,255, 0))
                    draw_utils.draw_cross_lines(left_iris_frame, left_reflection_center, color=(0, 0, 255))

                except NoCornealReflectionFound:
                    pass

                try:
                    right_iris = right_iris_extractor.get_iris_ellipse(relative_to_roi=False, draw_ellipse=True)
                    right_reflection = right_reflection_extractor.get_corneal_reflection(relative_to_roi=False, draw_rect=True)

                    right_iris_center = right_iris[0]
                    right_reflection_center = right_reflection[0]

                    vector_equation = geometry_utils.get_linear_equation(right_iris_center, right_reflection_center)

                    delta_x = right_iris_center[0] - right_reflection_center[0]

                    start_range_x = 0
                    end_range_x = right_reflection_center[0]

                    if delta_x > 0:
                        start_range_x = right_reflection_center[0]
                        end_range_x = frame.shape[1]

                    domain_range = [start_range_x, end_range_x]
                    draw_utils.draw_equation_line(frame, domain_range, vector_equation, color=(0, 255, 0))

                    draw_utils.draw_cross_lines(right_iris_frame, right_iris_center, color=(0, 255, 0))
                    draw_utils.draw_cross_lines(right_iris_frame, right_reflection_center, color=(0, 0, 255))

                except NoCornealReflectionFound:
                    pass
            
            except (NoIrisFound, NoFittingEllipseFound):
                pass

        except NoFaceFound:
            pass

        cv2.imshow('frame', frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

