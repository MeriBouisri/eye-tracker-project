import cv2
import numpy as np

from eye_tracking.face_frame import FaceFrame
from eye_tracking.eye_keypoints import eye_dict
from eye_tracking.iris_ellipse_extractor import IrisEllipseExtractor
from eye_tracking.corneal_reflection_extractor import CornealReflectionExtractor
from eye_tracking.exceptions import NoFaceFound, NoIrisFound, NoFittingEllipseFound, NoCornealReflectionFound

from eye_tracking.utils import image_utils
from eye_tracking.utils import geometry_utils
from eye_tracking.utils import draw_utils



def example_corneal_reflection():
    cam = cv2.VideoCapture(0)

    face_frame = FaceFrame()

    left_iris_ellipse_extractor = IrisEllipseExtractor(face_frame, 0)
    left_reflection_extractor = CornealReflectionExtractor(left_iris_ellipse_extractor)

    right_iris_ellipse_extractor = IrisEllipseExtractor(face_frame, 1)
    right_reflection_extractor = CornealReflectionExtractor(right_iris_ellipse_extractor)

    while True:
        _, frame = cam.read()
        frame = cv2.flip(frame, 1)

        try:
            
            face_frame.update_frame(frame)

            left_iris_frame = face_frame.get_iris_roi_frame(0)
            right_iris_frame = face_frame.get_iris_roi_frame(1)

            try:
                # ========== LEFT EYE ==========

                left_iris_ellipse = left_iris_ellipse_extractor.get_iris_ellipse(relative_to_roi=True,draw_ellipse=True)
                left_corneal_reflection = left_reflection_extractor.get_corneal_reflection(relative_to_roi=True, draw_rect=True)

                left_iris_ellipse_center = left_iris_ellipse[0]
                left_corneal_reflection_center = left_corneal_reflection[0]

                draw_utils.draw_cross_lines(left_iris_frame, left_iris_ellipse_center, color=(0,255, 0))
                draw_utils.draw_cross_lines(left_iris_frame, left_corneal_reflection_center, color=(0, 0, 255))

            except NoCornealReflectionFound:
                pass

            try:

                # ========== RIGHT EYE ==========

                right_iris_ellipse = right_iris_ellipse_extractor.get_iris_ellipse(relative_to_roi=True,draw_ellipse=True)
                right_corneal_reflection = right_reflection_extractor.get_corneal_reflection(relative_to_roi=True, draw_rect=True)

                right_iris_ellipse_center = right_iris_ellipse[0]
                right_corneal_reflection_center = right_corneal_reflection[0]

                draw_utils.draw_cross_lines(right_iris_frame, right_iris_ellipse_center, color=(0, 255, 0))
                draw_utils.draw_cross_lines(right_iris_frame, right_corneal_reflection_center, color=(0, 0, 255))

            except NoCornealReflectionFound:
                pass
            
        except NoIrisFound:
            pass

        except NoFittingEllipseFound:
            pass

        
        except NoFaceFound:
            pass

        cv2.imshow('frame', frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

