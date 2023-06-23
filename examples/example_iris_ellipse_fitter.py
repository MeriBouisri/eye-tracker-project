import cv2
import numpy as np

from eye_tracking.face_frame import FaceFrame
from eye_tracking.iris_ellipse_extractor import IrisEllipseExtractor
from eye_tracking.eye_keypoints import eye_dict
from eye_tracking.exceptions import NoFaceFound, NoIrisFound, NoFittingEllipseFound

from eye_tracking.utils import draw_utils

def example_iris_fitting():
    """
    The following demo will show the iris ellipse fitting algorithm.
    Call this function in main.py to run the demo.
    """
    cam = cv2.VideoCapture(0)
    face_frame = FaceFrame()

    # Create two instances of the IrisEllipseFitter class, one for each eye
    left_iris = IrisEllipseExtractor(face_frame, 'left')
    right_iris = IrisEllipseExtractor(face_frame, 'right')

    while True:
        _, frame = cam.read()
        frame = cv2.flip(frame, 1)
        
        try:
            # Throws FaceNotFound exception if no face is found
            face_frame.update_frame(frame)

            try:
                """
                FIT IRIS ELLIPSE FUNCTION

                Fit an ellipse to the iris of each eye

                The function returns an ellipse in the following format:

                    ellipse = center, axes, angle
                        center = (center_x, center_y) : tuple(float, float)
                        axes = (major_axis, minor_axis) : tuple(float, float)
                        angle : float
                
                The relative_to_roi parameter is set to False to return the ellipse coordinates relative to the frame.
                The draw_ellipse parameter is set to true to visualize the ellipse on the frame automatically.

                The function throws NoIrisFound and NoFittingEllipseFound exceptions, which should be caught.
                """
                left_iris_ellipse = left_iris.get_iris_ellipse(relative_to_roi=False, draw_ellipse=True)
                right_iris_ellipse = right_iris.get_iris_ellipse(relative_to_roi=False, draw_ellipse=True)

                # Get the center coordinates of the ellipse relative to the full frame
                left_iris_center = left_iris_ellipse[0]
                right_iris_center = right_iris_ellipse[0]

                # Draw cross lines at the center of the irises to visualize the center coordinates better
                draw_utils.draw_cross_lines(frame, left_iris_center, color=(255, 0, 0))
                draw_utils.draw_cross_lines(frame, right_iris_center, color=(0, 0, 255))

                # Write values of center coordinates at the bottom of the frame
                bottom_text_y = frame.shape[0] - 40
                cv2.putText(frame, f'Left Iris Center: {str(left_iris_center)}', (20,bottom_text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                cv2.putText(frame, f'Right Iris Center: {str(right_iris_center)}', (20, bottom_text_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
            except NoIrisFound:
                """
                Error handling for NoIrisFound exception thrown by the fit_iris_ellipse function.
                Caused by the iris not being detected by the face mesh.
                """
                pass

            except NoFittingEllipseFound:
                """
                Error handling for NoFittingEllipseFound exception thrown by the fit_iris_ellipse function
                Caused by the iris ellipse fitter not being able to fit an ellipse to the iris.
                """
                pass

        except NoFaceFound:
            """
            Error handling for FaceNotFound exception thrown by the update_frame function.
            Caused by the face mesh not being able to detect a face in the frame.
            """
            pass
        
        # Show the image outside of the try/except blocks so that the frame doesnt freeze
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    example_iris_fitting()