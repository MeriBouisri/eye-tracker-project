import cv2
import numpy as np

from face_frame import FaceFrame
from eye_keypoints import *
from utils import image_utils
from utils import geometry_utils
from utils import draw_utils

class NoIrisFound(Exception):
    pass

class NoFittingEllipseFound(Exception):
    pass

class IrisEllipse:
    """
    The IrisEllipse class is used to extract the irises from a FaceFrame object and fit an ellipse to them
    """
    def __init__(self, face_frame: FaceFrame):
        """
        Parameters
        ----------
        face_frame : FaceFrame
            The FaceFrame object from which the irises will be extracted. 
        """
        self.face_frame = face_frame

    def fit_ellipse_to_iris(self, eye_id):
        """
        Extracts the iris from the FaceFrame object and fits an ellipse to it. This method returns the 
        ellipse in the format of openCV's ellipse function. This method also returns the ROI of the iris, in order
        to determine its location in the FaceFrame.

        Parameters
        ----------
        eye_id : str | int | EyeID
            The id of the eye to fit the ellipse to.
            The eye_id can be specified as a string ('left' or 'right'), an integer (0 or 1), or an EyeID object (EyeID.left or EyeID.right).

        Returns
        ----------
        iris_ellipse, iris_roi : 
            The iris_ellipse is the ellipse fitted to the iris -> ((center_x, center_y), (major_axis, minor_axis), angle).

            The iris_roi is the rectangle in which the iris is located in the FaceFrame object -> (upper_left_corner, lower_right_corner).

        Raises
        ----------
        NoIrisFound :
            Raised when the iris ROI cannot be found in the FaceFrame object, such as when the eye is out of the frame and the landmarks could not be detected.

        NoFittingEllipseFound :
            Raised when no fitting ellipse could be found for the iris (less than 5 contour points). Also raised when the ellipse box points are invalid.

        See Also
        ----------
        EyeDict : 
            For more information on the eye_id parameter.
        """
        
        iris_roi = self.face_frame.face_mesh.get_iris_roi(eye_id)
        iris_frame = self.face_frame.crop_frame(*iris_roi)

        if iris_frame is None:
            raise NoIrisFound("No iris found for eye_id : {}".format(eye_id))
        
        gray_scale_frame = cv2.cvtColor(iris_frame, cv2.COLOR_BGR2GRAY)
        gray_scale_frame = cv2.equalizeHist(gray_scale_frame)

        retinex_frame = np.array(image_utils.apply_single_scale_retinex(gray_scale_frame, 100))
        normalized_retinex_frame = cv2.normalize(src=retinex_frame, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        normalized_retinex_frame = cv2.equalizeHist(normalized_retinex_frame)

        _, thresh = cv2.threshold(normalized_retinex_frame, 160, 255, cv2.THRESH_BINARY)
        canny_transform = cv2.Canny(thresh, 240, 255)

        contours, hierarchy = cv2.findContours(canny_transform, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contour_points = []

        for contour in contours:
            for point in contour:
                contour_points.append(point[0])

        # Error checking for valid cv2.fitEllipse input
        if len(contour_points) < 5:
            raise NoFittingEllipseFound("No fitting ellipse found for eye_id: {}".format(eye_id))
        
        iris_ellipse = cv2.fitEllipse(np.array(contour_points))

        # Error checking for ellipse box
        box = cv2.boxPoints(iris_ellipse)
        box_width = np.abs(box[0][0] - box[1][0])
        box_height = np.abs(box[0][1] - box[1][1])

        if box_width < 0 and box_height < 0:
            raise NoFittingEllipseFound("No fitting ellipse found for eye_id: {}".format(eye_id))

        return iris_ellipse, iris_roi
    