import cv2
import numpy as np

from eye_tracking.face_frame import FaceFrame
from eye_tracking.exceptions import NoIrisFound, NoFittingEllipseFound
from eye_tracking.eye_keypoints import eye_dict, EyeID

from eye_tracking.utils import image_utils
from eye_tracking.utils import geometry_utils
from eye_tracking.utils import draw_utils

class IrisEllipseExtractor:
    """
    This class contains the necessary methods to extract the iris from a FaceFrame object and fit an ellipse to it.
    Each eye (left, right) contained in the FaceFrame should have its own associated IrisEllipseFitter object.

    Attributes
    ----------
    face_frame : FaceFrame
        The FaceFrame object from which the irises will be extracted. 

    eye_id : str | int | EyeID
        The id that determines which eye to extract the iris from (left eye or right eye). 
        Can be either a string ('left', 'right'), int (0, 1), or EyeID (EyeID.LEFT, EyeID.RIGHT).

    iris_ellipse :
        The iris_ellipse is the ellipse fitted to the iris relative to the iris ROI.
        The iris_ellipse attribute is updated at every call of the get_iris_ellipse method.

    Notes
    ----------
    The iris_ellipse's center is relative to the iris ROI, not the associated face_frame.
    To get the iris_ellipse's center relative to the face_frame, use the associated roi to translate back.
    """
    def __init__(self, face_frame: FaceFrame, eye_id):
        self.face_frame = face_frame
        self.eye_id = eye_id

        self.iris_ellipse = None

    def get_iris_ellipse(self, relative_to_roi=False, draw_ellipse=False):
        """
        Extracts the iris from the FaceFrame object and fits an ellipse to it. This method returns the 
        ellipse in the format of openCV's ellipse function.

        Parameters
        ----------

        relative_to_roi : bool, optional = True
            If True, the center coordinates of the ellipse will be relative to the iris ROI, not the frame associated with this instance's FaceFrame.
            If False, the center coordinates of the ellipse will be relative to the frame.
            Set to True

        draw_ellipse : bool, optional = False
            If True, the iris ellipse and center will be drawn on the frame associated with this instance's FaceFrame.

        Returns
        ----------
        iris_ellipse : 
            The iris_ellipse is the ellipse fitted to the iris -> ((center_x, center_y), (major_axis, minor_axis), angle).
            The center coordinates of the ellipse are relative to the iris ROI, not the FaceFrame.

        Raises
        ----------
        NoIrisFound :
            Raised when the iris ROI cannot be found in the FaceFrame object, such as when the eye is out of the frame and the landmarks could not be detected.

        NoFittingEllipseFound :
            Raised when no fitting ellipse could be found for the iris (less than 5 contour points). Also raised when the ellipse box points are invalid.

        See Also
        ----------
        FaceFrame.get_iris_roi : Extracts the iris ROI from the FaceFrame object.
        """
        iris_roi = self.face_frame.face_mesh.get_iris_roi(self.eye_id)
        iris_frame = self.face_frame.crop_frame(*iris_roi)

        if iris_frame is None:
            raise NoIrisFound("No iris found for eye_id : {}".format(self.eye_id))

        processed_iris_frame = IrisEllipseExtractor.process_iris_frame(iris_frame)
        contour_points = IrisEllipseExtractor.get_contour_points(processed_iris_frame)

        # Error checking for valid cv2.fitEllipse input

        if len(contour_points) < 5:
            raise NoFittingEllipseFound("No fitting ellipse found for eye_id: {}".format(self.eye_id))
        
        self.iris_ellipse = cv2.fitEllipse(np.array(contour_points))

        if not IrisEllipseExtractor.is_valid_ellipse(self.iris_ellipse):
            raise NoFittingEllipseFound("No fitting ellipse found for eye_id: {}".format(self.eye_id))
    
        
        if draw_ellipse:
            self.draw_iris_ellipse(iris_frame)

        if relative_to_roi:
            return self.iris_ellipse
        
        # Translate the ellipse center coordinates to the frame
        frame_iris_center_x = iris_roi[0][0] + self.iris_ellipse[0][0]
        frame_iris_center_y = iris_roi[0][1] + self.iris_ellipse[0][1]
        
        frame_iris_ellipse = ((frame_iris_center_x, frame_iris_center_y), self.iris_ellipse[1], self.iris_ellipse[2])

        return frame_iris_ellipse
    
    def draw_iris_ellipse(self, iris_frame):
        if self.iris_ellipse is None:
            return
        
        iris_ellipse_center = int(self.iris_ellipse[0][0]), int(self.iris_ellipse[0][1])

        cv2.ellipse(iris_frame, self.iris_ellipse, (0, 255, 0))
        draw_utils.draw_cross(iris_frame, iris_ellipse_center)
    
    @staticmethod
    def is_valid_ellipse(ellipse):
        """
        Checks if the ellipse is valid by checking if the box points are valid.

        Parameters
        ----------
        ellipse :
            The ellipse to check. Must be in the following format :
            >>> ((center_x, center_y), (major_axis, minor_axis), angle)

        Returns
        ----------
        is_valid_ellipse :
            True if the ellipse is valid, False otherwise. 

        Notes
        ----------
        Attempting to draw an invalid ellipse will result in a cv2 error. 
        """
        ellipse_box = cv2.boxPoints(ellipse)
        box_width = np.abs(ellipse_box[0][0] - ellipse_box[1][0])
        box_height = np.abs(ellipse_box[0][1] - ellipse_box[1][1])

        return not (box_width < 0 and box_height < 0)
    
    @staticmethod
    def process_iris_frame(iris_frame):
        """
        Processes the iris frame by applying a single scale retinex filter, normalizing the frame, and applying a canny transform.
        The iris frame will then be ready for contour detection.

        Parameters
        ----------
        iris_frame : 
            The iris frame to process.

        Returns
        ----------
        processed_iris_frame :
            The processed iris frame fit for contour detection.
        """

        SIGMA = 100
        LOWER_THRESH_BOUND = 160
        UPPER_THRESH_BOUND = 255
        LOWER_CANNY_THRESH_BOUND = 240

        gray_scale_frame = cv2.cvtColor(iris_frame, cv2.COLOR_BGR2GRAY)
        gray_scale_frame = cv2.equalizeHist(gray_scale_frame)

        retinex_frame = np.array(image_utils.apply_single_scale_retinex(gray_scale_frame, SIGMA))
        normalized_retinex_frame = cv2.normalize(src=retinex_frame, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        normalized_retinex_frame = cv2.equalizeHist(normalized_retinex_frame)

        _, thresh = cv2.threshold(normalized_retinex_frame, LOWER_THRESH_BOUND, UPPER_THRESH_BOUND, cv2.THRESH_BINARY)
        canny_transform = cv2.Canny(thresh, LOWER_CANNY_THRESH_BOUND, UPPER_THRESH_BOUND)

        return canny_transform
                
    @staticmethod
    def get_contour_points(frame):
        """
        Returns the iris contour points for the given frame.

        Parameters
        ----------
        frame : 
            The frame to extract the iris contour points from.

        Returns
        ----------
        iris_contour_points :
            The iris contour points for the given frame.
        """
        contours, _ = cv2.findContours(frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        contour_points = []
        
        for contour in contours:
            for point in contour:
                contour_points.append(point[0])

        return contour_points