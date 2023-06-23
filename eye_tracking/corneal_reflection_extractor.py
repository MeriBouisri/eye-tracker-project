import cv2
import numpy as np


from eye_tracking.eye_keypoints import eye_dict, EyeID
from eye_tracking.iris_ellipse_extractor import IrisEllipseExtractor
from eye_tracking.exceptions import NoCornealReflectionFound
from eye_tracking.face_frame import FaceFrame
from eye_tracking.utils import draw_utils
from eye_tracking.utils import geometry_utils
from eye_tracking.utils import image_utils



class CornealReflectionExtractor:
    """
    The purpose of this class is to extract the corneal reflection from the iris.
    
    
    """
    def __init__(self, iris_ellipse_extractor: IrisEllipseExtractor):
        self.iris_ellipse_extractor = iris_ellipse_extractor

        self.corneal_reflection_rect = None
    
    @staticmethod
    def process_iris_frame(iris_frame, iris_ellipse):
        # TODO: Find actual justification for these values
        # These are the values I randomly chose by trial and error

        LOWER_THRESH_BOUND = 30
        UPPER_THRESH_BOUND = 255

        LOWER_CANNY_THRESH_BOUND = 0
        UPPER_CANNY_THRESH_BOUND = 200

        # Apply threshold technique to
        gray_iris_frame = cv2.cvtColor(iris_frame, cv2.COLOR_BGR2GRAY)
        gray_iris_frame = cv2.normalize(src=gray_iris_frame, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        morphed_iris_frame = cv2.morphologyEx(gray_iris_frame, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        _, thresh_iris_frame = cv2.threshold(morphed_iris_frame, LOWER_THRESH_BOUND, UPPER_THRESH_BOUND, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

        # Prepare the mask to isolate the region bounded by the iris ellipse
        iris_ellipse_mask = np.zeros(iris_frame.shape[:2], dtype=np.uint8)
        cv2.ellipse(iris_ellipse_mask, iris_ellipse, 255, -1)

        # Apply the elliptical mask to the thresholded iris frame
        masked_iris_frame = cv2.bitwise_and(thresh_iris_frame, iris_ellipse_mask)
        masked_iris_frame = cv2.normalize(src=masked_iris_frame, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

        # Apply canny edge detection to the masked iris frame
        canny_iris_frame = cv2.Canny(masked_iris_frame, LOWER_CANNY_THRESH_BOUND, UPPER_CANNY_THRESH_BOUND)

        return canny_iris_frame


    @staticmethod
    def get_corneal_reflection_contour(iris_frame, iris_ellipse_mask):
        """
        The contour we are looking for is contained within the iris ellipse mask.
        The iris ellipse mask is the largest contour in the iris frame.

        The filtering process is as follows:
            1. Filter out any contour that is not contained within another contour
            2. Filter out any contour that is part of the mask border
            3. Sort the remaining contours by area
            4. Return the largest contour
        
        """

        MAX_CONTOUR_AREA_RATIO = 0.15

        # Calculate area of the ellipse : A = pi * major_axis * minor_axis
        mask_area = np.pi * iris_ellipse_mask[1][0] * iris_ellipse_mask[1][1]

        processed_iris_frame = CornealReflectionExtractor.process_iris_frame(iris_frame, iris_ellipse_mask)

        contours, hierarchy = cv2.findContours(processed_iris_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        filtered_contours = []

        for i, contour in enumerate(contours):

            # Step 1 : Filter out any contour that is not contained within another contour
            contour_has_no_parents = hierarchy[0][i][3] == -1
            if contour_has_no_parents:
                continue
            
            # Step 2 : Filter out any contour that is part of the mask border
            contour_area = cv2.contourArea(contour)
            contour_area_ratio = contour_area / mask_area
            contour_is_part_of_mask_border = contour_area_ratio > MAX_CONTOUR_AREA_RATIO

            if contour_is_part_of_mask_border:
                continue

            # Keep the remaining contours
            filtered_contours.append(contour)

        # Step 3 : Sort the remaining contours by area
        filtered_contours = sorted(filtered_contours, key=cv2.contourArea, reverse=True)

        if len(filtered_contours) == 0:
            return None
        
        # Step 4 : Return the largest contour
        return filtered_contours[0]

    def get_corneal_reflection(self, relative_to_roi=False, draw_rect=False):
        """
        Retrieve the rotated rectangle of the minimum area enclosing the corneal reflection contour.

        This method uses this instance's iris_ellipse_extractor to retrieve the object's iris_ellipse.
        If the attribute iris_ellipse_extractor.iris_ellipse is None, then this method will call the 
        iris_ellipse_extractor.get_iris_ellipse() method to retrieve the iris ellipse and update the attribute.

        Returns
        -------
        corneal_reflection_rect : cv2.RotatedRect
            The rotated rectangle of the minimum area enclosing the corneal reflection contour.

        """
        face_frame = self.iris_ellipse_extractor.face_frame
        eye_id = self.iris_ellipse_extractor.eye_id

        iris_ellipse = self.iris_ellipse_extractor.iris_ellipse

        if iris_ellipse is None:
            iris_ellipse = self.iris_ellipse_extractor.get_iris_ellipse(relative_to_roi=True)

        iris_roi = face_frame.face_mesh.get_iris_roi(eye_id)
        iris_roi_frame = face_frame.crop_frame(*iris_roi)

        corneal_reflection_contour = CornealReflectionExtractor.get_corneal_reflection_contour(iris_roi_frame, iris_ellipse)

        if corneal_reflection_contour is None:
            raise NoCornealReflectionFound('No corneal reflection found in eye {}.'.format(eye_id))
        
        self.corneal_reflection_rect = cv2.minAreaRect(corneal_reflection_contour)

        if draw_rect:
            self.draw_corneal_reflection_rect(iris_roi_frame)

        if relative_to_roi:
            return self.corneal_reflection_rect
        
        corneal_reflection_rect_center_x = self.corneal_reflection_rect[0][0] + iris_roi[0][0]
        corneal_reflection_rect_center_y = self.corneal_reflection_rect[0][1] + iris_roi[0][1]
        corneal_reflection_rect_center = (corneal_reflection_rect_center_x, corneal_reflection_rect_center_y)

        corneal_reflection_rect = (corneal_reflection_rect_center, self.corneal_reflection_rect[1], self.corneal_reflection_rect[2])

        return corneal_reflection_rect
    
    def draw_corneal_reflection_rect(self, iris_frame):
        """
        Draw the rotated rectangle of the minimum area enclosing the corneal reflection contour on the iris frame.

        Parameters
        ----------
        iris_frame : numpy.ndarray
            The iris frame.
        corneal_reflection_rect : cv2.RotatedRect
            The rotated rectangle of the minimum area enclosing the corneal reflection contour.

        Returns
        -------
        numpy.ndarray
            The iris frame with the corneal reflection rectangle drawn on it.

        """
        if self.corneal_reflection_rect is None:
            return
        
        corneal_reflection_rect_vertices = cv2.boxPoints(self.corneal_reflection_rect)
        corneal_reflection_rect_vertices = np.int0(corneal_reflection_rect_vertices)

        cv2.drawContours(iris_frame, [corneal_reflection_rect_vertices], 0, (0, 255, 0), 1)



    
