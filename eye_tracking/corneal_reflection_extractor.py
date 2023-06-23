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

    # def get_pupil_reflection(self):
    #     frame = self.iris_ellipse_extractor.face_frame.frame
    #     kernel = np.ones((3, 3), np.uint8)

    #     iris_ellipse = self.iris_ellipse_extractor.extract_iris_ellipse(relative_to_roi=True)
    #     iris_ellipse_center = iris_ellipse[0]

    #     # iris_ellipse_center = iris_ellipse[0][0] * 10, iris_ellipse[0][1] * 10
    #     # iris_ellipse_axes = iris_ellipse[1][0] * 10, iris_ellipse[1][1] * 10
    #     # iris_ellipse_angle = iris_ellipse[2]

    #     # iris_ellipse = (iris_ellipse_center, iris_ellipse_axes, iris_ellipse_angle)


    #     self.iris_roi = self.iris_ellipse.face_frame.face_mesh.get_iris_roi(self.iris_ellipse.eye_id)
    #     self.iris_frame = self.iris_ellipse.face_frame.crop_frame(*self.iris_roi)

    #     # self.iris_frame = cv2.resize(self.iris_frame, (0, 0), fx=10, fy=10)

    #     cv2.ellipse(self.iris_frame, iris_ellipse, (0, 255, 0), 1)

    #     ellipse_mask = np.zeros(self.iris_frame.shape[:2], dtype=np.uint8)
    #     cv2.ellipse(ellipse_mask, iris_ellipse, 255, -1)

    #     gray = cv2.cvtColor(self.iris_frame, cv2.COLOR_BGR2GRAY)
    #     norm = cv2.normalize(src=gray, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    #     hist = cv2.equalizeHist(norm)
        

    #     morph = cv2.morphologyEx(norm, cv2.MORPH_OPEN, kernel)
    #     thresh = cv2.threshold(morph, 30, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    #     blank = np.ones(self.iris_frame.shape[:2], dtype=np.uint8)

    #     bitwise_mask = cv2.bitwise_and(thresh, ellipse_mask)
    #     bitwise_mask = cv2.normalize(src=bitwise_mask, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
    #     bitwise_mask_canny = cv2.Canny(bitwise_mask, 0, 200)

    #     contours, hierarchy = cv2.findContours(bitwise_mask_canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  

    #     ellipse_area = np.pi * iris_ellipse[1][0] * iris_ellipse[1][1]



    #     filter_contours = []
    #     filter_hierarchy = []
    #     colors = [(0, 255, 0), (0,0,255),(255, 0, 0)]
  
    #     for i, cnt in enumerate(contours):
    #         if hierarchy[0][i][3] == -1:
    #             continue
            
    #         area = cv2.contourArea(cnt)
    #         ratio = area / ellipse_area

    #         if ratio > 0.15:
    #             continue

            
    #         filter_contours.append(cnt)

    #     filter_contours.sort(key=lambda cnt: cv2.contourArea(cnt), reverse=True)

    #     if len(filter_contours) == 0:
   
    #         return None

    #     self.screen_reflection = cv2.minAreaRect(filter_contours[0])
    #     screen_reflection = cv2.boxPoints(self.screen_reflection)
    #     screen_box = np.int0(screen_reflection)

    #     cv2.drawContours(self.iris_frame, [screen_box], 0, (0, 255, 0), 1)

    #     center_rect = self.screen_reflection[0]


    #     draw_utils.draw_cross_lines(self.iris_frame, center_rect, (0, 255, 0))
    #     draw_utils.draw_cross_lines(self.iris_frame, iris_ellipse_center, (0, 0, 255))

    #     slope = geometry_utils.calculate_slope(iris_ellipse_center, center_rect)
    #     equation = geometry_utils.get_linear_equation(iris_ellipse_center, center_rect)

    #     domain = [0, iris_ellipse_center[0]]

    #     diff_x = iris_ellipse_center[0] - center_rect[0]
    #     diff_y = iris_ellipse_center[1] - center_rect[1]

    #     # draw_utils.draw_equation_line(self.iris_frame, domain, equation, (0, 255, 0))
    #     # draw_utils.draw_line(self.iris_frame, iris_ellipse_center, center_rect, (0, 255, 0))

    #     iris_roi = self.iris_ellipse.face_frame.face_mesh.get_iris_roi(self.iris_ellipse.eye_id)
    #     frame_screen_reflection = center_rect[0] + iris_roi[0][0], center_rect[1] + iris_roi[0][1]
    #     frame_iris_center = iris_ellipse_center[0] + iris_roi[0][0], iris_ellipse_center[1] + iris_roi[0][1]

    #     slope = geometry_utils.calculate_slope(iris_ellipse_center, center_rect)
    #     equation = geometry_utils.get_linear_equation(frame_iris_center, frame_screen_reflection)

    #     start_range_x = 0
    #     end_range_x = self.iris_ellipse.face_frame.frame.shape[1]

    #     delta = 1000

    #     if diff_x > 0:
    #         start_range_x = frame_screen_reflection[0] 
    #         end_range_x = frame.shape[1]
        
    #     elif diff_x < 0:
    #         start_range_x = 0
    #         end_range_x = frame_screen_reflection[0]

    #     domain = [start_range_x, end_range_x]

    #     draw_utils.draw_equation_line(frame, domain, equation, (0, 255, 0))
    #     title = 'eyeframe :' + str(self.iris_ellipse.eye_id)

    #     cv2.imshow("frame", frame)

     



    
