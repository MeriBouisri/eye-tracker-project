import cv2

from utility.draw_utils import *
from utility.geometry_utils import *

from face_mesh import FaceMesh
from landmark_vector import LandmarkVector
from stable_face_mesh import StableFaceMesh
from landmark_constants import *

class FaceFrame:
    """
    A FaceFrame object is a frame that can be processed by the FaceMesh class, and then transformed accordingly.

    Attributes
    ----------
    frame : ndarray
        The frame to be processed by the face mesh.

    face_mesh : FaceMesh
        The FaceMesh instance used to detect the landmarks.
    """


    def __init__(self, face_mesh: FaceMesh = FaceMesh()):
        self.face_mesh = face_mesh
        self.stable_mesh = StableFaceMesh()

        self.frame: np.ndarray = None
        self.frame_dimensions = None

        self.horizontal_vector = LandmarkVector(self.face_mesh)
        self.horizontal_vector.set_landmark_keypoints(*EYE_OUTER_CORNER_KEYPOINTS)

        self.vertical_vector = LandmarkVector(self.face_mesh)
        self.vertical_vector.set_landmark_keypoints(*FACE_CENTER_KEYPOINTS)

    def update_frame(self, frame):
        """
        Update the frame to be processed by the FaceMesh instance.
        Call this method before calling any other methods that rely on the frame.

        Parameters
        ----------
        frame : ndarray
            The image to be processed by the face mesh. The image will also be the target for all subsequent methods

        Raises
        ----------
        FaceNotFound :
            Raised when the face mesh fails to detect any landmarks in the frame.
        """
        self.frame = frame
        self.frame_dimensions = np.array([self.frame.shape[0], self.frame.shape[1]])
        self.face_mesh.apply_face_mesh(frame)

    # ======================
    # STABILISATION METHODS
    # ======================

    def get_horizontal_vector(self):
        """
        Returns
        ----------
        horizontal_vector : tuple[float, float]
            The (magnitude, angle) of the horizontal vector.
        """
        return self.horizontal_vector.get_geometric_vector()
    
    def get_vertical_vector(self):
        """
        Returns
        ----------
        vertical_vector : tuple[float, float]
            The (magnitude, angle) of the vertical vector.
        """
        return self.vertical_vector.get_geometric_vector()
    
    def get_scale_factor(self):
        frame_surface_area = self.frame.shape[0] * self.frame.shape[1]
        face_surface_area = self.get_min_area_rectangle()

        return face_surface_area / frame_surface_area
    
    def get_rotation_angle(self):
        return self.horizontal_vector.get_angle_degrees()
    
    def get_stable_mesh(self):
        """
        Apply necessary transformations to this instance's face mesh.

        Returns
        ----------
        stable_mesh : StableFaceMesh
            A stable_face_mesh instance
        """
        center_vector = self.horizontal_vector.get_center_coordinates()

        # Rotate the face mesh around the center of the horizontal vector
        landmarks = self.face_mesh.get_scaled_landmarks() - center_vector
        rotation = self.get_rotation_angle()
        rotated_landmarks = rotate_points(landmarks, math.radians(rotation)) 

        translated_landmarks = self.translate_to_center(rotated_landmarks) + center_vector

        # Map coordinates to origin (0, 0) and scale to frame dimensions
        self.stable_mesh.update_mesh(translated_landmarks)

    def translate_to_center(self, landmarks):
        center_vector_x, center_vector_y = self.horizontal_vector.get_center_coordinates()
        center_frame_x = self.frame.shape[1] // 2
        center_frame_y = self.frame.shape[0] // 2

        dx = center_frame_x - center_vector_x
        dy = center_frame_y - center_vector_y

        for landmark in landmarks:
            landmark[0] += dx
            landmark[1] += dy

        return landmarks


    # ======================
    # FACE AREA METHODS
    # ======================

    def get_face_rectangle(self):
        """
        Returns
        ----------
        face_rectangle : ndarray
            The vertices of the minimal rectangle that encloses the face mesh. It may be rotated.
        """
        face_area_convex_hull = self.get_convex_hull(scale_to_frame=False)
        return get_rectangular_vertices(face_area_convex_hull)
    
    def get_eye_zone_rectangle(self):
        """
        Returns
        ----------
        eye_zone_rectangle : ndarray
            The vertices of the minimal rectangle that encloses the eye zone. It may be rotated.
        """
        # We will use several keypoints in order to create a convex hull that encloses the eye zone as much as we want
        eye_area_keypoints = EYE_AREA_KEYPOINTS
        eye_area_convex_hull = get_convex_hull(self.face_mesh.get_landmarks(eye_area_keypoints))

        return get_rectangular_vertices(eye_area_convex_hull)

    # ======================
    # LANDMARK METHODS
    # ======================
    
    def get_pupil_center_landmarks(self, scale_to_frame=True):
        """
        Get the (x, y) coordinates of the pupil center landmarks.

        Constants
        ----------
        PUPIL_CENTER_KEYPOINTS : list[int] = [468, 473]

        Parameters
        ----------
        scale_to_frame : bool, optional, default = True
            If True, the returned landmark coordinates are scaled to the frame's dimensions. If False, the normalized coordinates are returned.

        Returns
        ----------
        pupil_center_landmarks : ndarray
            The (x, y) coordinates of the pupil center landmarks.
        """
        return self.face_mesh.get_landmarks(PUPIL_CENTER_KEYPOINTS, scale_to_frame=scale_to_frame)
    
    def get_eye_outer_corner_landmarks(self, scale_to_frame=True):
        """
        Get the (x, y) coordinates of the eye outer corner landmarks.

        Constants
        ----------
        EYE_OUTER_CORNER_KEYPOINTS : list[int] = [33, 263]

        Parameters
        ----------
        scale_to_frame : bool, optional, default = True
            If True, the returned landmark coordinates are scaled to the frame's dimensions. If False, the normalized coordinates are returned.

        Returns
        ----------
        eye_outer_corner_landmarks : ndarray
            The (x, y) coordinates of the eye outer corner landmarks.
        """
        return self.face_mesh.get_landmarks(EYE_OUTER_CORNER_KEYPOINTS, scale_to_frame=scale_to_frame)
    
    def get_face_center_landmarks(self, scale_to_frame=True):
        """
        Get the (x, y) coordinates of the face center landmarks.

        Constants
        ----------
        FACE_CENTER_KEYPOINTS : list[int] = [197, 8]

        Parameters
        ----------
        scale_to_frame : bool, optional, default = True
            If True, the returned landmark coordinates are scaled to the frame's dimensions. If False, the normalized coordinates are returned.

        Returns
        ----------
        face_center_landmarks : ndarray
            The (x, y) coordinates of the face center landmarks.
        """
        return self.face_mesh.get_landmarks(FACE_CENTER_KEYPOINTS, scale_to_frame=scale_to_frame)
    
    def get_iris_landmarks(self, scale_to_frame=True):
        left_iris = self.face_mesh.get_landmarks(IRIS_KEYPOINTS[0], scale_to_frame=scale_to_frame)
        right_iris = self.face_mesh.get_landmarks(IRIS_KEYPOINTS[1], scale_to_frame=scale_to_frame)
        return left_iris, right_iris
    
    # ======================
    # GEOMETRIC METHODS
    # ======================
    
    def get_convex_hull(self, scale_to_frame=True):
        """
        Get the vertices of the convex non-self-intersecting polygon that encloses the face mesh.

        Parameters
        ----------
        scale_to_frame : bool, optional, default = True
            If True, the returned landmark coordinates are scaled to the frame's dimensions. If False, the normalized coordinates are returned.
        
        Returns
        ----------
        convex_hull : ndarray
            The vertices of the convex non-self-intersecting polygon that encloses the face mesh.
        """
        return get_convex_hull(self.face_mesh.get_landmarks(scale_to_frame=scale_to_frame))
    
    def get_min_area_rectangle(self):
        face_landmarks = self.get_convex_hull(scale_to_frame=False)
        normalized_rotated_landmarks = rotate_points(face_landmarks, self.horizontal_vector.get_angle_degrees())
        scaled_rotated_landmarks = normalized_rotated_landmarks * self.face_mesh.frame_dimensions
        min_area_rectangle = get_min_enclosing_rectangle(scaled_rotated_landmarks)

        return get_rectangular_area(min_area_rectangle)
    
    def rotate_frame(self, angle, rotation_axis=None, scale=1):
        return self.rotate_frame(self.frame, angle, rotation_axis, scale)

    def crop_frame(self, upper_left_corner, lower_right_corner):
        return self.crop_frame(self.frame, upper_left_corner, lower_right_corner)

    
    @staticmethod
    def rotate_frame(frame, angle, rotation_axis=None, scale=1):
        """
        Rotate a frame by a given angle around a given axis.

        Parameters
        ----------
        frame : np.ndarray
            The frame to rotate.

        angle : float
            The angle in degrees by which to rotate the frame.

        rotation_axis : tuple[int, int], optional, default = None
            The (x, y) coordinates of the point around which to rotate the frame. 
            If None, the frame is rotated around its center.

        scale : int, optional, default = 1
            The scale factor to apply to the frame after rotation. 
            The default is to not scale the frame.

        Returns
        ----------
        rotated_frame : ndarray
            A copy of the input frame, rotated by the given angle around the point specified by the rotation_axis.
        """

        if rotation_axis is None:
            rotation_axis = np.array(frame.shape[:2]) / 2

        rotation_matrix = cv2.getRotationMatrix2D(rotation_axis, angle, scale)
        return cv2.warpAffine(frame, rotation_matrix, frame.shape[:2][::-1])

    @staticmethod
    def crop_frame(frame, upper_left_corner, lower_right_corner):
        """
        Crop the frame to the area enclosed inside a rectangle.
        A rectangle is defined by the upper left and lower right corners.

        Parameters
        ----------
        frame : np.ndarray
            The frame to crop.
        
        upper_left_corner : tuple[int, int]
            The (x, y) coordinates of the upper left corner of the rectangle to crop to.
            The upper left corner is made up of the minimum x and y values of the rectangle.

        lower_right_corner : tuple[int, int]
            The (x, y) coordinates of the lower right corner of the rectangle to crop to.
            The lower right corner is made up of the maximum x and y values of the rectangle.

        Returns
        ----------
        cropped_frame : ndarray
            A copy of the input frame, cropped to the area enclosed inside the rectangle defined by the upper left and lower right corners.

        Notes
        ----------
        If the rectangle defined by the upper left and lower right corners extends beyond the frame's dimensions, 
        the rectangle is clipped to the frame's dimensions.

        See Also
        ----------
        get_min_enclosing_rectangle : Get the minimum enclosing rectangle of a set of points.
        draw_min_enclosing_rectangle : Draw the minimum enclosing rectangle of a set of points on a frame.
        """

        min_height, min_width = 0, 0
        max_height, max_width, _ = frame.shape
        
        min_column, min_row = np.clip(upper_left_corner, min_height, max_height)
        max_column, max_row = np.clip(lower_right_corner, min_width, max_width)

        return frame[min_row:max_row, min_column:max_column]





    
        