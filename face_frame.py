import cv2

from utility.draw_utils import *
from utility.geometry_utils import *

from face_mesh import FaceMesh

class FaceFrame:
    """
    A FaceFrame object is a frame that can be processed by the FaceMesh class, and then transformed accordingly.

    Attributes
    ----------
    frame : ndarray
        The frame to be processed by the face mesh.

    face_mesh : FaceMesh
        The FaceMesh instance used to detect the landmarks.

    Constants
    ----------
    PUPIL_CENTER_KEYPOINTS : list[int] = [468, 473]
    EYE_OUTER_CORNER_KEYPOINTS : list[int] = [33, 263]
    FACE_CENTER_KEYPOINTS : list[int] = [197, 8]
    """

    PUPIL_CENTER_KEYPOINTS = [468, 473]
    EYE_OUTER_CORNER_KEYPOINTS = [33, 263]
    FACE_CENTER_KEYPOINTS = [197, 8]

    IRIS_KEYPOINTS = [[474, 475, 476, 477], 
                      [469, 470, 471, 472]]

    def __init__(self, face_mesh: FaceMesh = FaceMesh()):
        self.face_mesh = face_mesh

        self.frame: np.ndarray = None
        self.frame_dimensions = None

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
        return self.face_mesh.get_landmarks(self.PUPIL_CENTER_KEYPOINTS, scale_to_frame=scale_to_frame)
    
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
        return self.face_mesh.get_landmarks(self.EYE_OUTER_CORNER_KEYPOINTS, scale_to_frame=scale_to_frame)
    
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
        return self.face_mesh.get_landmarks(self.FACE_CENTER_KEYPOINTS, scale_to_frame=scale_to_frame)
    
    def get_face_convex_hull(self, scale_to_frame=True):
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





    
        