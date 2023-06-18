import numpy as np

from eye_tracking.utils import image_utils
from eye_tracking.face_mesh import RawFaceMesh

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
    def __init__(self):
        self.face_mesh = RawFaceMesh()

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

    def get_eye_frame(self, eye_id):
        """
        Returns the frame of the eye specified by the eye_id parameter.

        Parameters
        ----------
        eye_id : str | int | EyeID
            Specifies which eye the function should be applied to (left or right)
            The eye_id can be specified as a string ('left' or 'right'), an integer (0 or 1), or an EyeID object (EyeID.left or EyeID.right).

        Returns
        ----------
        eye_frame : ndarray
            The frame of the eye specified by the eye_id parameter. 

        Notes
        ----------
        The returned frame is not a copy of the original frame, but rather a view of it. 
        Any changes made to cropped area of the frame will also be applied to that area of the original frame.

        See Also 
        ----------
        EyeDict :
            For more information on the eye_id parameter.

        image_utils.crop_frame :
            For more information on the crop_frame function and how it affects the original frame.
        """
        eye_roi = self.face_mesh.get_eye_roi(eye_id)
        return self.crop_frame(*eye_roi)
 
    
    def get_iris_frame(self, eye_id):
        """
        Returns the frame of the iris specified by the eye_id parameter.

        Parameters
        ----------
        eye_id : str | int | EyeID
            Specifies which eye the function should be applied to (left or right)
            The eye_id can be specified as a string ('left' or 'right'), an integer (0 or 1), or an EyeID object (EyeID.left or EyeID.right).

        Returns
        ----------
        eye_frame : ndarray
            The frame containing the iris region specified by the eye_id parameter.

        Notes
        ----------
        The returned frame is not a copy of the original frame, but rather a view of it. 
        Any changes made to cropped area of the frame will also be applied to that area of the original frame.

        See Also 
        ----------
        EyeDict :
            For more information on the eye_id parameter.

        image_utils.crop_frame :
            For more information on the crop_frame function and how it affects the original frame.
        """
        iris_roi = self.face_mesh.get_iris_roi(eye_id)
        return self.crop_frame(*iris_roi)

    def rotate_frame(self, angle, rotation_axis=None, scale=1):
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

        See Also
        ----------
        image_utils.rotate_frame :
            The static version of this method. See for more information about the implementation.
        """
        return image_utils.rotate_frame(self.frame, angle, rotation_axis, scale)

    def crop_frame(self, upper_left_corner, lower_right_corner):
        """
        Crop the frame to the area enclosed within the upper left and lower right corners of a rectangle.

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
        cropped_frame : ndarray | None
            The section of the input frame cropped to the area enclosed inside the rectangle defined by the upper left and lower right corners.
            The output frame is not a copy of the input frame, such as any changes made to the output frame will also affect the input frame.
            Returns None if the frame's dimensions are invalid or the frame is empty.

        Notes
        ----------
        If the rectangle defined by the upper left and lower right corners extends beyond the frame's dimensions, 
        the rectangle is clipped to the frame's dimensions.

        See Also
        ----------
        image_utils.crop_frame :
            The static version of this method. See for more information about the implementation.
        """
        cropped_frame = image_utils.crop_frame(self.frame, upper_left_corner, lower_right_corner)

        if cropped_frame.shape[0] < 0 or cropped_frame.shape[1] < 0:
            return None
        
        elif cropped_frame.size == 0:
            return None
        
        return cropped_frame




    




    
        