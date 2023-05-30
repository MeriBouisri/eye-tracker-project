from utility.draw_utils import *
from utility.geometry_utils import *
from face_mesh import FaceMesh

import cv2

class MeshFrame:
    """
    A MeshFrame is a frame that can be processed by the FaceMesh class, and then transformed accordingly.

    Attributes
    ----------
    frame : ndarray
        The frame to be processed by the face mesh.

    mesh : FaceMesh
        The FaceMesh instance 

    rotation_angle : float
        The angle of the last rotation transformation applied to the frame.
    
    rotation_axis : tuple[int, int]
        The (x, y) coordinates of the point around which the frame is rotated.
    
    scale_factor : int
        The scale factor applied to the frame after each transformation.

    reference_vector: tuple[tuple[int, int], tuple[int, int]]
        The two points that define the reference vector used to determine the transformations applied to the frame.
    """

    PUPIL_LANDMARK = [468, 473]

    def __init__(self):
        self.mesh = FaceMesh()

        self.frame: np.ndarray = None
        self.frame_dimensions = None

        self.reference_vector = 0, 0
        self.rotation_angle = 0
        self.scale_factor = 1
        self.rotation_axis = None


    def update_frame(self, frame):
        """
        Update the frame to be processed by the FaceMesh instance.

        Parameters
        ----------
        frame : ndarray
            The image to be processed by the face mesh. The image will also be the target for all subsequent methods
        """
        self.frame = frame
        self.frame_dimensions = np.array([self.frame.shape[0], self.frame.shape[1]])

        self.mesh.update_frame(self.frame)
        self.mesh.apply_face_mesh()

        self.reference_vector = self.mesh.get_normalized_landmarks(self.PUPIL_LANDMARK)


    def set_reference_vector(self, keypoint1, keypoint2, set_rotation_axis=False):
        """
        Set the reference vector to be used to determine the transformations applied to the frame.

        Parameters
        ----------
        keypoint1 : int
            The index of the first keypoint of the face mesh representing the initial point of the reference vector.

        keypoint2 : int
            The index of the second keypoint of the face mesh representing the terminal point of the reference vector.

        set_rotation_axis : bool, optional, default = False
            If True, the rotation axis is set to the midpoint of the reference vector. Set to False by default.


        Notes
        ----------
        The reference_vector attribute contains the two points necessary for determining the angle and magnitude of the 
        segment that connects the two points. The idea is to keep track of the head movements through the variation of the magnitude and angle
        of the vector.
        The landmark coordinates are not scaled to the frame's dimensions, as it is not necessary.
        """
        self.reference_vector = keypoint1, keypoint2

        if set_rotation_axis:
            self.rotation_axis = self.mesh.mean_landmark_coordinates(self.reference_vector)

    def set_rotation_axis(self, rotation_axis):
        """
        Set the rotation axis to be used to determine the transformations applied to the frame.

        Parameters
        ----------
        rotation_axis : tuple[int, int]
            The (x, y) coordinates of the point around which the frame is rotated.
        """
        self.rotation_axis = rotation_axis

    def locate_face_area(self):
        landmarks = self.mesh.map_landmarks_to_frame()
        return get_min_enclosing_rectangle(landmarks)
    
    def locate_eye_area(self):
        landmarks = self.mesh.map_landmarks_to_frame()
        return get_min_enclosing_rectangle(landmarks)
    
    def get_face_frame(self):
        return MeshFrame.crop_frame(self.frame, *self.locate_face_area() )
    
    def get_eye_frame(self):
        return MeshFrame.crop_frame(self.frame, *self.locate_eye_area())
    
    def rotate_mesh_frame(self):
        vector_points = self.mesh.get_normalized_landmarks(self.reference_vector)
        self.rotation_angle = calculate_angle_degrees(*vector_points)
        
        return MeshFrame.rotate_frame(self.frame, self.rotation_angle, self.rotation_axis)
    
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
    
if __name__ == '__main__':
    cam = cv2.VideoCapture(0)
    mesh = MeshFrame()

    while True:
        _, frame = cam.read()
        frame = cv2.flip(frame, 1)

        mesh.update_frame(frame)

        

        mesh.set_reference_vector(*MeshFrame.PUPIL_LANDMARK, set_rotation_axis=True)

        mapped_points = mesh.mesh.map_landmarks_to_frame(mesh.reference_vector)
        draw_line(frame, *mapped_points)
        draw_cross(frame, mesh.rotation_axis, color = (0, 255, 0), length = 10)


        rotated_frame = mesh.rotate_mesh_frame()



        cv2.imshow('Face', rotated_frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()
    





    
        