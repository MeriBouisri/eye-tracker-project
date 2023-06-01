from face_mesh import FaceMesh

from utility.geometry_utils import *
from utility.draw_utils import draw_line


class LandmarkVector:
    """
    The LandmarkVector class can be used to track the movement of the head by using two landmarks as reference points.
    The vector joining the two landmarks is used to calculate the angle and magnitude of the head movement.

    Attributes
    ----------
    face_mesh : FaceMesh
        The FaceMesh instance used to retrieve the landmark coordinates.
    """

    def __init__(self, face_mesh):
        """
        Parameters
        ----------
        face_mesh : FaceMesh
            The FaceMesh instance used to detect the landmarks.
        """
        self.face_mesh = face_mesh

    def set_landmark_keypoints(self, keypoint1, keypoint2):
        """
        Set the two keypoints that will be tracked by the LandmarkVector instance.
        The landmark vector will be a segment joining the landmark coordinates of the two keypoints.
        """
        self.vector_keypoints = keypoint1, keypoint2

    def get_landmark_coordinates(self, scaled_to_frame=True):
        """

        Parameters
        ----------
        scaled_to_frame : bool, optional, default = True
            If True, the returned landmark coordinates are scaled to the frame's dimensions. If False, the normalized coordinates are returned.
            Set to True by default.

        Returns
        ----------
        landmark_coordinates : tuple[int, int]
            The (x, y) coordinates of the two landmarks, scaled to the frame's dimensions.
        """
        if scaled_to_frame:
            return self.face_mesh.get_scaled_landmarks(self.vector_keypoints)
        
        return self.face_mesh.get_normalized_landmarks(self.vector_keypoints)
    
    def get_angle_degrees(self):
        """
        Get the angle (in degrees) of the vector joining the two landmarks.

        Returns
        ----------
        angle : float
            The angle in degrees of the vector joining the two landmarks.
        """
        return calculate_angle_degrees(*self.get_landmark_coordinates())
    
    def get_magnitude(self):
        """
        Get the magnitude of the vector joining the two landmarks.

        Returns
        ----------
        magnitude : float
            The magnitude of the vector joining the two landmarks.
        """
        return calculate_magnitude(self.get_landmark_coordinates())
    
    def get_geometric_vector(self):
        """
        Get the magnitude and the angle of the vector joining the two landmarks.

        Returns
        ----------
        geometric_vector: tuple[float, float]
            A tuple containing the magnitude and the angle of the vector joining the two landmarks.
        """
        return as_geometric_vector(*self.get_landmark_coordinates())
    
    def get_center_coordinates(self, scale_to_frame=True):
        """
        Get the coordinates of the center point between the two landmarks.

        Returns
        ----------
        center_coordinates : tuple[int, int]
            The (x, y) coordinates of the center point between the two landmarks.
        """
        return self.face_mesh.mean_landmark_coordinates(self.vector_keypoints, scale_to_frame=scale_to_frame)

    def draw_landmark_vector(self, frame, color=(0, 0, 255), thickness=1):
        """
        Draw a line representing the vector joining the two landmarks on the given frame.
        """
        draw_line(frame, *self.get_landmark_coordinates(), color=color, thickness=thickness)
    