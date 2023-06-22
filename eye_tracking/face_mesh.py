from abc import ABC, abstractmethod
import numpy as np
import mediapipe as mp
import cv2

from eye_tracking.utils import geometry_utils
from eye_tracking.eye_keypoints import eye_dict

KEYPOINT_COUNT = 478


class FaceNotFound(Exception):
    pass

class FaceMesh(ABC):

    def get_landmarks(self, keypoints=None, scale_to_frame=True):
        """
        Parameters
        ----------
        keypoints : list, optional
            A list of the indices of the landmarks to be returned. If None, all the face landmarks will be returned.

        scale_to_frame : bool, optional = True
            If True, the coordinates of the landmarks will be scaled to the frame dimensions. If False, the normalized coordinates will be returned.

        Returns
        ----------
        The coordinates of the landmarks at the given keypoints.
        """
        if scale_to_frame:
            return self.get_scaled_landmarks(keypoints)
        
        return self.get_normalized_landmarks(keypoints)
    
    def mean_landmark_coordinates(self, keypoints, axis=0, scale_to_frame=False):
        """
        Parameters
        ----------
        keypoints : list
            A list of the indices of the landmarks to be used to calculate the mean coordinates.

        axis : int, optional = 0
            Axis or axes along which the means are computed. The default is to compute the mean of the flattened array

        scale_to_frame : bool, optional = False
            If True, the coordinates of the landmarks will be scaled to the frame dimensions. If False, the normalized coordinates will be returned.

        Returns
        ----------
        The mean coordinates of the landmarks along the specified axis.
        """
        landmark_coordinates = self.get_landmarks(keypoints, scale_to_frame=scale_to_frame)
        return np.mean(landmark_coordinates, axis=axis)

    def get_convex_hull(self, keypoints=None, scale_to_frame=True):
        """
        Parameters
        ----------
        keypoints : list, optional
            A list of the indices of the landmarks contained in the convex hull.
            If None, all the face landmarks will be used.

        scale_to_frame : bool, optional = True
            If True, the coordinates of the landmarks will be scaled to the frame dimensions. If False, the normalized coordinates will be returned.

        Returns
        ----------
        The smallest convex polygon that contains all the landmark coordinates.
        """
        landmark_coordinates = self.get_landmarks(keypoints, scale_to_frame=scale_to_frame)
        return geometry_utils.get_convex_hull(landmark_coordinates)
    
    # ========== Head rotation angle ==========

    def get_roll_rotation_angle(self):
        """
        Returns
        ----------
        The rotation of the head (in degrees) around the longitudinal axis.

        Notes
        ----------
        The "so-so" head gesture is done by rotating the head around the roll axis.
        """
        # TODO: Implement this method
        raise NotImplementedError("This method has not been implemented yet.")

    def get_pitch_rotation_angle(self):
        """
        Returns
        ----------
        The rotation angle of the head (in degrees) around the transverse axis.

        Notes
        ----------
        The "yes" head gesture is done by rotating the head around the pitch axis.
        """
        # TODO: Implement this method
        raise NotImplementedError("This method has not been implemented yet.")

    def get_yaw_rotation_angle(self):
        """
        Returns
        ----------
        The rotation of the head (in degrees) around the vertical axis.

        Notes
        ----------
        The "no" head gesture is done by rotating the head around the yaw axis.
        """
        # TODO: Implement this method
        raise NotImplementedError("This method has not been implemented yet.")
    
    # ========== Predefined landmark methods ==========

    def get_region_of_interest(self, keypoints=None):
        """
        Parameters
        ----------
        keypoints : list, optional
            A list of the indices of the landmarks within the region of interest.

        Returns
        ----------
        The upper left corner and lower right corner of the rectangle that contains the landmarks at the given keypoints.
        """
        return geometry_utils.get_min_enclosing_rectangle(self.get_convex_hull(keypoints))

    def get_face_roi(self):
        """
        Returns
        ----------
        The upper left corner and lower right corner of the rectangle that contains the entire face (all landmarks)
        """
        return self.get_region_of_interest()
    
    def get_eye_roi(self, eye_id):
        """
        Parameters
        ----------
        eye_id : str | int | EyeID
            Specifies which eye the function should be applied to (left or right)
            The eye_id can be specified as a string ('left' or 'right'), an integer (0 or 1), or an EyeID object (EyeID.left or EyeID.right).

        Returns
        ----------
        The upper left corner and lower right corner of the rectangle that contains the eye.
        """
        return self.get_region_of_interest(eye_dict[eye_id].eye)

    def get_iris_roi(self, eye_id):
        """
        Parameters
        ----------
        eye_id : str | int | EyeID
            Specifies which eye the function should be applied to (left or right)
            The eye_id can be specified as a string ('left' or 'right'), an integer (0 or 1), or an EyeID object (EyeID.left or EyeID.right).

        Returns
        ----------
        The upper left corner and lower right corner of the rectangle that contains the iris.

        See Also 
        ----------
        EyeDict : 
            For more information about the eye_id parameter.
        """
        return self.get_region_of_interest(eye_dict[eye_id].iris)

    # ========== Abstract Methods ==========

    @abstractmethod
    def apply_face_mesh(self, frame):
        pass

    @abstractmethod
    def get_normalized_landmarks(self, keypoints, keep_array_shape=False):
        """
        Parameters
        ----------
        keypoints : list
            A list of the indices of the landmarks to be returned.

        keep_array_shape : bool, optional = False
            If True, the returned array will maintain its original shape, such as the (x, y) coordinates will be stored at the same 
            index as their respective keypoint. If False, the returned array will be flattened.

        Returns
        ----------
        normalized_landmarks : numpy.ndarray
            The coordinates of the landmarks at the given keypoints, normalized to [0, 1] range
        """
        pass

    @abstractmethod
    def get_scaled_landmarks(self, keypoints, keep_array_shape=False):
        """
        Parameters
        ----------
        keypoints : list
            A list of the indices of the landmarks to be returned.

        keep_array_shape : bool, optional = False
            If True, the returned array will maintain its original shape, such as the (x, y) coordinates will be stored at the same
            index as their respective keypoint. If False, the returned array will be flattened.

        Returns
        ----------
        scaled_landmarks : numpy.ndarray
            The coordinates of the landmarks at the given keypoints, scaled to the frame dimensions.
        """
        pass

    

class RawFaceMesh(FaceMesh):
    """
    The RawFaceMesh class is used to detect the face landmarks in a given frame with the mediapipe face mesh model.

    Attributes
    ----------
    landmarks : numpy.ndarray
        The coordinates of the face landmarks in the last processed frame.

    See Also
    ----------
    ProcessedFaceMesh :
        A class used to store transformations of the face landmarks.
    """
    mediapipe_face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

    def __init__(self):
        self.landmarks = None

    def apply_face_mesh(self, frame):
        """
        Process the given frame in order to determine the face landmarks, and update the self.landmarks attribute.
        Parameters
        ----------
        frame : numpy.ndarray
            The frame to be processed.

        Raises
        ----------
        FaceNotFound
            If no face landmarks could be found in the frame
        """

        # Store as (width, height) for consistency with (x, y) coordinates
        self.frame_dimensions = np.array([frame.shape[1], frame.shape[0]])

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output = self.mediapipe_face_mesh.process(rgb_frame)
        landmark_points = output.multi_face_landmarks

        if not landmark_points:
            raise FaceNotFound('Face not found in frame')
        
        self.landmarks = landmark_points[0].landmark

    def get_normalized_landmarks(self, keypoints=None, keep_array_shape=False):
        if keypoints is None:
            keypoints = range(KEYPOINT_COUNT)
            
        elif not np.iterable(keypoints):
            landmarks = self.landmarks[keypoints]
            return landmarks.x, landmarks.y

        keypoints = np.array(keypoints)
        original_shape = keypoints.shape

        # Flatten multidimensional array, but keep original shape in memory 
        if keypoints.ndim > 1:
            keypoints = keypoints.flatten()
        
        # Retrieve landmarks at the given keypoints
        normalized_landmarks = np.take(self.landmarks, keypoints)

        coordinates = np.array([[landmark.x, landmark.y] 
                                for landmark in normalized_landmarks])
        
        # Return the array to its original shape if necessary 
        if keep_array_shape:
            return coordinates.reshape(original_shape + (2,))
        
        
        return coordinates
    
    def get_scaled_landmarks(self, keypoints=None, keep_dimensions=False):
        normalized_landmarks = self.get_normalized_landmarks(keypoints, keep_dimensions)
        return normalized_landmarks * self.frame_dimensions

    def mean_landmark_coordinates(self, keypoints, axis=0, scale_to_frame=False):
        landmarks = self.get_normalized_landmarks(keypoints, keep_dimensions=True)
        mean_landmark = np.mean(landmarks, axis=axis)

        if scale_to_frame:
            return mean_landmark * self.frame_dimensions
        
        return mean_landmark


class ProcessedFaceMesh(FaceMesh):
    """
    The ProcessedFaceMesh class is used to store the face landmarks after applying transformations to them.
    It extends the FaceMesh interface to allow compatibility with components that use the RawFaceMesh class.

    TODO: Discuss whether this class is actually necessary or not.
    """
    def __init__(self):
        pass

    def apply_face_mesh(self, frame):
        self.frame_dimensions = np.array([frame.shape[1], frame.shape[0]])	

    def update_mesh(self, landmarks):
        self.landmarks = landmarks

    def get_scaled_landmarks(self, keypoints=None, keep_dimensions=False):
        if keypoints is None:
            keypoints = range(KEYPOINT_COUNT)

        return np.take(self.landmarks, keypoints, axis=0)

    def get_normalized_landmarks(self, keypoints=None, keep_dimensions=False):
        return self.get_scaled_landmarks(keypoints, keep_dimensions) / self.frame_dimensions
    
