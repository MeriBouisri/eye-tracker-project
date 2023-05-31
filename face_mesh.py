import mediapipe as mp
import numpy as np
import cv2

from utility.draw_utils import *

class FaceNotFound(Exception):
    pass

class FaceMesh():
    """
    A class that encapsulates the face mesh functionality provided by the mediapipe library.

    Attributes
    ----------
    frame : ndarray
        The frame to be processed by the face mesh.

    landmark_points : list
        A list of the face mesh landmarks detected in the current frame

    Constants 
    ----------
    KEYPOINT_COUNT : int = 478
        The number of keypoints in the face mesh provided by the mediapipe library. 
    """
    FACE_MESH = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

    KEYPOINT_COUNT = 478

    def __init__(self):
        pass

    def apply_face_mesh(self, frame):
        """
        Apply the face mesh to the current frame.

        Parameters
        ----------
        frame : ndarray
            The frame to be processed by the face mesh. All subsequent method calls will be applied to the face mesh detected in this frame,
            until the next call to this method. The frame's dimensions are 

        Raises
        ----------
        FaceNotFound : 
            Raised when the face mesh fails to detect any landmarks in the frame. 
        """

        # Store as (width, height) for consistency with (x, y) coordinates
        self.frame_dimensions = np.array([frame.shape[1], frame.shape[0]])

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output = self.FACE_MESH.process(rgb_frame)
        self.landmark_points = output.multi_face_landmarks

        if not self.landmark_points:
            raise FaceNotFound('Face not found in frame')


    def get_normalized_landmarks(self, keypoints=range(KEYPOINT_COUNT), keep_dimensions=False):
        """
        Returns the normalized value of (x, y) coordinates of the face mesh landmark at the given keypoints.
        If no keypoints are given, returns (x, y) coordinates of all the landmarks in the face mesh.

        Parameters
        ----------
        keypoints: array_like, optional = range(KEYPOINT_COUNT)
            The keypoints at which to retrieve the landmarks. The default is to retrieve all the landmarks in the face mesh.
        
        keep_dimensions: bool, optional = False
            If True, the tuple of coordinates will be indexed according to the original position of the corresponding keypoint from the input array,
            such as output_array.ndim() == input_array.ndim() + 1.
            If False, the tuple of coordinates will be organized in a 2d list. Set to False by default.

        Returns
        ----------
        coordinates: ndarray
            A list of the (x, y) coordinates of the face mesh landmarks at the given keypoints.

        Examples
        ----------
        let the keypoint [0] be the landmark at normalized coordinates (x = 0.5, y = 0.5)

        Single keypoint
        >>> mesh.get_normalized_landmarks(Landmark.PUPIL_LANDMARK)
        array([[0.5, 0.5]])

        Multiple keypoints
        >>> keypoints = [0, 0, 0, 0]
        >>> mesh.get_normalized_landmarks(keypoints)
        array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])

        Multiple keypoints, keep_dimensions=True
        >>> keypoints = [[0, 0], [0, 0]]
        >>> mesh.get_normalized_landmarks(keypoints, keep_dimensions=True)
        array([[[0.5, 0.5], [0.5, 0.5]],
                [[0.5, 0.5], [0.5, 0.5]]])
        """

        landmarks = self.landmark_points[0].landmark

        keypoints = np.array(keypoints)
        original_shape = keypoints.shape

        # Flatten multidimensional array, but keep original shape in memory 
        if keypoints.ndim > 1:
            keypoints = keypoints.flatten()
        
        # Retrieve landmarks at the given keypoints
        normalized_landmarks = np.take(landmarks, keypoints)
        coordinates = np.array([[landmark.x, landmark.y] 
                                for landmark in normalized_landmarks])
        
        # Return the array to its original shape if necessary 
        if keep_dimensions:
            return coordinates.reshape(original_shape + (2,))
        
        return coordinates
    

    def get_scaled_landmarks(self, keypoints=range(KEYPOINT_COUNT), keep_dimensions=False):
        """
        Return a list of the (x, y) coordinates of the face mesh landmarks at the given keypoints.
        The coordinates are scaled to this instance's frame dimensions.
        If no keypoints are given, the default behavior is to map and return all the landmarks in the face mesh.

        Parameters
        ----------
        keypoints : array_like, optional, default = range(KEYPOINT_COUNT)
            The keypoints at which to map the landmarks. The default is to map all the landmarks in the face mesh.
        
        keep_dimensions: bool, optional, default = False
            If True, the tuple of coordinates will be indexed according to the original position of the corresponding keypoint from the input array,
            such as output_array.ndim() == input_array.ndim() + 1.
            If False, the tuple of coordinates will be organized in a 2d list. Set to False by default.

        Returns
        ----------
        coordinates : ndarray
            A list of the (x, y) coordinates of the face mesh landmarks at the given keypoints, scaled to the frame's dimensions.
        """
        normalized_landmarks = self.get_normalized_landmarks(keypoints, keep_dimensions)
        return normalized_landmarks * self.frame_dimensions
    

    def mean_landmark_coordinates(self, keypoints, axis=0, scale_to_frame=False):
        """
        Compute the average point of the landmark coordinates at the given keypoints.

        Parameters
        ----------
        keypoints : array_like
            The face_mesh keypoints at which to calculate the mean. 

        axis : int, optional, default = 0
            The axis along which the mean is calculated. The default is to compute the mean of the flattened array (axis = 0)

        scale_to_frame : bool, optional, default = False
            If True, the mean coordinates are scaled to the frame's dimensions. If False, the mean of the normalized coordinates is returned.
            Set to False by default.

        Returns
        ----------
        mean : ndarray
            The mean of the face mesh landmarks at the given keypoints, along the specified axis. 
        """

        landmarks = self.get_normalized_landmarks(keypoints, keep_dimensions=True)
        mean_landmark = np.mean(landmarks, axis=axis)

        if scale_to_frame:
            mean_landmark * self.frame_dimensions
        
        return mean_landmark

# ------------------------------
# EXAMPLE USAGE
# ------------------------------

if __name__ == '__main__':
    cam = cv2.VideoCapture(0)
    mesh = FaceMesh()

    PUPIL_LANDMARKS = [468, 473]
    OUTER_EYE_CORNER_LANDMARKS = [33, 263]

    IRIS_LANDMARKS = [[474, 475,476, 477], 
                      [469, 470, 471, 472]]
    
    NOSE_LANDMARKS = [197, 8]

    while True:
        _, frame = cam.read()
        frame = cv2.flip(frame, 1)
 

        try:
            mesh.apply_face_mesh(frame)

            face_landmarks = mesh.get_scaled_landmarks()
            draw_min_enclosing_rectangle(frame, face_landmarks)

            mean = mesh.mean_landmark_coordinates(IRIS_LANDMARKS, axis=1, scale_to_frame=True)
            norm_mean = mesh.mean_landmark_coordinates(IRIS_LANDMARKS, axis=1)

            m = mesh.get_scaled_landmarks(IRIS_LANDMARKS)

            # a = mesh.mean_landmark_coordinates(IRIS_LANDMARKS, axis=1)

            # draw_cross(frame, mean, length=10)
            draw_all_crosses(frame, mean, length=10)

        

        except FaceNotFound:
            pass

        cv2.imshow('Frame', frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()
                
    


    
