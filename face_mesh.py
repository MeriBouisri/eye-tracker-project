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

    face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

    KEYPOINT_COUNT = 478

    def __init__(self):
        pass
        
    

    def update_frame(self, frame):
        """
        Update the frame to be processed.

        Parameters
        ----------
        frame : ndarray
            The image to be processed by the face mesh. The image will also be the target for all subsequent methods 
            of the FaceMesh instance until the next call to update_frame().
        """
        self.frame = frame
        self.frame_h, self.frame_w, _ = self.frame.shape


    def apply_face_mesh(self):
        """
        Apply the face mesh to the current frame.

        Exceptions
        ----------
        FaceNotFound : Raised when the face mesh fails to detect any landmarks in the frame. 
        Surround with try-except block if necessary.
        """
        rgb_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        output = self.face_mesh.process(rgb_frame)
        self.landmark_points = output.multi_face_landmarks

        if not self.landmark_points:
            raise FaceNotFound('Face not found in frame')


    def get_normalized_landmarks(self, keypoints=range(KEYPOINT_COUNT), keep_dimensions=False):
        """
        Returns the normalized value of (x, y) coordinates of the face mesh landmark at the given keypoints.
        If no keypoints are given, returns (x, y) coordinates of all the landmarks in the face mesh.

        Parameters
        ----------
        keypoints: array_like, optional
            The keypoints at which to retrieve the landmarks. The default is to retrieve all the landmarks in the face mesh.
        
        keep_dimensions: bool, optional
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
        original_shape = 0

        # Flatten multidimensional array, but keep original shape in memory 
        if keypoints.ndim > 1:
            original_shape = keypoints.shape
            keypoints = keypoints.flatten()
        
        # Retrieve landmarks at the given keypoints
        normalized_landmarks = np.take(landmarks, keypoints)
        coordinates = np.array([[landmark.x, landmark.y] for landmark in normalized_landmarks])
        
        # Return the array to its original shape if necessary 
        if keep_dimensions and keypoints.ndim > 1:
            return coordinates.reshape(original_shape + (2,))
        
        return coordinates
    

    def map_landmarks_to_frame(self, keypoints=range(KEYPOINT_COUNT), keep_dimensions=False):
        """
        Return a list of the (x, y) coordinates of the face mesh landmarks at the given keypoints.
        The coordinates of the normalized landmarks are mapped to the frame's dimensions.
        If no keypoints are given, the default behavior is to map and return all the landmarks in the face mesh.

        Parameters
        ----------
        keypoints : array_like, optional
            The keypoints at which to map the landmarks. The default is to map all the landmarks in the face mesh.
        
        keep_dimensions: bool, optional
            If True, the tuple of coordinates will be indexed according to the original position of the corresponding keypoint from the input array,
            such as output_array.ndim() == input_array.ndim() + 1.
            If False, the tuple of coordinates will be organized in a 2d list. Set to False by default.

        Returns
        ----------
        coordinates : ndarray
            A list of the (x, y) coordinates of the face mesh landmarks at the given keypoints, scaled to the frame's dimensions.
        """
        normalized_landmarks = self.get_normalized_landmarks(keypoints, keep_dimensions)
        frame_dimensions = np.array([self.frame_w, self.frame_h])

        return normalized_landmarks * frame_dimensions
    

    def mean_landmark_coordinates(self, keypoints, axis=0):
        """
        Compute the average point of the landmark coordinates at the given keypoints. The coordinates are mapped to the frame's dimensions before the mean is calculated.

        Parameters
        ----------
        keypoints : array_like
            The face_mesh keypoints at which to calculate the mean. 

        axis : int, optional
            The axis along which the mean is calculated. The default is to compute the mean of the flattened array (axis = 0)

        Returns
        ----------
        mean : ndarray
            The mean of the face mesh landmarks at the given keypoints, along the specified axis. 
        """
        mapped_landmarks = self.map_landmarks_to_frame(keypoints, keep_dimensions=True)
        return np.mean(mapped_landmarks, axis=axis)

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
        mesh.update_frame(frame)    

        try:
            mesh.apply_face_mesh()

            face_landmarks = mesh.map_landmarks_to_frame()
            draw_min_enclosing_rectangle(frame, face_landmarks)

            pupil_landmarks = mesh.map_landmarks_to_frame(PUPIL_LANDMARKS)
            draw_all_crosses(frame, pupil_landmarks, color=(0, 255, 0), length=5)

            eye_area_landmarks = mesh.map_landmarks_to_frame([IRIS_LANDMARKS[0] + IRIS_LANDMARKS[1] + OUTER_EYE_CORNER_LANDMARKS])
            nose_area_landmarks = mesh.map_landmarks_to_frame(NOSE_LANDMARKS + OUTER_EYE_CORNER_LANDMARKS)
            
            draw_convex_hull(frame, eye_area_landmarks)
            draw_convex_hull(frame, nose_area_landmarks, color=(0,255, 0))

        except FaceNotFound:
            pass

        cv2.imshow('Frame', frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()
                
    


    
