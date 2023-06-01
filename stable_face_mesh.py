import cv2
import numpy as np

from utility.draw_utils import *
from utility.geometry_utils import *

from landmark_constants import *

class StableFaceMesh():
    """
    A StableFaceMesh is a FaceMesh, but with geometric transformations applied to the landmarks in order to filter out unnecessary movement.

    Notes
    -----------
    Unlike the FaceMesh class, the StableFaceMesh class does not have a method to apply the face mesh to a frame. The stable face mesh is generated from an already-applied
    face mesh.
    Note also that StableFaceMesh landmarks are scaled to the frame by default, unlike FaceMesh landmarks which are normalized by default.
    Otherwise, feel free to use the StableFaceMesh class as you would the FaceMesh class.
    """

    def __init__(self):
        pass

    def update_mesh(self, mesh):
        self.stable_mesh = mesh

    def get_landmarks(self, keypoints=None, scale_to_frame=True):
        if scale_to_frame:
            return self.get_scaled_landmarks(keypoints)
        return self.get_normalized_landmarks(keypoints)
    
    def get_scaled_landmarks(self, keypoints=None):
        if keypoints is None:
            keypoints = range(KEYPOINT_COUNT)
        return np.take(self.stable_mesh, keypoints, axis=0)
    
    def get_normalized_landmarks(self, keypoints=None):
        # TODO: Implement this method
        return self.get_scaled_landmarks(keypoints)
    
    def mean_landmark_coordinates(self, keypoints=None, scale_to_frame=True, axis=0):
        return np.mean(self.get_landmarks(keypoints, scale_to_frame=scale_to_frame), axis=axis)
    



    

        