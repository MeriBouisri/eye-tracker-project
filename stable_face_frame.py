from face_frame import FaceFrame
from landmark_vector import LandmarkVector
 
from utility.geometry_utils import *
from utility.draw_utils import * 
from landmark_constants import *

class StableFaceFrame(FaceFrame):
    """
    A StableFaceFrame is a FaceFrame, but with a stable face mesh applied to it, after the transformations from a parent FaceFrame.

    See Also
    ----------
    StableMesh:
        The stable mesh class can be used to access the stabilized landmarks.
    """

    def __init__(self, face_frame: FaceFrame = FaceFrame()):
        super().__init__()

        self.parent_frame = face_frame
        self.face_mesh = self.parent_frame.stable_mesh

        self.horizontal_vector = LandmarkVector(self.face_mesh)
        self.horizontal_vector.set_landmark_keypoints(*EYE_OUTER_CORNER_KEYPOINTS)

        self.vertical_vector = LandmarkVector(self.face_mesh)
        self.vertical_vector.set_landmark_keypoints(*FACE_CENTER_KEYPOINTS)

    def update_frame(self, frame):
        self.parent_frame.update_frame(frame)
        self.parent_frame.get_stable_mesh()

    def get_stable_mesh(self):
        center_vector = self.parent_frame.horizontal_vector.get_center_coordinates()

        # Rotation
        rotated_landmarks = self.rotate_around_center()
        translated_landmarks = self.translate_to_center(rotated_landmarks - center_vector)

        self.stable_mesh.update_mesh(translated_landmarks)

    def translate_to_center(self, landmarks):
        center_vector = self.parent_frame.horizontal_vector.get_center_coordinates()
        
        displacement_x, displacement_y = self.get_distance_from_center()
        translate_center = translate_points(landmarks, (displacement_x, displacement_y))

        return translate_center + center_vector
    
    def get_distance_from_center(self):
        center_vector = self.parent_frame.horizontal_vector.get_center_coordinates()

        center_frame_x = self.parent_frame.frame_dimensions[0] // 2
        center_frame_y = self.parent_frame.frame_dimensions[1] // 2

        displacement_x = center_frame_x - center_vector[0]
        displacement_y = center_frame_y - center_vector[1]

        return displacement_x, displacement_y

    def rotate_around_center(self):
        center_vector = self.parent_frame.horizontal_vector.get_center_coordinates()

        translate_center = self.parent_frame.face_mesh.get_scaled_landmarks() - center_vector
        rotation_angle = self.parent_frame.get_rotation_angle()
        rotated_landmarks = rotate_points(translate_center, rotation_angle)

        return rotated_landmarks + center_vector


