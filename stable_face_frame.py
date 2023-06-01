from face_frame import FaceFrame
 
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

    def update_frame(self, frame):
        self.parent_frame.update_frame(frame)
        self.parent_frame.get_stable_mesh()