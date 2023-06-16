from enum import Enum
from utils.nested_dict import NestedDict

# ========== EYE KEYPOINTS ==========

LEFT_EYE_KEYPOINTS = [33, 133, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246, 27, 23]
RIGHT_EYE_KEYPOINTS = [263, 362, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466, 446, 442]

# ========== IRIS KEYPOINTS ==========

LEFT_IRIS_KEYPOINTS = [469, 470, 471, 472]
RIGHT_IRIS_KEYPOINTS = [474, 475, 476, 477]

# ========== CENTER PUPIL KEYPOINTS ==========

LEFT_PUPIL_CENTER_KEYPOINT = 468
RIGHT_PUPIL_CENTER_KEYPOINT = 473

# ========== OUTER CORNER KEYPOINTS ==========

LEFT_OUTER_CORNER_KEYPOINT = 33
RIGHT_OUTER_CORNER_KEYPOINT = 263

class EyeID(Enum):
    """
    The EyeID enum class provides constants for identifying the left and right eye. 
    The purpose of using EyeID constants is to provide type safety and readability 
    when accessing the eye keypoints.
    """
    LEFT = 0
    RIGHT = 1

class EyeDict(NestedDict):
    """
    The EyeDict class provides a more convenient way to access the eye keypoints.

    The eye components are organized in the eye_keypoints dictionary according to their respective side (left or right).

    This class overrides the __getitem__ method to allow for indexing using the EyeID enum for the first level 
    of the eye_keypoints dictionary (EyeID.LEFT or EyeID.RIGHT). 
    If the key is not an EyeID option, the key is passed to the super class NestedDict.

    Properties
    ----------
    left : NestedDict
        A NestedDict instance of key-value pairs nested under the 'left' key.

    right : NestedDict
        A NestedDict instance of key-value pairs nested under the 'right' key.

    Examples 
    ----------
    Instantiate an EyeDict object
    >>> eye_dict = EyeDict()

    The following notations are all equivalent and valid ways to access the same value :
    >>> eye_dict['left']['eye']
    >>> eye_dict.left.eye
    >>> eye_dict[EyeID.LEFT].eye
    >>> eye_dict[EyeID.LEFT]['eye']
    >>> eye_dict[0].eye
    """

    eye_keypoints = {
        'left': {
            'eye': LEFT_EYE_KEYPOINTS,
            'iris': LEFT_IRIS_KEYPOINTS,
            'pupil': LEFT_PUPIL_CENTER_KEYPOINT,
            'outer_corner': LEFT_OUTER_CORNER_KEYPOINT
        },

        'right': {
            'eye': RIGHT_EYE_KEYPOINTS,
            'iris': RIGHT_IRIS_KEYPOINTS,
            'pupil': RIGHT_PUPIL_CENTER_KEYPOINT,
            'outer_corner': RIGHT_OUTER_CORNER_KEYPOINT
        }
    }

    def __init__(self):
        super().__init__(self.eye_keypoints)

    def __getitem__(self, key):
        if isinstance(key, EyeID):
            return super().__getitem__(key.value)

        return super().__getitem__(key)

    @property
    def left(self):
        """
        The dictionary key containing the keypoints of different parts of the left eye.
        """
        return NestedDict(self['left'])
    
    @property
    def right(self):
        """
        Returns a NestedDict instance of the right eye keypoints.
        """
        return NestedDict(self['right'])

# Static instance of the EyeDict class.
eye_dict = EyeDict()






 




    


    


    

    

    
