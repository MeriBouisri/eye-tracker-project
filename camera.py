import cv2

class Camera:
    """
    The Camera class is used to get frames from a video capture.
    For now, it is only used to make our usual series of function calls more concise.
    However, I am planning on adding camera calibration and other features to this class so that we can
    change the camera settings more easily, as well as trouble shooting.

    Notes
    ----------
    For more info on camera calibration : https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html

    """
    def __init__(self, camera_id=0):
        """
        Attributes
        ----------
        camera : cv2.VideoCapture
            The video capture used to get frames from the camera.

        Parameters
        ----------
        camera_id : int, optional = 0
            Specifies which camera to use. The default value is 0, which is the default camera on most computers.

        """
        self.camera = cv2.VideoCapture(camera_id)

        # TODO: I forgot what this is
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    def get_frame(self, flip=True):
        """
        Parameters
        ----------
        flip : bool, optional = True
            Specifies whether the frame should be flipped horizontally. This is useful when using the front camera.

        Returns
        ---------- 
            The frame from the video capture.
        """
        _, self.frame = self.camera.read()
        if flip:
            self.frame = cv2.flip(self.frame, 1)
            
        return self.frame

    def release(self):
        """
        Releases the camera and destroys all windows.
        """
        self.camera.release()
        cv2.destroyAllWindows()