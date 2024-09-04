import cv2
import logging
import numpy as np
import json
from .calibration import CalibrationError
from eye_tracking.utils.savefile import SaveFile, SaveFileError, SaveFileAlreadyExistsError

from eye_tracking.calibration.calibration import Calibration, CalibrationError

logger = logging.getLogger('calibration.camera')

class CameraCalibrationError(CalibrationError):
    """Raised for camera calibration related errors"""

class InvalidCameraCalibrationValueError(CameraCalibrationError):
    """Raised when camera calibration values do not respect camera property criteria"""
    def __init__(self, property_key, property_value):
        super().__init__(f'The camera property {property_key} contains invalid value: {property_value}')

class InvalidCameraCalibrationKeyError(CameraCalibrationError):
    """Raised when camera calibration data contains unexpected keys """
    def __init__(self, property_key):
        super().__init__(f'The camera property {property_key} could not be found')

class CameraCalibration(Calibration):
    _savefile_name = 'camera_calibration'

    _cam_properties = ['camera_matrix', 'dist_coeffs']

    _cam_properties_criteria = {
        _cam_properties[0] : {
            'key': 'camera_matrix',
            'shape': (3, 3),
            'dtype': np.float64
        },

        _cam_properties[1] : {
                'key': 'dist_coeffs',
                'shape': (1, 5),
                'dtype': np.float64
            }
        }

    def __init__(self, chessboard_size=(9, 6), capture_limit=20, camera_id=0):
        super().__init__(self._savefile_name)

        self.chessboard_size = chessboard_size
        self.capture_limit = capture_limit
        self.camera_id = camera_id

    def is_valid_data(self, property_key: str, data: dict) -> bool:
        try:
            if property_key not in self._cam_properties:
                raise InvalidCameraCalibrationKeyError(f'Invalid camera property: {property_key}')
            
            if data is None:
                raise InvalidCameraCalibrationValueError(property_key, data)
            
            property_data = np.array(data[property_key])

            if property_data.size == 0:
                raise InvalidCameraCalibrationValueError(property_key, property_data)
            
            elif property_data.shape != self._cam_properties_criteria[property_key]['shape']:
                raise InvalidCameraCalibrationValueError(property_key, property_data.shape)
            
            elif property_data.dtype != self._cam_properties_criteria[property_key]['dtype']:
                raise InvalidCameraCalibrationValueError(property_key, property_data.dtype)
        except CameraCalibrationError as err:
            logger.error(err)
            return False

        return True
    
    def is_valid_calibration_data(self, data: dict) -> bool:
        for key in self._cam_properties:
            try:
                self.is_valid_data(key, data)
            except CameraCalibrationError as err:
                logger.error(err)
                return False
        
        return True

    def calibrate(self):
        logger.info('Starting camera calibration.')

        print('-------------------- CAMERA CALIBRATION --------------------')
        print('Hold up a checkerboard image pattern in front of the camera.')
        print('Try getting different angles and positions of the checkerboard pattern.')
        print('Colored lines will appear on the chessboard pattern, which means that the pattern is detected.')
        print('When you are satisfied with , Press \'c\' on the keyboard to capture the frame.')
        print('Press \'q\' to quit the calibration process.')
        print('------------------------------------------------------------')

        self.video_capture = cv2.VideoCapture(self.camera_id)

        capture_counter = 0

        obj_points = []
        img_points = []

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        objp = np.zeros((1, self.chessboard_size[0] * self.chessboard_size[1], 3), np.float32)
        objp[0,:,:2] = np.mgrid[0:self.chessboard_size[0], 0:self.chessboard_size[1]].T.reshape(-1, 2)

        while capture_counter < self.capture_limit:
            _, frame = self.video_capture.read()

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray_frame, self.chessboard_size, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

            if ret:
                corners2 = cv2.cornerSubPix(gray_frame, corners, (11, 11), (-1, -1), criteria)
                cv2.drawChessboardCorners(frame, self.chessboard_size, corners, ret)

                if cv2.waitKey(1) == ord('c'):
                    obj_points.append(objp)
                    img_points.append(corners2)
                    capture_counter += 1

            cv2.putText(frame, f'Captured: {capture_counter}/{self.capture_limit}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow('frame', frame)

            if cv2.waitKey(1) == ord('q'):
                self.video_capture.release()
                cv2.destroyAllWindows()
                print('Calibration aborted.')
                return
            
        self.video_capture.release()
        cv2.destroyAllWindows()

        print('Calibrating camera.')
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray_frame.shape[::-1], None, None)
        camera_matrix = np.array(mtx).tolist()
        dist_coeffs = np.array(dist).tolist()

        successfully_saved = False

        while not successfully_saved:
            try:
                successfully_saved = self.save_calibration_data(camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
            except (SaveFileError, CalibrationError) as err:
                logger.error(err)
                error_handled_successfully = self.handle_calibration_error(err)

                if not error_handled_successfully:
                    break

        if successfully_saved:

            print('Calibration successful.')

        else:
            print('Calibration failed...')

