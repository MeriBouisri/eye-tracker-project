from . import calibrator

import cv2
import numpy as np
import logging
import json

class CameraCalibrationError(calibrator.CalibrationError):
    def __init__(self, message):
        super().__init__(message)

class InvalidCameraPropertyError(CameraCalibrationError):
    def __init__(self, property_name, attribute, attribute_value):
        super().__init__(f'The camera property {property_name} has an invalid {attribute} of {attribute_value}.')

class NoCameraPropertyFoundError(CameraCalibrationError):
    def __init__(self, property_name):
        super().__init__(f'The camera property {property_name} was not found.')

class EmptyCameraPropertiesFileError(CameraCalibrationError):
    def __init__(self, savefile_name):
        super().__init__(f'The camera properties savefile {savefile_name} is empty.')

class AlreadyCalibratedError(CameraCalibrationError):
    def __init__(self, savefile_name):
        super().__init__(f'A calibration savefile {savefile_name} already exists.')


class CameraCalibrator(calibrator.Calibrator):
    """
    The CameraCalibrator class is used to calculate the extrinsic and intrinsic camera parameters. 
    These parameters can then be saved to a file in the same directory as the CameraCalibrator class.

    See Also
    ----------
    For more information on the camera calibration process, see the OpenCV documentation: https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html

    """

    logger = logging.getLogger(__name__)

    _cam_properties_file_basename = 'camera_properties'
    _cam_properties_file_extension = 'json'

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

    def __init__(self, camera_id, chessboard_size=(9, 6), capture_limit=20):
        super().__init__(self._cam_properties_file_basename, self._cam_properties_file_extension)

        self.camera_id = camera_id
        self.chessboard_size = chessboard_size
        self.capture_limit = capture_limit

    def deserialize_calibration_data(self):
        calibration_data = []

        try:
            for property_key in self._cam_properties_criteria:
                calibration_data.append(self._validate_camera_property(property_key))

        except CameraCalibrationError as e:
            self._handle_camera_calibration_error(e)
            return None
        
        if len(calibration_data) == 0:
            return None
        
        return calibration_data 
    
    def serialize_calibration_data(self, **kwargs):
        try:
            with open(self._savefile_path, 'r') as file:
                savefile = json.load(file)
        except FileNotFoundError:
            if not self._handle_savefile_not_found():
                return False
            
        try:
            for key, value in kwargs.items():
                savefile[key] = np.array(value).tolist()
        except KeyError:
            self.logger.error('Invalid key: \'{}\''.format(key))
            return False

        try:
            with open(self._savefile_path, 'w') as file:
                json.dump(savefile, file, indent=4)
        except FileNotFoundError:
            if not self._handle_savefile_not_found():
                return False
        
        return True
    
    def _format_savefile_data(self):
        """
        
        """
        data_format = {}

        for property_key in self._cam_properties_criteria:
            data_format[property_key] = {}

        return data_format
    
    # ========== ERROR HANDLING ==========
        
    def _handle_camera_calibration_error(self, e):
        if isinstance(e, InvalidCameraPropertyError):
            self.logger.error(e.message)
            return self._handle_incorrect_data()

        elif isinstance(e, NoCameraPropertyFoundError):
            self.logger.error(e.message)
            return self._handle_missing_data()

        elif isinstance(e, EmptyCameraPropertiesFileError):
            self.logger.error(e.message)
            return self._handle_empty_savefile()

        else:
            self.logger.error('An unknown error occurred.')
            return False
        
    def _validate_calibration(self):
        try:
            for property_key in self._cam_properties_criteria:
                self._validate_camera_property(property_key)
        except CameraCalibrationError as e:
            return False
        
        return True

    def _validate_camera_property(self, property_key):
        """
        Returns the camera property with the given key, while checking if the property is valid according to the self._cam_properties_criteria .
        Code adapted from https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html .

        Parameters
        ----------
        property_key : str
            The key of the camera property to be returned. Must be one of the keys in self._cam_properties_criteria .

        Returns
        -------
        property_data : np.ndarray
            The camera property with the given key.

        Raises
        ------
        EmptyCameraPropertiesFileError
            If the camera properties file is empty.

        InvalidCameraPropertyError
            If the camera property with the given key is invalid according to the self._cam_properties_criteria .

        NoCameraPropertyFoundError
            If the camera property with the given key is not found in the savefile.
        """
        criteria = self._cam_properties_criteria[property_key]
        data = self.load_savefile_data()

        if data is None:
            raise EmptyCameraPropertiesFileError(self._savefile_name)
        
        try:
            property_data = np.array(data[property_key])

        except KeyError:
            raise NoCameraPropertyFoundError(property_key)
        
        if property_data.size == 0:
            raise InvalidCameraPropertyError(property_key, 'size', property_data.size)
        
        if property_data.shape != criteria['shape']:
            raise InvalidCameraPropertyError(property_key, 'shape', property_data.shape)
        
        if property_data.dtype != criteria['dtype']:
            raise InvalidCameraPropertyError(property_key, 'data type', property_data.dtype)

        return property_data
    
    def calibrate(self):
        self.logger.info('Starting camera calibration.')
        
        if self._validate_calibration():
            self.logger.info('Camera is already calibrated.')
            
            if not self._handle_recalibration_overwrite():
                return None
            
        self.logger.info('Starting camera calibration.')

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

        print('Calibrating camera...')
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray_frame.shape[::-1], None, None)

        if self.serialize_calibration_data(camera_matrix=mtx, dist_coeffs=dist):
            print('Calibration complete. Camera properties saved to \'{}\''.format(self._savefile_path))

        else:
            print('Calibration failed...')
    


        


            
            



        

        


        




