from eye_tracking.calibration.camera_calibrator import CameraCalibrator

def main():
    """
    Calibrate the camera by holding a chessboard pattern image.
    Save the calibration data to a file for later use.
    """
    camera_calibrator = CameraCalibrator(0)
    camera_calibrator.calibrate()

    data = camera_calibrator.deserialize_calibration_data()

    print('Calibration data:')
    print(f'Camera matrix: {data[0]}')
    print(f'Distortion coefficients: {data[1]}')