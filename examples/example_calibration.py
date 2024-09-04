from eye_tracking.calibration.camera_calibration import CameraCalibration

def example_camera_calibration():
    """
    Calibrate the camera by holding a chessboard pattern image.
    Save the calibration data to a file for later use.
    """
    camera_calibration = CameraCalibration()

    # Begin camera calibration process if the camera has not been calibrated yet.
    if not camera_calibration.is_calibrated():
        camera_calibration.calibrate()

    # Retrieve the saved calibration data.
    data = camera_calibration.get_calibration_data()

    # Use the data :)
    
    print('Calibration data:')
    print(f'Camera matrix : {data["camera_matrix"]}')
    print(f'Distortion coefficients : {data["dist_coeffs"]}')

def example_gaze_vector_calibration():
    raise NotImplementedError("This example has not been implemented yet.")