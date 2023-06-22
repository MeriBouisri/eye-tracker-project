from abc import ABC, abstractmethod
import logging
import os

from eye_tracking.utils.savefile import SaveFile, SaveFileError, SaveFileAlreadyExistsError

logger = logging.getLogger('Calibration')

class CalibrationError(Exception):
    """Base class for calibration-related errors."""

class Calibration(ABC):
    _current_dir = os.path.dirname(__file__)
    _savefile_folder = 'savefiles'

    def __init__(self, savefile_name):
        self._savefile_name = savefile_name
        self._savefile_dir = os.path.join(self._current_dir, self._savefile_folder)
        self._savefile = SaveFile(self._savefile_name, self._savefile_dir)



    @abstractmethod
    def calibrate(self):
        """
        Implement the calibration process for child classes.
        """
        pass

    @abstractmethod
    def is_valid_data(self, property_key, property_value) -> bool:
        """
        Implement custom verification for each key of a given data dict.
        """
        pass
    
    @abstractmethod
    def is_valid_calibration_data(self, data: dict) -> bool:
        """
        Verify that the all the incoming data is valid before saving to file.
        This method must iterate through the data and verify that each property is valid.
        """
        pass

    def is_calibrated(self):
        return self.get_calibration_data() is not None
        
    def save_calibration_data(self, check_savefile_already_exists=True, **kwargs):
        """
        Verify that the incoming data is valid before saving to file.

        Parameters
        ----------
        check_savefile_already_exists : bool, optional = True
            If True, check if the savefile already exists and contains valid data and prompts user to overwrite the file.
            If False, overwrite the savefile without checking.
        
        **kwargs
            Arbitrary keyword arguments. Each keyword argument is a property of the calibration data.

        Returns
        -------
        successfully_saved : bool
            True if the calibration data was successfully saved to file.

        Raises
        ----------
        SaveFileAlreadyExistsError
            Only raises if check_savefile_already_exists is True and the savefile already exists and contains valid data.
        """

        successfully_saved = False
        data_dict = {key: value for key, value in kwargs.items()}

        # Check if calibration data already exists
        if check_savefile_already_exists:
            if self.get_calibration_data() is not None:
                raise SaveFileAlreadyExistsError(self._savefile_name)
            
        try:
            self.is_valid_calibration_data(data_dict)
        except CalibrationError as err:
            self.handle_calibration_error(err)
            return False

        while not successfully_saved:
            try:
                successfully_saved = self._savefile.write_savefile_data(data_dict)

            except SaveFileError as err:
                error_successfully_handled = self._savefile.handle_savefile_error(err)

                if not error_successfully_handled:
                    logger.error(f'Couldn\'t save calibration data: {err}')
                    return False

        return True

    def get_calibration_data(self):
        """
        Verify that the outgoing data from savefile is valid before returning to caller.
        """
        savefile_data = None

        try:
            savefile_data = self._savefile.load_savefile_data()
        except SaveFileError as err:
            self._savefile.handle_savefile_error(err)
            return None
            
        try:    
            self.is_valid_calibration_data(savefile_data)
        except CalibrationError as err:
            self.handle_calibration_error(err)
            return None
        
        return savefile_data
    
    # ==============================
    # ERROR HANDLING
    # ==============================

    def handle_calibration_error(self, error: Exception) -> bool:
        """
        No implementation for other possible CalibrationErrors yet.
        Might not be needed.
        """
        if isinstance(error, SaveFileError):
            return self._savefile.handle_savefile_error(error)
        
        elif isinstance(error, CalibrationError):
            logger.error(error)
            return False
        
        else:
            logger.error(error)
            return False



    

