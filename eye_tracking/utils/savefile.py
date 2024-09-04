import os
import logging
import sys
import json

from eye_tracking.utils.user_input import get_user_input

logger = logging.getLogger('EyeTracker.SaveFile')

class SaveFileError(Exception):
    """Base class for file-saving related errors."""

class SaveFileNotFoundError(SaveFileError):
    """Raised when a savefile is not found at the specified path."""
    def __init__(self, savefile):
        super().__init__(f'Savefile not found: {savefile}')

class SaveFileDirectoryNotFoundError(SaveFileError):
    """Raised when a savefile directory is not found at the specified path."""
    def __init__(self, savefile_dir):
        super().__init__(f'Savefile directory not found: {savefile_dir}')

class InvalidSaveFileDataError(SaveFileError):
    """Raised when a savefile is found but the data is invalid."""
    def __init__(self, savefile):
        super().__init__(f'Invalid savefile data in: {savefile}')

class EmptySaveFileError(SaveFileError):
    """Raised when attempting to load data from an empty savefile."""
    def __init__(self, savefile):
        super().__init__(f'Savefile is empty: {savefile}')

class SaveFileAlreadyExistsError(SaveFileError):
    """Raised when attempting to write new data to a savefile that already contains valid data"""
    def __init__(self, savefile):
        super().__init__(f'Savefile already exists: {savefile}')

class SaveFile:

    def __init__(self, savefile_basename, savefile_dir):
        """
        Raises
        ----------
        SaveFileDirectoryNotFoundError
            If the savefile directory does not exist and could not be created. 
            Raised only if user chooses not to create the directory.

        SaveFileNotFoundError
            If the savefile does not exist and could not be created.
            Raised only if user chooses not to create the savefile.
        """
        self._savefile_dir = savefile_dir

        if not os.path.exists(self._savefile_dir):
            error_handled_successfully = self.handle_savefile_error(SaveFileDirectoryNotFoundError(self._savefile_dir))

            if not error_handled_successfully:
                logger.error(f'Couldn\'t create savefile directory: {self._savefile_dir}')
                raise SaveFileDirectoryNotFoundError(self._savefile_dir)

        self._savefile_name = f'{savefile_basename}.json'
        self._savefile_path = os.path.join(self._savefile_dir, self._savefile_name)

        if not os.path.exists(self._savefile_path):
            error_handled_successfully = self.handle_savefile_error(SaveFileNotFoundError(self._savefile_name))

            if not error_handled_successfully:
                logger.error(f'Couldn\'t create savefile: {self._savefile_path}')
                raise SaveFileNotFoundError(self._savefile_path)

    def load_savefile_data(self):
        """
        Raises
        ----------
        SaveFileNotFoundError
            If the savefile is not found.

        InvalidSaveFileDataError
            If the savefile data is invalid (Can't be decoded as json).

        EmptySaveFileError
            If the savefile is empty.
        """
        data = None

        try:
            with open(self._savefile_path, 'r') as savefile:
                try:
                    data = json.load(savefile)

                except json.decoder.JSONDecodeError as err:
                    logger.error(err)
                    raise InvalidSaveFileDataError(self._savefile_path)
                
                if not data:
                    raise EmptySaveFileError(self._savefile_path)
                
        except FileNotFoundError as err:
            logger.error(err.strerror)
            raise SaveFileNotFoundError(self._savefile_path)
        
        return data
        
    def write_savefile_data(self, data):
        with open(self._savefile_path, 'w') as savefile:
            try:
                json.dump(data, savefile)
            except TypeError as err:
                logger.error(err)
                raise InvalidSaveFileDataError(self._savefile_path)
            
        return True
            

    # ===============
    # ERROR HANDLING
    # ===============

    def handle_savefile_error(self, error: SaveFileError):
        if isinstance(error, SaveFileNotFoundError):
            logger.warning(error)
            return self._handle_savefile_not_found_error()

        elif isinstance(error, SaveFileDirectoryNotFoundError):
            logger.warning(error)
            return self._handle_savefile_directory_not_found_error()

        elif isinstance(error, EmptySaveFileError):
            logger.warning(error)
            return self._handle_empty_savefile_error()

        elif isinstance(error, SaveFileAlreadyExistsError):
            logger.warning(error)
            return self._handle_savefile_already_exists_error()
        
        elif isinstance(error, InvalidSaveFileDataError):
            logger.warning(error)
            return self._handle_invalid_savefile_data_error()

    def _handle_savefile_not_found_error(self) -> bool:
        """
        Returns
        ----------
        error_handled_successfully : bool
            True if a new file was created, False otherwise.
        """
        prompt = f'Create new savefile: {self._savefile_name}?'
        create_new_savefile = get_user_input(prompt)

        if create_new_savefile:
            logger.info(f'Creating new savefile: {self._savefile_path}')
            self.write_savefile_data({})
            return True

        return False   

    def _handle_savefile_directory_not_found_error(self) -> bool:
        """
        Returns
        ----------
        error_handled_successfully : bool
            True if a new folder was created, False otherwise.
        """
        prompt = f'Create new folder directory: {self._savefile_dir}?'
        create_new_folder = get_user_input(prompt)

        if create_new_folder:
            logger.info(f'Creating new folder directory: {self._savefile_dir}')
            os.mkdir(self._savefile_dir)
            return True
        
        return False

    def _handle_empty_savefile_error(self) -> bool:
        """
        Returns
        ----------
        error_handled_successfully : bool
            False.
        """
        # I don't think theres a situation where the error can be handled successfully enough to proceed.
        # Might not be necessary like the other error handling methods
        return False
    
    def _handle_savefile_already_exists_error(self) -> bool:
        """
        Returns
        ----------
        confirm_overwrite_savefile : bool
            True if the user chooses to overwrite the savefile, False otherwise.
        """
        prompt = f'Overwrite savefile: {self._savefile_name}?'
        confirm_overwrite_savefile = get_user_input(prompt)

        if confirm_overwrite_savefile:
            logger.info(f'Overwriting savefile: {self._savefile_name}')
            self.write_savefile_data({})
            return True

        return False
    
    def _handle_invalid_savefile_data_error(self) -> bool:
        prompt = f'Erase invalid savefile: {self._savefile_name}?'
        confirm_overwrite_savefile = get_user_input(prompt)

        if confirm_overwrite_savefile:
            logger.info(f'Erasing invalid data in savefile: {self._savefile_name}')
            self.write_savefile_data({})
            return True
        
        return False
        



        

        
    
