from abc import ABC, abstractmethod

import logging
import os

import json
import yaml

class UnsupportedSavefileFormatError(Exception):
    def __init__(self, format):
        self.message = f"Unsupported savefile format: {format}"

class CalibrationError(Exception):
    def __init__(self, message):
        self.message = message

class InvalidCalibrationError(Exception):
    def __init__(self, message):
        self.message = message

class EmptySavefileError(Exception):
    def __init__(self, savefile_name):
        self.message = f"The savefile {savefile_name} is empty."

class Calibrator:
    _supported_savefile_formats = ['json', 'yaml', 'yml']
    _current_dir = os.path.dirname(__file__)


    def __init__(self, savefile_basename, savefile_extension):
        self.calibrated = False

        if savefile_extension not in self._supported_savefile_formats:
            raise UnsupportedSavefileFormatError(savefile_extension)
        
        self._savefile_extension = savefile_extension

        self._savefile_name = f"{savefile_basename}.{savefile_extension}"
        self._savefile_path = os.path.join(self._current_dir, self._savefile_name)

        if not os.path.exists(self._savefile_path):
            self._handle_savefile_not_found()

    @abstractmethod
    def calibrate(self):
        pass

    @abstractmethod
    def validate_calibration_data(self):
        pass

    @abstractmethod
    def _format_savefile_data(self):
        pass

    # ========== LOAD SAVEFILE DATA FUNCTIONS ==========

    def load_savefile_data(self):
        """
        Loads the savefile data from the savefile specified by self._savefile_path.
        Handles different savefile formats just in case we disagree on which format to use.

        Returns
        -------
        savefile_data : dict or None
            The data from the savefile. Returns None if an error occurs while loading the savefile.

        Raises
        ------
        UnsupportedSavefileFormatError
            Raised when the savefile format is not supported.
        """
        if self._savefile_extension not in self._supported_savefile_formats:
            raise UnsupportedSavefileFormatError(self._savefile_extension)
        
        if self._savefile_extension == 'json':
            return self._load_json_savefile_data()

        elif self._savefile_extension in ['yaml', 'yml']:
            return self._load_yaml_savefile_data()
        
        return None

    def _load_json_savefile_data(self):
        """
        Loads the json savefile data from the savefile specified by self._savefile_path.
        

        Returns
        -------
        savefile_data : dict or None
            The data from the savefile. Returns None if an error occurs while loading the savefile.
        """
        savefile_data = None

        try:
            with open(self._savefile_path, 'r') as file:
                try:
                    savefile_data = json.load(file)

                except json.JSONDecodeError as e:
                    print(e)
                    self._handle_incorrect_data()
                    return None
                
        except FileNotFoundError:
            print(e)
            self._handle_savefile_not_found()
            return None

        return savefile_data

    def _load_yaml_savefile_data(self):
        """
        Loads the yaml savefile data from the savefile specified by self._savefile_path.

        Returns
        -------
        savefile_data : dict or None
            The data from the savefile. Returns None if an error occurs while loading the savefile.
        """
        savefile_data = None
        
        try:
            with open(self._savefile_path, 'r') as file:
                try:
                    savefile_data = yaml.load(file, Loader=yaml.FullLoader)

                except yaml.YAMLError as e:
                    print(e)
                    self._handle_incorrect_data()
                    return None

        except FileNotFoundError:
            print(e)
            self._handle_savefile_not_found()
            return None

        return savefile_data

    # ========== CREATE SAVEFILE FUNCTIONS ==========

    def _create_empty_savefile(self):
        """
        Creates an empty savefile at the path specified by self._savefile_path.
        Handles different savefile formats just in case we disagree on which format to use.

        Raises
        ------
        UnsupportedSavefileFormatError
            If the savefile format is not supported, as specified by the class attribute _supported_savefile_formats.
        """

        # Always check if savefile format is still correct
        if self._savefile_extension not in self._supported_savefile_formats:
            raise UnsupportedSavefileFormatError(self._savefile_extension)
        
        if self._savefile_extension == 'json':
            self._create_empty_json_savefile()

        elif self._savefile_extension in ['yaml', 'yml']:
            self._create_empty_yaml_savefile()

    def _create_empty_json_savefile(self):
        """
        Creates an empty json savefile at the path specified by this instance.
        """
        empty_data = self._format_savefile_data()
        with open(self._savefile_path, 'w') as file:
            json.dump(empty_data, file, indent=4)

    def _create_empty_yaml_savefile(self):
        """
        Creates an empty yaml savefile at the path specified by this instance.
        """
        empty_data = self._format_savefile_data()
        with open(self._savefile_path, 'w') as file:
            yaml.dump(empty_data, file)


    # ========== EXCEPTION HANDLING FUNCTIONS ==========

    def _handle_savefile_not_found(self):
        """
        Prompts the user to create a new savefile if the savefile is not found.

        Returns
        -------
        error_handled : bool
            True if error was correctly handled, False otherwise.
        """
        prompt = f"Savefile {self._savefile_name} not found. Create new savefile? 1 (yes) / 0 (no) : "
        error_handled = self._prompt_user(prompt, yes_option_callback=self._create_empty_savefile)

        # Return true if error correctly handled
        return error_handled == 1
        
    def _handle_incorrect_data(self):
        """
        Prompts the user to create a new savefile if the savefile contains incorrect data.

        Returns
        -------
        error_handled : bool
            True if error was correctly handled, False otherwise.
        """
        prompt = f"Savefile {self._savefile_name} contains incorrect data. Create new savefile? 1 (yes) / 0 (no) : "
        error_handled = self._prompt_user(prompt, yes_option_callback=self._create_empty_savefile)

        # Return true if error correctly handled
        return error_handled == 1
    
    def _handle_missing_data(self):
        """
        Prompts the user to overwrite the savefile if the savefile is missing data.

        Returns
        -------
        overwrite : bool
            True if user wants to overwrite the savefile, False otherwise.
        """

        prompt = f"Savefile {self._savefile_name} is missing some data. Start calibration session? 1 (yes) / 0 (no) : "
        confirm_overwrite = self._prompt_user(prompt, yes_option_callback=self.calibrate)

        # Return true if confirmed overwrite
        return confirm_overwrite == 1
    
    def _handle_recalibration_overwrite(self):
        """
        Prompts the user to overwrite the savefile if the savefile already contains calibration data. 

        Returns
        -------
        overwrite : bool
            True if user wants to overwrite the savefile, False otherwise.
        """
        prompt = f"Savefile {self._savefile_name} already contains calibration data. Overwrite the savefile? 1 (yes) / 0 (no) : "
        confirm_overwrite = self._prompt_user(prompt, yes_option_callback=self.calibrate)

        # Return true if confirmed overwrite
        return confirm_overwrite == 1
    
    def _handle_empty_savefile(self):
        """
        Prompts the user to create a new savefile if the savefile is empty.

        Returns
        -------
        error_handled : bool
            True if user wants to start calibration, False otherwise.
        """
        prompt = f"Savefile {self._savefile_name} is empty. Start calibration session? 1 (yes) / 0 (no) : "
        error_handled = self._prompt_user(prompt, yes_option_callback=self.calibrate)

        # Return true if error correctly handled
        return error_handled == 1
    
    def _handle_calibration_abort(self):
        print("Calibration aborted.")

    def _prompt_user(self, prompt, yes_option_callback: callable):
        try:
            user_input = int(input(prompt))
        except ValueError:
            user_input = -1

        if user_input == 1:
            yes_option_callback()
            return user_input
        
        elif user_input == 0:
            self._handle_calibration_abort()
            return user_input
        
        print("Invalid input. Please enter 1 (yes) or 0 (no).")
        self._prompt_user(prompt, yes_option_callback=yes_option_callback)
                

        

        



