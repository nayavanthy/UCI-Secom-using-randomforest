"""
Author: Dimaag-AI
Date: July 2024
Description: This code is used to train a machine learning model to predict pass/fail in UCI SECOM dataset.

Version: 1.0
License: DIMAAG-AI Inc, Fremont, CA, USA.
This Dimaag-AI Code is authorized for internal use at DIMAAG-AI Inc only, Copyright (c) 2018-2024. All rights reserved.
Contact: info@dimaag.ai
"""

# External Imports
from argparse import ArgumentParser
from os import _exit
from sys import exc_info
from os.path import basename
from traceback import extract_tb

# Internal Imports
from utils.parser import get_config_value
from tools.exceptions import (
    ConfigKeyError,
    DataNotFoundError,
    InvalidDataFormatError
)
try:
    from tools.main import main
except ImportError as e:
    _ , _ , exc_traceback = exc_info()
    error_file = basename(extract_tb(exc_traceback)[-1][0])
    print(f"Error caught in file {error_file}, error:- {e}, Exiting")
    _exit(0)

if __name__ == "__main__":
    # Getting the required arguments for the application.
    parser = ArgumentParser(description="Sample application demonstrating better code readability.")

    try:
        parser.add_argument("--mode",
                            default=get_config_value("./configs/general.ini", 'GENERAL', 'MODE'),
                            help="Mode to run the code (train/test)")

        parser.add_argument("--data_path",
                            default=get_config_value("./configs/general.ini", 'PATH', 'TRAIN_DIR'),
                            help="Path of the main folder for training/testing")

        parser.add_argument("--save_path",
                            default=get_config_value("./configs/general.ini", 'PATH', 'SAVE_DIR'),
                            help="Model path to save the model.",
                            required=False)
        parser.add_argument("--save_mode",
                            default=get_config_value("./configs/general.ini", 'GENERAL', 'SAVE_MODE'),
                            help="Mode to save model or not",
                            required=False)

    except ConfigKeyError as exp:
        print(exp)
        _exit(0)

    # Parse arguments
    args = parser.parse_args()

    # Call the main function
    try:
        if main(args=args):
            print("Modelling Succesful")
    except (DataNotFoundError, InvalidDataFormatError) as e:
        print(e)
        _exit(0)
    except AttributeError as e:
        _ , _ , exc_traceback = exc_info()
        error_file = basename(extract_tb(exc_traceback)[-1][0])
        print(f"Error caught in file {error_file}, error:- {e}, Exiting")
        _exit(0)
    except Exception as e:
        print(f"Unexpected error: {e}. Exiting.")
        _exit(0)
