# External Imports
from pickle import dump
from pandas import read_csv
from os.path import isfile

# Internal Imports
from tools.exceptions import DataNotFoundError, InvalidDataFormatError
from configs.model_config import General

def load_data(filepath):
    """
    Load data from a CSV file using pandas.

    :param filepath: Path to the CSV file.
    :return: DataFrame containing the loaded data.
    """
    # Check if file path exists
    if not isfile(filepath):
        raise DataNotFoundError(filepath, __file__)
    # Check if data is in valid format
    if not filepath.endswith(General.DATA_FORMAT):
        raise InvalidDataFormatError(filepath, __file__)
    return read_csv(filepath)


def save_artifact(obj, filepath):
    """
    Save an object to a file using pickle.

    :param obj: Object to be serialized and saved.
    :param filepath: Path to the file where the object will be saved.
    """
    # Check if given file path is in the correct format"
    if not filepath.endswith(General.SAVE_FORMAT):
        raise InvalidDataFormatError(filepath, __file__)

    with open(filepath, 'wb') as f:
            dump(obj, f)
