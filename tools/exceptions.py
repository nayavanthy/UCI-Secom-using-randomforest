class ConfigKeyError(Exception):
    """Exception raised for errors in the configuration file key access."""

    def __init__(self, section, key, message="Key not found in section"):
        self.section = section
        self.key = key
        self.message = f"{message}: [{section}] {key}"
        super().__init__(self.message)


class DataNotFoundError(Exception):
    """Exception raised when data is not found in file path"""

    def __init__(self, file_path, file):
        self.file_path = file_path
        self.file = file
        self.message = f"Exception occured in '{file}' Training Data not found in file path '{self.file_path}'"
        super().__init__(self.message)

    def __str__(self):
        return f'DataNotFoundError: {self.message}'

class InvalidDataFormatError(Exception):
    """Exception raised when the data format is invalid."""

    def __init__(self, file_name, file):
        self.file_name = file_name
        self.message = f"Exception occured in '{file}', Invalid data format for file '{file_name}'."
        super().__init__(self.message)

    def __str__(self):
        return f'InvalidDataFormatError: {self.message}'
