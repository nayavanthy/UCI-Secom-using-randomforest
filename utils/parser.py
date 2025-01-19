# External Imports
from configparser import ConfigParser

# internal Imports
from tools.exceptions import ConfigKeyError

def get_config_value(file_path, section, key):
    """
    Read a value from a config.ini file.

    :param file_path: Path to the config.ini file
    :param section: Section in the config file
    :param key: Key within the section to retrieve the value for
    :return: Value associated with the provided section and key
    """
    config = ConfigParser()
    config.read(file_path)

    if section in config and key in config[section]:
        return config[section][key]
    else:
        raise ConfigKeyError(section, key)
