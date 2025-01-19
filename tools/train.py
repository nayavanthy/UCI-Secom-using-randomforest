# Internal Imports
from utils.file_io import load_data
from data.data_preparation import preprocess_data
from models.random_forest import build_random_forest
from utils.under_sampling import undersample
from utils.significant_test import find_significant_column
from configs.model_config import General, SaveName

class Train:
    def __init__(self, input_path):
        self.input_path = input_path

    def run(self):
        """
        Description: Follows the flow of data from loading to modeling
        :return: dictionary containing all processing info
        """
        # Load data
        data = load_data(self.input_path)

        # Split target and independent variables
        features = data.drop(columns=General.TARGET_COLUMN)
        target = data[General.TARGET_COLUMN]

        # Data preparation
        features, processing_info = preprocess_data(dataframe=features)

        # Find significant column
        features = find_significant_column(features, target)

        # Find significant columns
        significant_columns= features.columns

        # Undersample data
        features, target = undersample(features, target)

        # Build model
        model = build_random_forest()
        model.fit(features, target)

        # Add to processing info
        additional_info = {
            SaveName.SIGNIFICANT_COLUMNS : significant_columns,
            SaveName.MODEL : model
        }

        processing_info.update(additional_info)

        return processing_info
