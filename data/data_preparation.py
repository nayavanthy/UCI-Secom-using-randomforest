# External Imports
from pandas import DataFrame
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

# Internal Imports
from configs.model_config import DataPrep, SaveName


def handle_null_values(dataframe):
    """
    Remove columns with null values above a certain threshold.

    :param dataframe: Input DataFrame.
    :return: DataFrame with columns removed.
    """
    null_ratio = dataframe.isnull().sum() / len(dataframe)
    columns_to_remove = null_ratio[null_ratio > DataPrep.NULL_THRESHOLD].index.tolist()
    dataframe = dataframe.drop(columns=columns_to_remove)
    return dataframe


def impute_missing_values(dataframe):
    """
    Impute missing values using KNN imputation.

    :param dataframe: Input DataFrame.
    :return: DataFrame with imputed values, and KNNImputer object.
    """
    imputer = KNNImputer(n_neighbors=DataPrep.K_NEIGHBOURS)
    imputed_dataframe = DataFrame(imputer.fit_transform(dataframe), columns=dataframe.columns)
    return imputed_dataframe, imputer


def handle_constant_values(dataframe):
    """
    Remove columns with constant values.

    :param dataframe: Input DataFrame.
    :return: DataFrame with constant columns removed.
    """
    summary_stats = dataframe.describe().transpose()
    constant_columns = summary_stats[summary_stats[DataPrep.VARIANCE_VARIABLE] == DataPrep.VARIANCE_THRESHOLD].index.tolist()
    dataframe = dataframe.drop(columns=constant_columns)
    return dataframe


def standardize_data(dataframe):
    """
    Standardize data using StandardScaler.

    :param dataframe: Input DataFrame.
    :return: DataFrame with standardized values, and StandardScaler object.
    """
    scaler = StandardScaler()
    scaled_dataframe = DataFrame(scaler.fit_transform(dataframe), columns=dataframe.columns)
    return scaled_dataframe, scaler


def preprocess_data(dataframe):
    """
    Preprocess data by handling null values, imputing missing values, removing constant columns,
    and standardizing data.

    :param dataframe: Input DataFrame.
    :return: Tuple containing the preprocessed DataFrame and a dictionary with non-null columns,
            imputer, non-constant columns, and scaler.
    """
    dataframe = handle_null_values(dataframe)
    non_null_columns = dataframe.columns

    dataframe, imputer = impute_missing_values(dataframe)

    dataframe = handle_constant_values(dataframe)
    non_constant_columns = dataframe.columns

    dataframe, scaler = standardize_data(dataframe)

    data_pipeline = {
        SaveName.NULL_COLUMNS: non_null_columns,
        SaveName.IMPUTER: imputer,
        SaveName.CONSTANT_COLUMNS: non_constant_columns,
        SaveName.SCALER: scaler
    }

    return dataframe, data_pipeline
