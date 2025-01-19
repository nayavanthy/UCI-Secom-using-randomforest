# External Imports
from pandas import concat, cut, crosstab
from statsmodels.stats.weightstats import ttest_ind
from scipy.stats import mannwhitneyu, chi2_contingency

# Internal Imports
from configs.model_config import General, SignificanceTest

def find_significant_column(features, target, alpha = SignificanceTest.ALPHA):
    """
    Find significant columns based on statistical tests.

    :param features: Features DataFrame.
    :param target: Target variable Series.
    :param alpha: Significance level for tests (default is 0.05).
    :return: DataFrame with significant columns
    """
    dataframe = concat([features, target], axis=1)
    # Initialize empty list
    significant_columns = []
    # Split dataset into pass data and fail data
    pass_data = dataframe[dataframe[General.TARGET_COLUMN] == General.PASS]
    fail_data = dataframe[dataframe[General.TARGET_COLUMN] == General.FAIL]

    # For each column conduct signficance tests
    for column in dataframe.columns:
        if column != General.TARGET_COLUMN:
            _ , ttest_p_value , _ = ttest_ind(pass_data[column], fail_data[column], usevar = SignificanceTest.TTEST_USEVAR)
            _ , mwu_p_value = mannwhitneyu(pass_data[column], fail_data[column])
            contingency_table = crosstab(dataframe[General.TARGET_COLUMN], cut(dataframe[column], bins = SignificanceTest.BINS))
            _ , chi2_p_value , _ , _ = chi2_contingency(contingency_table)

            # If column passes any test, append it list of significant columns
            if ttest_p_value < alpha or mwu_p_value < alpha or chi2_p_value < alpha:
                significant_columns.append(column)

    dataframe = dataframe[significant_columns]
    return dataframe
