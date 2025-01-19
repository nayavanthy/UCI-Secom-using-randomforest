# External Imports
from imblearn.under_sampling import RandomUnderSampler

# Internal Imports
from configs.model_config import Model

def undersample(features, target):
    """
    Undersample the majority class using RandomUnderSampler.

    :param X: Features DataFrame or array.
    :param y: Target variable Series or array.
    :return: Resampled X and y.
    """
    rus = RandomUnderSampler(random_state= Model.RANDOM_SEED)
    features_resampled, target_resampled = rus.fit_resample(features, target)
    return features_resampled, target_resampled
