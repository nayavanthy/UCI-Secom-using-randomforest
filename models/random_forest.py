# External Imports
from sklearn.ensemble import RandomForestClassifier

# Internal Imports
from configs.model_config import Model


def build_random_forest():
    """
    Build a RandomForestClassifier using parameters from Model configuration.

    :return: RandomForestClassifier instance.
    """
    return RandomForestClassifier(
        n_estimators=Model.N_ESTIMATORS,
        random_state=Model.RANDOM_SEED,
        criterion=Model.CRITERION,
        max_depth=Model.MAX_DEPTH,
        min_samples_leaf=Model.MIN_SAMPLES_LEAF,
        min_samples_split=Model.MIN_SAMPLES_SPLIT
    )
