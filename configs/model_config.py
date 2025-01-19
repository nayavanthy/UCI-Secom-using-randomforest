# Parameters for null value imputation
class DataPrep:
    K_NEIGHBOURS = 5
    NULL_THRESHOLD = 0.3
    VARIANCE_THRESHOLD = 0
    VARIANCE_VARIABLE = 'std'

# Parameters for random forest model
class Model:
    N_ESTIMATORS = 100
    RANDOM_SEED = 42
    CRITERION = 'entropy'
    MAX_DEPTH = 40
    MIN_SAMPLES_SPLIT = 5
    MIN_SAMPLES_LEAF = 1

# Name of files where objects will be saved
class SaveName:
    IMPUTER = 'imputer.pkl'
    SCALER = 'scaler.pkl'
    NULL_COLUMNS= 'non_null_columns.pkl'
    CONSTANT_COLUMNS = 'non_constant_columns.pkl'
    MODEL = 'random_forest_model.pkl'
    SIGNIFICANT_COLUMNS = 'significant_columns.pkl'

# Common constants
class General:
    TARGET_COLUMN= 'col_Pass_Fail'
    PASS = -1
    FAIL = 1
    DATA_FORMAT = '.csv'
    SAVE_FORMAT = '.pkl'

# Parameters for signficance tests
class SignificanceTest:
    ALPHA = 0.05
    TTEST_USEVAR = 'pooled'
    BINS = 30
