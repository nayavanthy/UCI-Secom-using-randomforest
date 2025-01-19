import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
import pickle

# Load the list of columns to delete
with open('./artifacts/columns_to_remove.pkl', 'rb') as file:
    columns_to_delete = pickle.load(file)

with open('./artifacts/constant_columns.pkl', 'rb') as file:
    constant = pickle.load(file)

with open('./artifacts/imputer.pkl', 'rb') as file:
    imputer = pickle.load(file)

with open('./artifacts/signficant_columns.pkl', 'rb') as file:
    significant_columns = pickle.load(file)

# Load the scaler
with open('./artifacts/scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Load the random forest model
with open('./artifacts/random_forest_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Function to process the data
def process_and_predict(df):
    # Drop the columns listed in columns_to_delete.txt
    df = df.drop(columns=columns_to_delete, errors='ignore')
    
    # Impute missing values
    df_imputed = imputer.transform(df)

    df_imputed = pd.DataFrame(df_imputed, columns=df.columns)

    df_imputed= df_imputed.drop(columns= constant)
    
    # Scale the data
    df_scaled = scaler.transform(df_imputed)

    df_scaled = pd.DataFrame(df_scaled, columns=df_imputed.columns)
    
    # Select the significant columns
    df_scaled = df_scaled[significant_columns]
    
    # Make predictions
    predictions = model.predict(df_scaled)
    
    return predictions

df = pd.read_csv('./data/uci-secom_test.csv')
X= df.drop(['col_Pass_Fail'], axis=1)
y = df['col_Pass_Fail']
predictions = process_and_predict(X)

print(accuracy_score(y, predictions))
print(confusion_matrix(y, predictions))
print(classification_report(y, predictions))
print(roc_auc_score(y, predictions))

