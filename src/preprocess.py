import os 
import pandas as pd

RAW_DATA_PATH = "data/raw/Telco-Customer-Churn.csv"
PROCESSED_DATA_PATH = "data/processed/processed_data.csv"

def load_data(path):
    """Load the raw data from the specified path."""
    return pd.read_csv(path)

def clean_data(df):
    #Clean and prepare the churn dataset
    df_clean = df.copy()

    #Drop customerID column
    df_clean = df_clean.drop(columns=['customerID'])

    #Convert TotalCharges to numeric, coerce errors to NaN
    df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')

    #Fill missing TotalCharges with median
    df_clean['TotalCharges'] = df_clean['TotalCharges'].fillna(df_clean['TotalCharges'].median())

    #Convert target variable 'Churn' to binary
    df_clean['Churn'] = df_clean['Churn'].map({'Yes': 1, 'No': 0})

    return df_clean

def encoded_features(df):
    #Encode categorical features using one-hot encoding
    X = df.drop(columns=['Churn'])
    y = df['Churn'] 

    X_encoded = pd.get_dummies(X, drop_first=True)
    
    #Convert True/False to 1/0
    X_encoded = X_encoded.astype(int)

    processed_df = X_encoded.copy()
    processed_df['Churn'] = y

    return processed_df

def save_data(df, path):
    """Save the processed data to the specified path."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)

def main():
    #Load raw data
    print("Loading raw data...")
    df = load_data(RAW_DATA_PATH)

    #Clean data
    print("Cleaning data...")
    df_clean = clean_data(df)

    #Encode features
    print("Encoding features...")
    processed_df = encoded_features(df_clean)

    #Save processed data
    print("Saving processed data...")
    save_data(processed_df, PROCESSED_DATA_PATH)

if __name__ == "__main__":
    main()