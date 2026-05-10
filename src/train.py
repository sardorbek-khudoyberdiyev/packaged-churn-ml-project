import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

PROCESSED_DATA_PATH = "data/processed/processed_data.csv"
MODEL_PATH = "models/churn_model.pkl"
FEATURES_NAMES_PATH = "models/feature_names.pkl"

X_TEST_PATH = "data/processed/X_test.csv"
Y_TEST_PATH = "data/processed/y_test.csv"

def load_processed_data(path):
    """Load the processed data from the specified path."""
    return pd.read_csv(path)

def split_features_target(df):
    """Split the DataFrame into features and target variable."""
    X = df.drop(columns=['Churn'])
    y = df['Churn']
    return X, y

def train_model(X_train, y_train):
    '''Train a Random Forest Classifier on the training data.'''
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth = None,
        min_samples_split = 5,
        min_samples_leaf = 1,
        class_weight = 'balanced',
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

def save_artifacts(model, feature_names, X_test, y_test):
    """Save the trained model and feature names to the specified paths."""
    os.makedirs("models", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(feature_names, FEATURES_NAMES_PATH)

    X_test.to_csv(X_TEST_PATH, index=False)
    y_test.to_csv(Y_TEST_PATH, index=False)

    print(f"Model saved to {MODEL_PATH}")
    print(f"Feature names saved to {FEATURES_NAMES_PATH}")
    print(f"X_test saved to {X_TEST_PATH}")
    print(f"y_test saved to {Y_TEST_PATH}") 

def main():
    print("Loading processed data...")
    df = load_processed_data(PROCESSED_DATA_PATH)

    print("Splitting features and target variable...")
    X, y = split_features_target(df)

    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training model...")
    model = train_model(X_train, y_train)

    print("Saving model and feature names...")
    save_artifacts(model, X.columns.tolist(), X_test, y_test)

    print("Training complete.")

if __name__ == "__main__":
    main()