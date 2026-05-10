import os
import pandas as pd
import joblib

MODEL_PATH = "models/churn_model.pkl"
FEATURES_NAMES_PATH = "models/feature_names.pkl"

def load_artifacts():
    """Load the trained model and feature names from the specified paths."""
    model = joblib.load(MODEL_PATH)
    feature_names = joblib.load(FEATURES_NAMES_PATH)
    return model, feature_names

def create_sample_customer(feature_names):
    """Create one sample customer for prediction."""
    sample_customer = {
        "tenure": 5,
        "MonthlyCharges": 90.0,
        "TotalCharges": 450.0,
        "SeniorCitizen": 0,
        "Contract_Month-to-month": 1,
        "InternetService_Fiber optic": 1,
        "PaymentMethod_Electronic check": 1,
        "PaperlessBilling_Yes": 1,
        "TechSupport_Yes": 0,
        "OnlineSecurity_Yes": 0
    }
    return sample_customer

def prepare_input(sample_customer, feature_names):
    """Prepare the input data for prediction."""
    input_df = pd.DataFrame([sample_customer], columns=feature_names)
    input_df = input_df.reindex(columns=feature_names, fill_value=0)  # Ensure all features are present
    return input_df

def predict_churn(model, input_df):
    """Predict churn for the sample customer."""
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]  # Probability of churn
    return prediction, probability
    
def main():
    print("Loading model and feature names...")
    model, feature_names = load_artifacts()

    print("Creating sample customer...")
    sample_customer = create_sample_customer(feature_names)

    print("Preparing input data...")
    input_df = prepare_input(sample_customer, feature_names)

    print("Predicting churn...")
    prediction, probability = predict_churn(model, input_df)

    print(f"Predicted Churn: {'Yes' if prediction == 1 else 'No'}")
    print(f"Churn Probability: {probability:.4f}")

if __name__ == "__main__":
    main()
