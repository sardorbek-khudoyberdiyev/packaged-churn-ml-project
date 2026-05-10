import os 
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

MODEL_PATH = "models/churn_model.pkl"
X_TEST_PATH = "data/processed/X_test.csv"
Y_TEST_PATH = "data/processed/y_test.csv"
REPORT_PATH = "reports/evaluation_report.txt"

def load_artifacts():
    """Load the trained model and test data from the specified paths."""
    model = joblib.load(MODEL_PATH)
    X_test = pd.read_csv(X_TEST_PATH)
    y_test = pd.read_csv(Y_TEST_PATH).squeeze()  # Convert to Series
    return model, X_test, y_test

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    conf_matrix = confusion_matrix(y_test, predictions)
    class_report = classification_report(y_test, predictions)

    return accuracy, precision, recall, f1, conf_matrix, class_report

def save_report(accuracy, precision, recall, f1, conf_matrix, class_report):
    """Save the evaluation report to a text file."""
    os.makedirs("reports", exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        f.write("Model Evaluation Report\n")
        f.write("=======================\n\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(f"{conf_matrix}\n\n")
        f.write("Classification Report:\n")
        f.write(f"{class_report}\n")

def main():
    print("Loading model and test data...")
    model, X_test, y_test = load_artifacts()

    print("Evaluating model...")
    accuracy, precision, recall, f1, conf_matrix, class_report = evaluate_model(model, X_test, y_test)

    print("Evaluation results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)
    print("Classification Report:")
    print(class_report)

    print("Saving evaluation report...")
    save_report(accuracy, precision, recall, f1, conf_matrix, class_report)

    print(f"Evaluation report saved to {REPORT_PATH}")

if __name__ == "__main__":
    main()

