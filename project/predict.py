#!/usr/bin/env python
"""
predict.py

This script loads the Random Forest model (logged with MLflow during training)
and performs a prediction on a sample row from the specified CSV file.

Usage:
    For the credit card dataset:
        python predict.py --dataset creditcard --csv ../data/creditcard.csv

    For the fraud (e-commerce) dataset:
        python predict.py --dataset fraud --csv ../data/Cleaned_Fraud_Data.csv
"""

import argparse
import mlflow
import mlflow.sklearn
import pandas as pd

def load_model():
    """
    Sets the MLflow tracking URI and loads the Random Forest model from a specific run.
    Update the run_id below to match the run in which your Random Forest model was logged.
    """
    mlflow.set_tracking_uri("http://localhost:5000")
    
    # Replace the run_id below with your actual run ID from which the model was logged.
    run_id = "91cc0b1d550047049946bf60e9cfdda2"  
    model_uri = f"runs:/{run_id}/Random Forest"
    print(f"Loading model from {model_uri} ...")
    
    model = mlflow.sklearn.load_model(model_uri)
    print("Model loaded successfully.")
    return model

def normalize_columns(df):
    """
    Normalize column names by stripping whitespace and removing extra quotes.
    This ensures that the feature names match those used during training.
    """
    df.columns = df.columns.str.strip().str.replace('"', '')
    return df

def predict_creditcard(model, csv_file):
    """
    Loads the credit card CSV file, normalizes column names, drops the target column ("Class"),
    selects the first row, and performs a prediction.
    """
    df = pd.read_csv(csv_file)
    df = normalize_columns(df)
    
    # For the credit card dataset, drop the target column "Class" if it exists.
    if "Class" in df.columns:
        features = df.drop(columns=["Class"])
    else:
        features = df

    # Use the first row as a sample.
    sample = features.iloc[[0]]
    print("Feature names for prediction (creditcard):", sample.columns.tolist())
    
    # Make a prediction.
    prediction = model.predict(sample)
    print("Credit Card sample prediction:", prediction[0])

def predict_fraud(model, csv_file):
    """
    Loads the fraud (e-commerce) CSV file, normalizes column names, drops the target column ("class"),
    selects the first row, and performs a prediction.
    """
    df = pd.read_csv(csv_file)
    df = normalize_columns(df)
    
    # For the fraud dataset, drop the target column "class" if it exists.
    if "class" in df.columns:
        features = df.drop(columns=["class"])
    else:
        features = df

    # Use the first row as a sample.
    sample = features.iloc[[0]]
    print("Feature names for prediction (fraud):", sample.columns.tolist())
    
    # Make a prediction.
    prediction = model.predict(sample)
    print("Fraud sample prediction:", prediction[0])

def main():
    parser = argparse.ArgumentParser(
        description="Run a prediction using the Random Forest model loaded from MLflow."
    )
    parser.add_argument(
        "--dataset",
        choices=["creditcard", "fraud"],
        required=True,
        help="Type of dataset to use for prediction: 'creditcard' or 'fraud'."
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="Path to the CSV file containing sample data for prediction."
    )
    
    args = parser.parse_args()
    
    # Load the Random Forest model.
    model = load_model()
    
    # Depending on the dataset type, run the appropriate prediction function.
    if args.dataset == "creditcard":
        predict_creditcard(model, args.csv)
    else:
        predict_fraud(model, args.csv)

if __name__ == "__main__":
    main()
