import pytest
import pandas as pd
from scripts.model_training import prepare_data, evaluate_model, create_mlp_model_keras, train_all_models

def test_prepare_data():
    # Test data preparation with a sample DataFrame
    sample_data = {
        'feature1': [1, 2, 3, 4],
        'feature2': [5, 6, 7, 8],
        'class': [0, 1, 0, 1]
    }
    df = pd.DataFrame(sample_data)
    X_train, X_test, y_train, y_test = prepare_data(df, 'class')
    assert len(X_train) + len(X_test) == len(df)  # Check if data is split correctly

def test_evaluate_model():
    # Test model evaluation with a mock model and data
    class MockModel:
        def predict(self, X):
            return [0] * len(X)  # Mock predictions

    model = MockModel()
    X_test = pd.DataFrame({'feature1': [1, 2], 'feature2': [5, 6]})
    y_test = [0, 1]
    acc, f1, roc = evaluate_model(model, X_test, y_test)
    assert acc == 0.5  # Check accuracy for mock predictions

def test_create_mlp_model_keras():
    # Test MLP model creation
    input_dim = 10
    model = create_mlp_model_keras(input_dim)
    assert model.input_shape[1] == input_dim  # Check input dimension

def test_train_all_models():
    # Test the train_all_models function with a sample DataFrame
    sample_data = {
        'feature1': [1, 2, 3, 4],
        'feature2': [5, 6, 7, 8],
        'class': [0, 1, 0, 1]
    }
    df = pd.DataFrame(sample_data)
    X_train, X_test, y_train, y_test = prepare_data(df, 'class')
    
    # Call train_all_models with and without SMOTE
    model_with_smote = train_all_models(df, 'class', use_smote=True)
    model_without_smote = train_all_models(df, 'class', use_smote=False)
    
    assert model_with_smote is not None  # Ensure model is trained
    assert model_without_smote is not None  # Ensure model is trained

    # Additional assertions can be added to check model performance
