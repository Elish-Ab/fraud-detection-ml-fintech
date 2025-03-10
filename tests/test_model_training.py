import pytest
import pandas as pd
from model_training import prepare_data, evaluate_model, create_mlp_model_keras

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

if __name__ == "__main__":
    pytest.main()
