import pytest
import pandas as pd
from scripts.data_cleaning import load_data, handle_missing_values, remove_duplicates, correct_data_types

def test_load_data():
    # Test loading data from a CSV file
    df = load_data("../data/Fraud_Data.csv")
    assert not df.empty  # Check if DataFrame is not empty

def test_handle_missing_values():
    # Test handling missing values
    sample_data = {
        'feature1': [1, 2, None, 4],
        'feature2': [5, None, 7, 8]
    }
    df = pd.DataFrame(sample_data)
    cleaned_df = handle_missing_values(df)
    assert cleaned_df.shape[0] == 2  # Check if rows with missing values are dropped

def test_remove_duplicates():
    # Test removing duplicates
    sample_data = {
        'feature1': [1, 2, 2, 4],
        'feature2': [5, 6, 6, 8]
    }
    df = pd.DataFrame(sample_data)
    cleaned_df = remove_duplicates(df)
    assert cleaned_df.shape[0] == 3  # Check if duplicates are removed

def test_correct_data_types():
    # Test correcting data types
    sample_data = {
        'signup_time': ['2021-01-01 00:00:00', '2021-01-02 00:00:00'],
        'purchase_time': ['2021-01-01 01:00:00', '2021-01-02 01:00:00']
    }
    df = pd.DataFrame(sample_data)
    corrected_df = correct_data_types(df)
    assert 'signup_year' in corrected_df.columns  # Check if new column is created

if __name__ == "__main__":
    pytest.main()
