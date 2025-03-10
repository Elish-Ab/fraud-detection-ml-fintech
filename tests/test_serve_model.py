import pytest
from fastapi.testclient import TestClient
from serve_model import app, load_model, preprocess_data, PredictionResponse

client = TestClient(app)

def test_load_model():
    # Test if the model loads correctly
    try:
        load_model()
        assert True  # If no exception is raised, the model loads successfully
    except Exception:
        assert False, "Model failed to load"

def test_preprocess_data():
    # Test preprocessing with a sample DataFrame
    sample_data = {
        'user_id': [1, 2],
        'sex': ['M', 'F'],
        'signup_time': ['2021-01-01 00:00:00', '2021-01-02 00:00:00'],
        'purchase_time': ['2021-01-01 01:00:00', '2021-01-02 01:00:00'],
        'purchase_value': [100, 200],
        'device_id': ['device1', 'device2'],
        'source': ['Ads', 'Direct'],
        'browser': ['Chrome', 'Firefox'],
        'age': [30, 25],
        'ip_address': ['192.168.1.1', '192.168.1.2'],
        'country': ['USA', 'Canada']
    }
    df = pd.DataFrame(sample_data)
    processed_df = preprocess_data(df)
    assert 'time_diff' in processed_df.columns  # Check if new column is created

def test_predict_from_file():
    # Test prediction from uploaded file
    with open("test_data.csv", "w") as f:
        f.write("user_id,sex,signup_time,purchase_time,purchase_value,device_id,source,browser,age,ip_address,country\n")
        f.write("1,M,2021-01-01 00:00:00,2021-01-01 01:00:00,100,device1,Ads,Chrome,30,192.168.1.1,USA\n")
    
    with open("test_data.csv", "rb") as f:
        response = client.post("/predict/upload", files={"file": f})
        assert response.status_code == 200
        assert isinstance(response.json(), PredictionResponse)  # Check if response is of correct type

def test_predict_from_form():
    # Test prediction from form data
    response = client.post("/predict/form", data={
        "user_id": "1",
        "sex": "M",
        "signup_time": "2021-01-01 00:00:00",
        "purchase_time": "2021-01-01 01:00:00",
        "purchase_value": "100",
        "device_id": "device1",
        "source": "Ads",
        "browser": "Chrome",
        "age": "30",
        "ip_address": "192.168.1.1",
        "country": "USA"
    })
    assert response.status_code == 200
    assert isinstance(response.json(), PredictionResponse)  # Check if response is of correct type

if __name__ == "__main__":
    pytest.main()
