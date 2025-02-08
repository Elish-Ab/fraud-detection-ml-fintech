import mlflow.sklearn
import pandas as pd
import json
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the model from MLflow
MODEL_URI = "models:/RandomForestModel/1"  # Replace with your registered model URI
model = mlflow.sklearn.load_model(MODEL_URI)

@app.route('/')
def home():
    return "Fraud Detection Model API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()  # Expecting JSON input
        df = pd.DataFrame(data)  # Convert JSON to DataFrame
        predictions = model.predict(df)  # Make predictions
        return jsonify({'predictions': predictions.tolist()})  # Return as JSON
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
