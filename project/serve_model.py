import os
import json
import logging
import joblib
import numpy as np
from flask import Flask, request, jsonify
from sklearn.preprocessing import StandardScaler

# Initialize the Flask application
app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the trained model (Ensure your model is in the same directory or provide the correct path)
MODEL_PATH = './model.pkl'  # Update this path if necessary
model = joblib.load(MODEL_PATH)
scaler = StandardScaler()

@app.route('/')
def index():
    return "Welcome to the Fraud Detection API!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        data = request.get_json()

        # Extract features from the request
        features = data['features']

        # Log the incoming request
        logger.info(f"Incoming request with features: {features}")

        # Preprocessing - Assuming the features are a list of numbers
        features = np.array(features).reshape(1, -1)
        scaled_features = scaler.fit_transform(features)  # Apply scaling if needed

        # Make prediction
        prediction = model.predict(scaled_features)
        
        # Log the prediction
        logger.info(f"Prediction result: {prediction[0]}")

        # Return prediction result
        response = {'prediction': int(prediction[0])}
        return jsonify(response)

    except Exception as e:
        # Log the error if something goes wrong
        logger.error(f"Error during prediction: {str(e)}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # Run the Flask application
    app.run(debug=True, host='0.0.0.0', port=5000)
