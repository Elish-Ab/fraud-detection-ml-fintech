from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import joblib
import io
import numpy as np
from datetime import datetime
from pydantic import BaseModel
from typing import List, Union
import os
app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load pre-trained model
try:
    if os.path.exists('/app/project/model.pkl'):  # Docker container path
        model_path = '/app/project/model.pkl'
    else:  # Local path
        model_path = './model.pkl'

    model = joblib.load(model_path)
except Exception as e:
    raise RuntimeError(f"Error loading model: {str(e)}")


# Country list from your dataset columns
# List of one-hot encoded country columns used during training.
COUNTRIES = [
            "Albania", "Algeria", "Angola", "Antigua and Barbuda", "Argentina", "Armenia", "Australia", "Austria", "Azerbaijan", "Bahamas",
            "Bahrain", "Bangladesh", "Barbados", "Belarus", "Belgium", "Belize", "Benin", "Bermuda", "Bhutan", "Bolivia", "Bonaire",
            "Bosnia and Herzegowina", "Botswana", "Brazil", "British Indian Ocean Territory", "Brunei Darussalam", "Bulgaria", "Burkina Faso",
            "Burundi", "Cambodia", "Cameroon", "Canada", "Cape Verde", "Cayman Islands", "Chile", "China", "Colombia", "Congo", "Costa Rica",
            "Cote D'ivoire", "Croatia", "Cuba", "Curacao", "Cyprus", "Czech Republic", "Denmark", "Djibouti", "Dominica", "Dominican Republic",
            "Ecuador", "Egypt", "El Salvador", "Estonia", "Ethiopia", "European Union", "Faroe Islands", "Fiji", "Finland", "France", "Gabon",
            "Gambia", "Georgia", "Germany", "Ghana", "Gibraltar", "Greece", "Guadeloupe", "Guam", "Guatemala", "Haiti", "Honduras", "Hong Kong",
            "Hungary", "Iceland", "India", "Indonesia", "Iran", "Iraq", "Ireland", "Israel", "Italy", "Jamaica", "Japan", "Jordan", "Kazakhstan",
            "Kenya", "Kiribati", "Korea", "Kuwait", "Kyrgyzstan", "Laos", "Latvia", "Lebanon", "Lesotho", "Liberia", "Libya", "Liechtenstein",
            "Lithuania", "Luxembourg", "Macau", "Madagascar", "Malawi", "Malaysia", "Maldives", "Mali", "Malta", "Martinique", "Mauritania",
            "Mauritius", "Mexico", "Micronesia (Federated States of)", "Moldova", "Monaco", "Mongolia", "Montenegro", "Montserrat", "Morocco",
            "Mozambique", "Namibia", "Nauru", "Nepal", "Netherlands", "New Zealand", "Nicaragua", "Niger", "Nigeria", "Niue", "Norfolk Island",
            "North Macedonia", "Northern Mariana Islands", "Norway", "Oman", "Pakistan", "Palau", "Panama", "Papua New Guinea", "Paraguay", "Peru",
            "Philippines", "Pitcairn", "Poland", "Portugal", "Puerto Rico", "Qatar", "Romania", "Russian Federation", "Rwanda", "Reunion", "Saint Barthelemy",
            "Saint Helena", "Saint Kitts and Nevis", "Saint Lucia", "Saint Martin", "Saint Pierre and Miquelon", "Saint Vincent and the Grenadines", "Samoa",
            "San Marino", "Sao Tome and Principe", "Saudi Arabia", "Senegal", "Serbia", "Seychelles", "Sierra Leone", "Singapore", "Sint Maarten", "Slovakia",
            "Slovenia", "Solomon Islands", "Somalia", "South Africa", "South Georgia and the South Sandwich Islands", "Spain", "Sri Lanka", "Sudan", "Suriname",
            "Svalbard and Jan Mayen", "Sweden", "Switzerland", "Syrian Arab Republic", "Taiwan", "Tajikistan", "Tanzania", "Thailand", "Timor-Leste", "Togo",
            "Tokelau", "Tonga", "Trinidad and Tobago", "Tunisia", "Turkey", "Turkmenistan", "Tuvalu", "Uganda", "Ukraine", "United Arab Emirates", "United Kingdom",
            "United States of America", "Uruguay", "Uzbekistan", "Vanuatu", "Venezuela", "Viet Nam", "Western Sahara", "Yemen", "Zambia", "Zimbabwe"
]

class PredictionResponse(BaseModel):
    predictions: List[float]
    fraud_probabilities: List[float]
    status: str

def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """Preprocess input data for model consumption"""
    data = data.copy()
    
    # Convert datetime fields
    data['signup_time'] = pd.to_datetime(data['signup_time'], unit='s')
    data['purchase_time'] = pd.to_datetime(data['purchase_time'], unit='s')
    
    # Extract temporal features
    data['signup_hour'] = data['signup_time'].dt.hour
    data['purchase_hour'] = data['purchase_time'].dt.hour
    data['time_diff_signup_purchase'] = (data['purchase_time'] - data['signup_time']).dt.total_seconds()
    
    # One-hot encode country
    for country in COUNTRIES:
        data[f'country_{country}'] = 0
    data[f'country_{data["country"].iloc[0]}'] = 1
    
    # Encode other categorical variables
    categorical_mappings = {
        'sex': {'male': 1, 'female': 0},
        'source': {'Ads': 0, 'Direct': 1, 'Organic': 2},
        'browser': {'Chrome': 0, 'FireFox': 1, 'IE': 2, 'Opera': 3, 'Safari': 4}
    }
    
    for col, mapping in categorical_mappings.items():
        data[col] = data[col].map(mapping)
    
    # Select final features based on your model's requirements
    required_features = [
        'purchase_value', 'age', 'signup_hour', 'purchase_hour',
        'time_diff_signup_purchase', 'sex', 'source', 'browser'
    ] + [f'country_{c}' for c in COUNTRIES]
    
    return data[required_features]

@app.post("/predict/upload", response_model=PredictionResponse)
async def predict_from_file(file: UploadFile = File(...)):
    try:
        # Read and parse uploaded file
        contents = await file.read()
        if file.filename.endswith('.csv'):
            data = pd.read_csv(io.BytesIO(contents))
        elif file.filename.endswith('.json'):
            data = pd.read_json(io.BytesIO(contents))
        else:
            raise HTTPException(400, "Unsupported file format")
        
        # Preprocess data
        processed_data = preprocess_data(data)
        
        # Make predictions
        predictions = model.predict(processed_data)
        probabilities = model.predict_proba(processed_data)[:, 1]
        
        return {
            "predictions": predictions.tolist(),
            "fraud_probabilities": probabilities.tolist(),
            "status": "success"
        }
        
    except Exception as e:
        raise HTTPException(500, f"Processing error: {str(e)}")

@app.post("/predict/form", response_model=PredictionResponse)
async def predict_from_form(
    user_id: str = Form(...),
    sex: str = Form(...),
    signup_time: float = Form(...),
    purchase_time: float = Form(...),
    purchase_value: float = Form(...),
    device_id: str = Form(...),
    source: str = Form(...),
    browser: str = Form(...),
    age: int = Form(...),
    ip_address: str = Form(...),
    country: str = Form(...)
):
    try:
        # Create DataFrame from form data
        form_data = {
            'user_id': [user_id],
            'sex': [sex],
            'signup_time': [signup_time],
            'purchase_time': [purchase_time],
            'purchase_value': [purchase_value],
            'device_id': [device_id],
            'source': [source],
            'browser': [browser],
            'age': [age],
            'ip_address': [ip_address],
            'country': [country]
        }
        
        data = pd.DataFrame(form_data)
        processed_data = preprocess_data(data)
        
        # Make predictions
        prediction = model.predict(processed_data)[0]
        probability = model.predict_proba(processed_data)[0][1]
        
        return {
            "predictions": [float(prediction)],
            "fraud_probabilities": [float(probability)],
            "status": "success"
        }
        
    except Exception as e:
        raise HTTPException(500, f"Form processing error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)