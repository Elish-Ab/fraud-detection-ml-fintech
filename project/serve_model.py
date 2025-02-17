from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import joblib
import io
import os
from datetime import datetime
from pydantic import BaseModel
from typing import List

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load pre-trained model
def load_model():
    global model
    try:
        model_path = "/app/project/model.pkl" if os.path.exists('/app/project/model.pkl') else "./model.pkl"
        model = joblib.load(model_path)
    except Exception as e:
        raise RuntimeError(f"Error loading model: {str(e)}")

# Load model at startup
load_model()

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
BROWSERS = ['Chrome', 'FireFox', 'IE', 'Opera', 'Safari']
SOURCES = ['Ads', 'Direct', 'Organic']

REQUIRED_COLUMNS = {"user_id", "sex", "signup_time", "purchase_time", "purchase_value", 
                    "device_id", "source", "browser", "age", "ip_address", "country"}

class PredictionResponse(BaseModel):
    predictions: List[float]
    fraud_probabilities: List[float]
    status: str

@app.post("/predict/upload", response_model=PredictionResponse)
async def predict_from_file(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        file_extension = os.path.splitext(file.filename)[-1].lower()
        if file_extension not in {".csv", ".json"}:
            raise HTTPException(400, detail="Only CSV and JSON files are supported")
        
        data = pd.read_csv(io.BytesIO(contents)) if file_extension == '.csv' else pd.read_json(io.BytesIO(contents))
        
        missing_columns = REQUIRED_COLUMNS - set(data.columns)
        if missing_columns:
            raise HTTPException(status_code=400, detail=f"Missing columns: {missing_columns}")
        
        processed_data = preprocess_data(data)
        predictions = model.predict(processed_data)
        probabilities = model.predict_proba(processed_data)[:, 1]
        
        return PredictionResponse(
            predictions=predictions.tolist(),
            fraud_probabilities=probabilities.tolist(),
            status="success"
        )
        
    except HTTPException as e:
        return JSONResponse(status_code=e.status_code, content={"status": "error", "detail": e.detail})
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "detail": "Internal Server Error", "message": str(e)})

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
        data = pd.DataFrame([{ "user_id": user_id, "sex": sex, "signup_time": signup_time, "purchase_time": purchase_time, "purchase_value": purchase_value, "device_id": device_id, "source": source, "browser": browser, "age": age, "ip_address": ip_address, "country": country }])
        processed_data = preprocess_data(data)
        prediction = model.predict(processed_data)[0]
        probability = model.predict_proba(processed_data)[0][1]
        
        return PredictionResponse(
            predictions=[float(prediction)],
            fraud_probabilities=[float(probability)],
            status="success"
        )
        
    except HTTPException as e:
        return JSONResponse(status_code=e.status_code, content={"status": "error", "detail": e.detail})
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "detail": "Internal Server Error", "message": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
