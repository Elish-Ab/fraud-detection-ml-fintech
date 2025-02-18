from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import joblib
import io
import os
import logging
from datetime import datetime
from pydantic import BaseModel
from typing import List

# Initialize FastAPI app
app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up logging
logging.basicConfig(
    filename="server.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Load pre-trained model
def load_model():
    global model
    try:
        model_path = "/app/project/model.pkl" if os.path.exists('/app/project/model.pkl') else "./model.pkl"
        model = joblib.load(model_path)
        logging.info("Model loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
        raise RuntimeError(f"Error loading model: {str(e)}")

# Load model at startup
load_model()

# One-hot encoded country list and other constants
# List of one-hot encoded country columns used during training.
COUNTRIES = ["Albania", "Algeria", "Angola", "Argentina", "Australia", "Austria", "Bangladesh", "Belgium", "Brazil", "Canada", "China", "Denmark", "Egypt", "Finland", "France", "Germany", "Greece", "Hungary", "India", "Indonesia", "Iran", "Iraq", "Ireland", "Israel", "Italy", "Japan", "Kenya", "Malaysia", "Mexico", "Netherlands", "New Zealand", "Nigeria", "Norway", "Pakistan", "Peru", "Philippines", "Poland", "Portugal", "Qatar", "Romania", "Russia", "Saudi Arabia", "South Africa", "South Korea", "Spain", "Sweden", "Switzerland", "Thailand", "Turkey", "Ukraine", "United Arab Emirates", "United Kingdom", "United States", "Vietnam", "Zimbabwe"]
BROWSERS = ['Chrome', 'FireFox', 'IE', 'Opera', 'Safari']
SOURCES = ['Ads', 'Direct', 'Organic']

REQUIRED_COLUMNS = {"user_id", "sex", "signup_time", "purchase_time", "purchase_value", 
                    "device_id", "source", "browser", "age", "ip_address", "country"}

# Preprocessing function
# Preprocessing function
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df = df.copy()
        df['signup_time'] = pd.to_datetime(df['signup_time'])
        df['purchase_time'] = pd.to_datetime(df['purchase_time'])
        df['time_diff'] = (df['purchase_time'] - df['signup_time']).dt.total_seconds()
        df.drop(columns=['signup_time', 'purchase_time'], inplace=True)
        
        # Handle country column (ensure it's in the correct format)
        country_column = f"country_{df['country'].iloc[0].replace(' ', '_')}"  # Ensure no spaces
        for country in COUNTRIES:
            if country == df['country'].iloc[0]:
                df[country_column] = 1
            else:
                df[f"country_{country.replace(' ', '_')}"] = 0

        # Apply one-hot encoding for other categorical features
        df = pd.get_dummies(df, columns=['source', 'browser'], drop_first=True)
        
        # Ensure all required columns are present, even if they have zero values
        for col in REQUIRED_COLUMNS:
            if col not in df.columns:
                df[col] = 0
        
        # Reorder columns to match the model's expected feature order
        missing_cols = set(model.feature_names_in_) - set(df.columns)
        for col in missing_cols:
            df[col] = 0  # Add missing columns with 0 or default values
        
        df = df[model.feature_names_in_]
        
        logging.info("Data preprocessing completed successfully.")
        return df
    except Exception as e:
        logging.error(f"Error in data preprocessing: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Preprocessing error: {str(e)}")


# Response model
class PredictionResponse(BaseModel):
    predictions: List[float]
    fraud_probabilities: List[float]
    status: str

# File upload prediction
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
            raise HTTPException(400, detail=f"Missing columns: {missing_columns}")
        
        processed_data = preprocess_data(data)
        predictions = model.predict(processed_data)
        probabilities = model.predict_proba(processed_data)[:, 1]
        
        logging.info("Prediction completed successfully for uploaded file.")
        return PredictionResponse(predictions=predictions.tolist(), fraud_probabilities=probabilities.tolist(), status="success")
    
    except HTTPException as e:
        logging.error(f"Client error: {e.detail}")
        return JSONResponse(status_code=e.status_code, content={"status": "error", "detail": e.detail})
    except Exception as e:
        logging.error(f"Internal server error: {str(e)}")
        return JSONResponse(status_code=500, content={"status": "error", "detail": "Internal Server Error", "message": str(e)})

# Form-based prediction
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
        
        logging.info("Form prediction completed successfully.")
        return PredictionResponse(predictions=[float(prediction)], fraud_probabilities=[float(probability)], status="success")
    
    except HTTPException as e:
        logging.error(f"Client error: {e.detail}")
        return JSONResponse(status_code=e.status_code, content={"status": "error", "detail": e.detail})
    except Exception as e:
        logging.error(f"Internal server error: {str(e)}")
        return JSONResponse(status_code=500, content={"status": "error", "detail": "Internal Server Error", "message": str(e)})

# Run FastAPI server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
