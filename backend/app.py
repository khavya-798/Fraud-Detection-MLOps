# app.py - Corrected Real-Time Fraud Prediction Service

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import RedirectResponse # Import the RedirectResponse tool

# --- 1. PYDANTIC INPUT SCHEMA ---
class TransactionData(BaseModel):
    Time: float
    V1: float; V2: float; V3: float; V4: float; V5: float
    V6: float; V7: float; V8: float; V9: float; V10: float
    V11: float; V12: float; V13: float; V14: float; V15: float
    V16: float; V17: float; V18: float; V19: float; V20: float
    V21: float; V22: float; V23: float; V24: float; V25: float
    V26: float; V27: float; V28: float
    Amount: float

# --- 2. APP INITIALIZATION & MODEL LOADING ---
app = FastAPI(title="Real-Time Fraud Detection API")

MODEL_PATH = 'models/credit_fraud.pkl'
SCALER_PATH = 'models/scaler.pkl'
MODEL = None
SCALER = None

try:
    MODEL = joblib.load(MODEL_PATH)
    SCALER = joblib.load(SCALER_PATH)
    print("Model & Scaler loaded successfully.")
except Exception as e:
    print(f"CRITICAL ERROR: Could not load model or scaler. Details: {e}")

# --- FIX: ADD A ROOT ENDPOINT TO REDIRECT USERS ---
@app.get("/")
def read_root():
    """
    This endpoint redirects the user from the root URL ('/')
    to the API documentation page ('/docs').
    """
    return RedirectResponse(url="/docs")

# --- 3. PREDICTION ENDPOINT ---
@app.post("/predict")
def predict_fraud(data: TransactionData):
    
    if MODEL is None or SCALER is None:
        raise HTTPException(status_code=500, detail="Model or scaler not loaded.")

    try:
        input_data_dict = data.dict()
        input_df = pd.DataFrame([input_data_dict])
        
        features_to_scale = [col for col in input_df.columns if col != 'Time']
        scaled_features = SCALER.transform(input_df[features_to_scale])
        
        prediction = MODEL.predict(scaled_features)
        is_fraud_value = int(prediction[0])

        return {
            "is_fraud": is_fraud_value
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error during prediction: {str(e)}")
