from fastapi import FastAPI, HTTPException
import uvicorn
import pandas as pd
import joblib
from datetime import datetime
from supabase import create_client, Client
import traceback
import os
import sys

# ✅ Supabase credentials
SUPABASE_URL = "https://oehpyaughmlhynmuhzrk.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im9laHB5YXVnaG1saHlubXVoenJrIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDMwMTYxOTUsImV4cCI6MjA1ODU5MjE5NX0.CPMU5-sEy2krskO27pwXB_85XaX6vtC75WxjKhGr3gk"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ✅ Load model
MODEL_PATH = "models/fraud_detection_xgboost.pkl"
try:
    with open(MODEL_PATH, "rb") as file:
        model = joblib.load(file)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    model = None

# ✅ FastAPI app
app = FastAPI()

@app.get("/")
def root():
    return {"message": "✅ Fraud Detection API is live!"}

@app.post("/predict/")
def predict(data: dict):
    try:
        if model is None:
            raise ValueError("Model not loaded")

        amount = data.get("amount")
        if amount is None:
            raise HTTPException(status_code=400, detail="Amount is required.")

        # Calculate features
        amount_ratio = float(amount / 100000 if amount > 0 else 0.00001)
        features = {
            "step": 1,
            "amount": float(amount),
            "isFlaggedFraud": 0,
            "isMerchant": 1,
            "amount_ratio": amount_ratio,
            "type_encoded": 2
        }

        # Predict
        df = pd.DataFrame([features])
        probability = float(model.predict_proba(df)[0][1])
        prediction = int(probability > 0.5)

        # Log to Supabase
        timestamp = datetime.utcnow().isoformat()
        payload = {
            "timestamp": timestamp,
            "amount": float(amount),
            "prediction": prediction,
            "probability": round(probability, 6)
        }

        print("📤 Supabase payload:", payload)
        response = supabase.table("new_logs").insert(payload).execute()
        print("🔍 Supabase response:", response)

        return {
            "fraud_prediction": prediction,
            "fraud_probability": round(probability, 6)
        }

    except Exception as e:
        error_msg = traceback.format_exc()
        print("❌ Error:\n", error_msg, file=sys.stderr)
        raise HTTPException(status_code=500, detail="Prediction error: see console logs")
