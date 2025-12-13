# --------------------------------------------------------
#  RUL Prediction API (Final Production Version)
# --------------------------------------------------------

from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
from tensorflow import keras
import json
import os
import numpy as np

# --------------------------------------------------------
# Initialize FastAPI
# --------------------------------------------------------
app = FastAPI(
    title="RUL Prediction API",
    description="Predicts Remaining Useful Life (RUL) using XGBoost, Neural Network, and Ensemble.",
    version="1.1"
)

# --------------------------------------------------------
# Resolve correct model directory
# --------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")  # ✅ Correct folder

print("📁 Loading models from:", MODEL_DIR)

# --------------------------------------------------------
# Load models & files
# --------------------------------------------------------
try:
    scaler = joblib.load(os.path.join(MODEL_DIR, "minmax_scaler.pkl"))   # ✅ Correct name
    xgb_model = joblib.load(os.path.join(MODEL_DIR, "xgb_model.pkl"))
    nn_model = keras.models.load_model(os.path.join(MODEL_DIR, "neural_network_rul.keras"))

    with open(os.path.join(MODEL_DIR, "feature_names.json"), "r") as f:
        feature_names = json.load(f)

    print("✅ All models loaded successfully")

except Exception as e:
    print("❌ Error loading models:", e)
    raise e


# --------------------------------------------------------
# Utility: Validate incoming data
# --------------------------------------------------------
def validate_input(data: dict):
    """Ensure all required features are provided."""
    missing = [f for f in feature_names if f not in data]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing features: {missing[:10]} ... ({len(missing)} total missing)"
        )


# --------------------------------------------------------
# Prediction Endpoint
# --------------------------------------------------------
@app.post("/predict_rul")
def predict_rul_api(data: dict):
    """
    Input:
        JSON with all numerical features.
    Output:
        XGBoost RUL, Neural Net RUL, Ensemble RUL
    """

    # Validate input
    validate_input(data)

    # Rebuild as DataFrame with correct column ordering
    df = pd.DataFrame([[data[f] for f in feature_names]], columns=feature_names)

    # Scale input
    try:
        X_scaled = scaler.transform(df)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Scaling failed: {str(e)}")

    # Predictions
    try:
        xgb_pred = float(xgb_model.predict(X_scaled)[0])
        nn_pred = float(nn_model.predict(X_scaled, verbose=0)[0][0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {str(e)}")

    # Ensemble
    ensemble = (xgb_pred + nn_pred) / 2

    # Output response
    return {
        "xgb_rul": round(xgb_pred, 2),
        "nn_rul": round(nn_pred, 2),
        "ensemble_rul": round(ensemble, 2)
    }


# --------------------------------------------------------
# Root Endpoint
# --------------------------------------------------------
@app.get("/")
def home():
    return {"message": "RUL Prediction API is running! Visit /docs to test."}
